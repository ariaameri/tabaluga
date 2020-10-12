from .preprocess import PreprocessManager, Preprocess
from .image import BWHCToBCWH, OneHotDecoder, ImageAugmentationAlbumentations
from .image import ImageTransformationPyTorch
import albumentations as A
from .pyTorch import ToTorchTensor, ToTorchGPU
from ...util.config import ConfigParser
from typing import Dict
import numpy as np
import torch
from torchvision import transforms
import cv2


class BackgroundToColor(Preprocess):
    """Converts image channels of the form [0, ..., 0] to 255 in one of the channels.
    In other words, turns background pixels to a color

    """

    def __init__(self, config: ConfigParser):
        """Initializes the class instance.

        Parameters
        ----------
        config : ConfigParser
            Contains the config needed including:
                axis : int, optional
                    The axis showing the channels. If not given will be the default value of -1
                new_channel : int, optional
                    The number of the new channel to have value. If not given, -1 will be assumed

        """

        super().__init__(config)

        # Set the axis
        self.axis = config.get_or_else('axis', -1)

        # Set the new channel to be filled
        self.new_channel = config.get_or_else('new_channel', -1)

    def process(self, data: np.ndarray) -> np.ndarray:
        """"Converts background pixels containing all zeros to 255 in some channel

        Parameters
        ----------
        data : np.ndarray
            A numpy array containing the data with some background, all-zero pixels

        Returns
        -------
        Numpy array of the data with the new_channel filled for the background

        """

        # Find the background pixels
        background = np.all(data == 0, axis=self.axis)

        # Fill the background with 255
        output = data.copy()
        dim_count = len(output.shape)
        prefix_count = self.axis if self.axis >= 0 else (dim_count + self.axis)
        suffix_count = dim_count - 1 - prefix_count
        output[(slice(None), ) * prefix_count + (self.new_channel, ) + (slice(None), ) * suffix_count] \
            = background * 255

        return output


class NecessaryTransformationalAugmentation(ImageAugmentationAlbumentations):
    """Class to perform some necessary augmentations using Albumentations for preprocessing the data."""

    def __init__(self, config: ConfigParser):
        """Initializes the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)

    def build_transformer(self) -> A.Compose:
        """Creates the transformer for the albumentations package for necessary image augmentation as preprocessing.

        Returns
        -------
        An instance of the albumentations.Compose class to do the image augmentations.

        """

        # Resize
        resizer = A.Resize(
            *self._config.get('resize.destination_image_size'),
            interpolation=self._config.get_or_else('interpolation', cv2.INTER_AREA)
        )

        # Divide all pixels by 255 and return float32
        normalizer = A.ToFloat(max_value=255)

        compose = A.Compose([resizer, normalizer])

        return compose

    def process(self, data: np.ndarray) -> np.ndarray:
        """To make life easier, we assume only a single item is given as input and it is iterable.

        Parameters
        ----------
        data : np.ndarray
            A single numpy array consisting of at least 1 batch

        Returns
        -------
        numpy array of the result

        """

        result = np.array([self.transformer(image=item)['image'] for item in data])

        return result


class TransformationsPyTorch(ImageTransformationPyTorch):
    """Abstract class for image transformations using torchvision.transforms."""

    def __init__(self, config: ConfigParser):
        """Initializes the pre-process instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)

        # Build the transformer
        self.transformer: transforms.Compose = self.build_transformer()

    def build_transformer(self) -> transforms.Compose:
        """Abstract method to create the transformer for the torchvision.transforms library for image transformation.

        Returns
        -------
        An instance of torchvision.transforms.Compose class to do the image transformation.

        """

        # Convert to torch tensor, from the form (W, H, C) to tensor of from (C, W, H)
        to_tensor = transforms.ToTensor()

        # Create the transformer object
        transformer = transforms.Compose([to_tensor])

        return transformer

    def process(self, data: np.ndarray):
        """To make life easier, we assume only a single item is given as input and it is iterable.

        Parameters
        ----------
        data : np.ndarray
            A single numpy array consisting of at least 1 batch

        Returns
        -------
        numpy array of the result

        """

        result = torch.stack([self.transformer(item) for item in data], 0)

        return result


class SampleImagePreprocessManager(PreprocessManager):
    """A simple class to manage Preprocess instances."""

    def __init__(self, config: ConfigParser):
        """Initializer.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance and its workers.

        """

        super().__init__(config)

        self.create_workers()

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image preprocess manager'
    #
    #     return string

    def create_workers(self):
        """Creates Preprocess instances."""

        self.workers['labels_background_to_color'] = BackgroundToColor(ConfigParser())

        self.workers['necessary_functional_aug'] = NecessaryTransformationalAugmentation(
            self._config.filter({'_bc': {'$regex': r'(resize|normalize)$'}})
        )

        self.workers['pytorch_transforms'] = TransformationsPyTorch(ConfigParser())

        self.workers['label_one_hot_decoder'] = OneHotDecoder(ConfigParser({"axis": -1}))

        self.workers['to_torch_tensor'] = ToTorchTensor(ConfigParser())

        self.workers['to_torch_gpu'] = ToTorchGPU(ConfigParser())

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, process the loaded train image data."""

        # data = info['data']['data']
        data = info['data']

        processed_data = \
            data\
                .update_map({'_bc': {'$regex': 'labels$'}}, self.workers['labels_background_to_color'].process)\
                .update_map({}, [
                        self.workers['necessary_functional_aug'].process,
                    ])\
                .update_map({'_bc': {'$regex': 'labels$'}}, self.workers['label_one_hot_decoder'].process)\
                .update({}, {'$map':
                        {'data': self.workers['pytorch_transforms'].process,
                         'labels': lambda x: self.workers['to_torch_tensor'].process(x, dtype=torch.long)}}
                    )\
                .update_map({}, self.workers['to_torch_gpu'].process)

        return processed_data

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, process the loaded val image data."""

        # Validation and train preprocessings are the same
        processed_data = self.on_batch_begin(info)

        return processed_data
