from .process import Process, ProcessManager
from ..util.config import ConfigParser
from typing import List, Union, Dict
import numpy as np
import cv2
import albumentations as A
from abc import abstractmethod, ABC


class ImageNormalizer(Process):
    """Normalizes images.

    It takes input images of dtype uint8 ranging from 0 to 255 and divides them by 255 with dtype of float.
    """

    def __init__(self):
        """Initializes the class."""

        super().__init__(ConfigParser())

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image normalizer by dividing by 255.'
    #
    #     return string

    def process(self, data: np.ndarray) -> np.ndarray:
        """"Normalizes the images given.

        It takes input images of dtype uint8 ranging from 0 to 255 and divides them by 255 with dtype of float.

        Parameters
        ----------
        data : np.ndarray
            A numpy array containing the image data of dtype uint8

        Returns
        -------
        Numpy array of image data of dtype float

        """

        # Get the type of the iterable
        collection_type = type(data)

        if collection_type == np.ndarray:
            output = data / 255.
        else:
            raise Exception('Unrecognized format passed to the image normalizer!')

        return output


class ImageResizer(Process):
    """Resizes images."""

    def __init__(self, config: ConfigParser):
        """Initializes the class instance.

        Parameters
        ----------
        config : ConfigParser
            Contains the config needed including:
                destination_image_size : (int, int)
                    The image size to resize images to.
                interpolation, optional
                    The type of interpolation to use

        """

        self.destination_image_size = config.get('destination_image_size')

        self.interpolation = config.get_or_else('interpolation', cv2.INTER_AREA)

        super().__init__(config)

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image resizer to {self.destination_image_size}'
    #
    #     return string

    def process(self, data: np.ndarray) -> np.ndarray:
        """"Resizes the images given.

        Parameters
        ----------
        data : np.ndarray
            A numpy array containing the image to be resized

        Returns
        -------
        Numpy array of resized image data

        """

        # Get the type of the iterable
        collection_type = type(data)

        output = [
            cv2.resize(
                image,
                dsize=self.destination_image_size,
                interpolation=self.interpolation
            )
            for image
            in data
        ]

        if collection_type == np.ndarray:
            output = np.array(output)
        else:
            raise Exception('Unrecognized format passed to the image resizer!')

        return output


class ImageAugmentationAlbumentations(Process, ABC):
    """Abstract class for image augmentation using the albumentations package."""

    def __init__(self, config: ConfigParser = None):
        """Initializes the pre-process instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)

        # Build the Albumentations transformer
        self.transformer: A.Compose = self.build_transformer()

    @abstractmethod
    def build_transformer(self) -> A.Compose:
        """Abstract method to create the transformer for the albumentations package for image augmentation.

        Returns
        -------
        An instance of the albumentations.Compose class to do the image augmentations.

        """

        pass

    def process(self, **data) -> Dict:
        """Transforms/augment the input images.

        Parameters
        ----------
        **data
            Images to be augmented. They should be passed with their specific keywords

        Returns
        -------
        The augmented images with the same keywords as passed.

        """

        # Do the augmentations
        aug_images = self.transformer(**data)

        return aug_images


class ImageResizerWithKeypoints(ImageAugmentationAlbumentations):
    """Resizes images and their corresponding keypoints."""

    def __init__(self, config: ConfigParser, keypoint_format: str):
        """Initializes the class instance.

        Parameters
        ----------
        config : ConfigParser
            Contains the config needed including:
                destination_image_size : (int, int)
                    The image size to resize images to.
                interpolation, optional
                    The type of interpolation to use
        keypoint_format : str
            The format of the keypoint to be used according to Albumentations library

        """

        self.destination_image_size = config.get('destination_image_size')

        self.interpolation = config.get_or_else('interpolation', cv2.INTER_AREA)

        self.keypoint_format = keypoint_format

        super().__init__(config)

    def build_transformer(self) -> A.Compose:
        """Creates the resize transformer using the albumentations package.

        Returns
        -------
        The instance of the albumentations.Compose class to do the image and keypoint resize.

        """

        transformer = A.Compose(
            [A.Resize(
                width=self.destination_image_size[0],
                height=self.destination_image_size[1],
                interpolation=self.interpolation
            )],
            keypoint_params=A.KeypointParams(format=self.keypoint_format)
        )

        return transformer

    def process(self, *, keypoints, **data) -> Dict:
        """"Resizes the images given.

        Parameters
        ----------
        keypoints : List[Union[List, Tuple]]
            The keypoints to be resized
        data
            Images to be resized passed with keywords, each a np.ndarray

        Returns
        -------
        The return of the results using Albumentations package compose

        """

        # Get the results
        aug_images = super().process(keypoints=keypoints, **data)

        return aug_images


# TODO: add/look Augmentor package as well
# TODO: add/look imgaug package as well

# TODO: Move some of these classes to preprocess.py?


class BWHCToBCWH(Process):
    """Converts image data of form (B, W, H, [...,] C) to (B, C, W, H[, ...])."""

    def __init__(self):
        """Initializes the class."""

        super().__init__(ConfigParser())

    def process(self, data: np.ndarray) -> np.ndarray:
        """"Converts image data of form (B, W, H, [...,] C) to (B, C, W, H[, ...]).

        Parameters
        ----------
        data : np.ndarray
            A numpy array containing the image data of form (B, W, H, [...,] C)

        Returns
        -------
        Numpy array of image data of form (B, C, W, H[, ...])

        """

        # Do the rolling
        output = np.rollaxis(data, -1, 1)

        return output


class OneHotDecoder(Process):
    """Converts one-hot-encoded data to index data. Also, makes index values from the biggest number in an axis"""

    def __init__(self, config: ConfigParser):
        """Initializes the class instance.

        Parameters
        ----------
        config : ConfigParser
            Contains the config needed including:
                axis : int, optional
                    The axis along which the decoding should happen. If not given, the default is -1

        """

        super().__init__(config)

        # Set the axis
        self.axis = config.get_or_else('axis', -1)

    def process(self, data: np.ndarray, axis: int = None) -> np.ndarray:
        """"Does the decoding.

        Converts one-hot-encoded data to index data.
        Also, makes index values/data from the biggest number in an axis.

        Parameters
        ----------
        data : np.ndarray
            A numpy array containing the (pseudo-)one-hot data of the shape ([A, ...,] C[, B...])
        axis : int, optional
            The axis along which the decoding should happen and would override self.axis, optional

        Returns
        -------
        Numpy array of decoded data of the shape ([A, ...,][ B...])

        """

        # Find the axis along which we should decode
        axis = axis if axis is not None else self.axis

        # Do the decoding based on argmax
        output = np.argmax(data, axis=axis)

        return output


class BackgroundToColor(Process):
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

        In other words:
        - look for all black pixels along axis of `self.axis`
        - in the same axis, choose channel `self.channel` and set all its value to 255
        For example, if we have a `data` of shape (100, 200, 3) and `axis` and `new_channel` are -1 and -2, then we look
        at the pixels that are all 0 along each of the 3 channels, along the last axis, there are 100 * 200 of such set
        of pixels.
        Then, we take (:, :, -2) and set them all equal to 255

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
        output[(slice(None),) * prefix_count + (self.new_channel,) + (slice(None),) * suffix_count] \
            = background * 255

        return output
