from .preprocess import PreprocessManager, Preprocess
from .image import ImageNormalizer, ImageResizer, BWHCToBCWH, OneHotDecoder
from ...util.config import ConfigParser
from typing import Dict
import numpy as np


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

        self.workers['image_resizer'] = ImageResizer(self._config.get('resize'))

        self.workers['image_normalizer'] = ImageNormalizer()

        self.workers['image_bwhc_to_bcwh'] = BWHCToBCWH()

        self.workers['label_one_hot_decoder'] = OneHotDecoder(ConfigParser({"axis": 1}))

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, process the loaded train image data."""

        # data = info['data']['data']
        data = info['data']

        # labels = data.get('labels')
        # labels = self.workers['labels_background_to_color'].process(labels)
        # processed_data = data.update('labels', labels)
        #
        # # processed_data = self.workers['image_resizer'].resize(data)
        # processed_data = processed_data.map(self.workers['image_resizer'].process)
        # # processed_data = self.workers['image_normalizer'].normalize(processed_data)
        # processed_data = processed_data.map(self.workers['image_normalizer'].process)
        # processed_data = processed_data.map(self.workers['image_bwhc_to_bcwh'].process)
        #
        # labels = processed_data.get('labels')
        # labels = self.workers['label_one_hot_decoder'].process(labels)
        # processed_data = processed_data.update('labels', labels)

        processed_data = \
            data\
                .update_map({'_bc': {'$regex': 'labels$'}}, self.workers['labels_background_to_color'].process)\
                .update_map({}, [
                        self.workers['image_resizer'].process,
                        self.workers['image_normalizer'].process,
                        self.workers['image_bwhc_to_bcwh'].process
                    ])\
                .update_map({'_bc': {'$regex': 'labels$'}}, self.workers['label_one_hot_decoder'].process)

        return processed_data

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, process the loaded val image data."""

        data = info['data']

        labels = data.labels
        labels = self.workers['labels_background_to_color'].process(labels)
        processed_data = data.update('labels', labels)

        # processed_data = self.workers['image_resizer'].resize(data)
        processed_data = processed_data.map(self.workers['image_resizer'].process)
        # processed_data = self.workers['image_normalizer'].normalize(processed_data)
        processed_data = processed_data.map(self.workers['image_normalizer'].process)
        processed_data = processed_data.map(self.workers['image_bwhc_to_bcwh'].process)

        labels = processed_data.labels
        labels = self.workers['label_one_hot_decoder'].process(labels)
        processed_data = processed_data.update('labels', labels)

        return processed_data
