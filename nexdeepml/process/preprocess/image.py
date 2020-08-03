from . import preprocess
from ...util.config import ConfigParser
from typing import List, Union
import numpy as np
import cv2


class ImageNormalizer(preprocess.Preprocess):
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

    def normalize(self, images: np.ndarray) -> np.ndarray:
        """"Normalizes the images given.

        It takes input images of dtype uint8 ranging from 0 to 255 and divides them by 255 with dtype of float.

        Parameters
        ----------
        images : np.ndarray
            A numpy array containing the image data of dtype uint8

        Returns
        -------
        Numpy array of image data of dtype float

        """

        # Get the type of the iterable
        collection_type = type(images)

        if collection_type == np.ndarray:
            output = images / 255.
        else:
            raise Exception('Unrecognized format passed to the image normalizer!')

        return output


class ImageResizer(preprocess.Preprocess):
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

        self.destination_image_size = config.destination_image_size

        self.interpolation = config.interpolation if config.interpolation is not None else cv2.INTER_AREA

        super().__init__(config)

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image resizer to {self.destination_image_size}'
    #
    #     return string

    def resize(self, images: np.ndarray) -> np.ndarray:
        """"Resizes the images given.

        Parameters
        ----------
        images : np.ndarray
            A numpy array containing the image to be resized

        Returns
        -------
        Numpy array of resized image data

        """

        # Get the type of the iterable
        collection_type = type(images)

        output = [
            cv2.resize(
                image,
                dsize=self.destination_image_size,
                interpolation=self.interpolation
            )
            for image
            in images
        ]

        if collection_type == np.ndarray:
            output = np.array(output)
        else:
            raise Exception('Unrecognized format passed to the image resizer!')

        return output
