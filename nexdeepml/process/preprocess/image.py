from . import preprocess
from ...util.config import ConfigParser
from typing import List, Union, Dict
import numpy as np
import cv2
import albumentations as A
from abc import abstractmethod


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

        self.interpolation = config.interpolation or cv2.INTER_AREA

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


class ImageAugmentationAlbumentations(preprocess.Preprocess):
    """Abstract class for image augmentation using the albumentations package."""

    def __init__(self, config: ConfigParser):
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

    def transform(self, **images) -> Dict:
        """Transforms/augment the input images.

        Parameters
        ----------
        **images
            Images to be augmented. They should be passed with their specific keywords

        Returns
        -------
        The augmented images with the same keywords as passed.

        """

        # Do the augmentations
        aug_images = self.transformer(**images)

        return aug_images


# TODO: add/look Augmentor package as well
# TODO: add/look imgaug package as well
