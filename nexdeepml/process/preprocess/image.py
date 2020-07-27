from . import preprocess
from typing import List, Union
import numpy as np


class ImageNormalizer(preprocess.Preprocess):
    """Normalizes images.

    It takes input images of dtype uint8 ranging from 0 to 255 and divides them by 255 with dtype of float.
    """

    def __init__(self):
        """Initializes the class."""

        super().__init__()

    def normalize(self, images: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        """"Normalizes the images given.

        It takes input images of dtype uint8 ranging from 0 to 255 and divides them by 255 with dtype of float.

        Parameters
        ----------
        images : Union[List[np.ndarray], np.ndarray]
            An iterable collection containing the image data of dtype uint8

        Returns
        -------
        Collection of image data of dtype float

        """

        # Get the type of the iterable
        collection_type = type(images)

        if collection_type == list:
            output = [image / 255. for image in images]
        elif collection_type == np.ndarray:
            output = images / 255.
        else:
            raise Exception('Unrecognized format passed to the image normalizer!')

        return output
