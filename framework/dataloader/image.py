from . import dataloader
from ..util.config import ConfigParser
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import cv2


class ImageLoader(dataloader.DataLoader):
    """This class reads and loads images."""

    def __init__(self, metadata: pd.DataFrame, config: ConfigParser = None):
        """Initializer for image data loader.

        Parameters
        ----------
        metadata : pd.DataFrame
            The metadata for the data to be loaded
        config : ConfigParser
            The configuration for the instance
        """

        super().__init__(metadata, config)

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image data loader'
    #
    #     return string

    def _load_single_data(self, row: Dict[str, Any]) -> np.ndarray:
        """Helper method to load a single image.

        Parameters
        ----------
        row: Dict[str, Any]
            The Pandas dataframe row corresponding to the current element

        Returns
        -------
        Numpy array of the loaded image

        """

        return cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)

    def _load_data_post(self, data: List) -> List:
        """Reforms the image data already loaded into a numpy array.

        Parameters
        ----------
        data : List
            The already loaded image data in a list

        Returns
        -------
        Loaded data itself

        """

        return data

    def _filter_file_name(self, file_name: str) -> bool:
        """"Helper function to filter a single file based on its name and criteria.

        Parameters
        ----------
        file_name : str
            The path of the file
        """

        extension = file_name.split('.')[-1].lower()
        accepted_extension = ['jpg', 'jpeg', 'png', 'gif', 'tif', 'tiff', 'eps', 'webp']

        return True if extension in accepted_extension else False
