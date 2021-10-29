from . import dataloader
from deepmlframework.util.config import ConfigParser
import pandas as pd
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
import cv2


class ImageLoader(dataloader.DataLoader):
    """This class reads and loads images."""

    def __init__(self, config: ConfigParser, metadata: pd.DataFrame):
        """Initializer for image data loader.

        Parameters
        ----------
        config : ConfigParser
            The configuration for the instance
        metadata : pd.DataFrame
            The metadata for the data to be loaded
        """

        super().__init__(config, metadata)

        # Modify the metadata
        self.modify_metadata()

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image data loader'
    #
    #     return string

    def load_single_data(self, row: pd.Series) -> np.ndarray:
        """Helper method to load a single image.

        Parameters
        ----------
        row: pd.Series
            The Pandas dataframe row corresponding to the current element

        Returns
        -------
        Numpy array of the loaded image

        """

        return cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)

    def load_data_post(self, data: List) -> np.ndarray:
        """Reforms the image data already loaded into a numpy array.

        Parameters
        ----------
        data : List
            The already loaded image data in a list

        Returns
        -------
        Loaded data in numpy array format

        """

        return np.stack(data)

    def _filter_file_name(self, file_name: str) -> bool:
        """"Helper function to filter a single file based on its name and criteria.

        Parameters
        ----------
        file_name : str
            The path of the file
        """

        extension = file_name.split('.')[-1].lower()
        accepted_extension = ['jpg', 'jpeg', 'png', 'gif', 'tif', 'tiff', 'eps']

        return True if extension in accepted_extension else False
