from . import dataloader
from ..util.config import ConfigParser
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt


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

    def load_data(self, metadata: pd.DataFrame) -> List[np.ndarray]:
        """Loads images provided in the metadata data frame.

        Parameters
        ----------
        metadata : pd.DataFrame
            Panda's data frame containing the image metadata to be loaded

        Returns
        -------
        List of images : List[np.ndarray]

        NOTE: This function returns a list of numpy arrays and not a single numpy array.
                This is because images might be of different sizes and may need to be
                preprocessed at a later time.

        """

        images = [plt.imread(row['path']) for index, row in metadata.iterrows()]

        return images

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