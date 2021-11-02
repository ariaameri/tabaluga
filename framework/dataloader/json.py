from . import dataloader
from ..util.config import ConfigParser
import pandas as pd
import json


class JSONLoader(dataloader.DataLoader):
    """This class reads and loads images."""

    def __init__(self, config: ConfigParser, metadata: pd.DataFrame):
        """Initializer for json data loader.

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

    def load_single_data(self, row: pd.Series):
        """Helper method to load a single json.

        Parameters
        ----------
        row: pd.Series
            The Pandas dataframe row corresponding to the current element

        Returns
        -------
        loaded json

        """

        with open(row['path']) as file:
            the_json = json.load(file)

        return the_json

    def load_data_post(self, data: list) -> list:
        """Reforms the json data already loaded into a list.

        Parameters
        ----------
        data : List
            The already loaded json data in a list

        Returns
        -------
        Loaded data in list

        """

        # do nothing, just return it!
        return data

    def _filter_file_name(self, file_name: str) -> bool:
        """"Helper function to filter a single file based on its name and criteria.

        Parameters
        ----------
        file_name : str
            The path of the file
        """

        extension = file_name.split('.')[-1].lower()
        accepted_extension = ['json']

        return True if extension in accepted_extension else False
