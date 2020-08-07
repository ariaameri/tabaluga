from __future__ import annotations
from typing import List, Dict, Any
from types import FunctionType
from .config import ConfigParser


class DataHolder(ConfigParser):
    """A class to contain all data in order."""

    # Set static variables
    item_begin_symbol = f'\u2058'
    item_color = f'\033[38;5;112m'

    def __init__(self, data_dict: Dict = None):
        """Initializes the class based on the input data dictionary.

        Parameters
        ----------
        data_dict : Dict
            A dictionary containing all the data
        """

        # Check if no input was given
        if data_dict is None:
            data_dict = {}
        elif type(data_dict) is not dict:
            data_dict = {'data': data_dict}

        super().__init__(data_dict)

    def map(self, function: FunctionType) -> DataHolder:
        """Maps the function on the data and returns a new instance.

        Parameter
        ---------
        function : FunctionType
            The function to be mapped on the data

        Returns
        -------
        An instance of the class with new mapped data

        """

        def map_helper(data):

            if type(data) is type(self):
                return data.map(function)
            elif type(data) == list:
                out = [map_helper(item) for item in data]
            else:
                out = function(data)

            return out

        out = self.__class__({key: map_helper(value) for key, value in self.__dict__.items()})

        return out