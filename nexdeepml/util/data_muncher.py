from __future__ import annotations
from typing import List, Dict, Any
from types import FunctionType
from .config import ConfigParser
import re


class DataMuncher(ConfigParser):
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

    def map(self, function: FunctionType, filter_dict: Dict = {}, update_regex: str = ''):

        return self.map_helper(function, filter_dict, update_regex)

    def map_helper(self, function: FunctionType, filter_dict: Dict = {}, update_regex: str = '', bc: str = '', bc_meta: str = ''):

        def map_helper(data, key_name: str = ''):

            if type(data) is type(self):
                out = data.map_helper(function, {}, update_regex, key_name, bc_meta)
            elif type(data) == list:
                out = [map_helper(item, key_name) for item in data]
            else:
                out = function(data) if re.search(update_regex, key_name) else data

            return out

        bc_meta = bc_meta + self._parameters.get('_meta', '')

        if self._filter_checker(filter_dict, bc, bc_meta) is True:
            value = {key: map_helper(value, bc + f'.{key}') for key, value in self._parameters.items()}
            return self.__class__(value)

        final_dict = {**self._parameters,
                      **{
                          key: value.map_helper(function, filter_dict, update_regex, bc + f'.{key}', bc_meta)
                          for key, value
                          in self._parameters.items()
                          if type(value) is type(self)
                        }
                      }

        # TODO: what if I want to get to a variable directly with filtering and update it?

        return self.__class__(final_dict)
