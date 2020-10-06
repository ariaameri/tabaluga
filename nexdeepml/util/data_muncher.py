from __future__ import annotations
from typing import List, Dict, Any, Callable
from .panacea import Panacea, PanaceaLeaf


class DataMuncher(Panacea):
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

        super().__init__(config_dict=data_dict, leaf_class=DataMuncherLeaf)

    def update_map(self, filter_dict: Dict = None, functions: List[Callable[[Any], Any]] = None) -> DataMuncher:
        """Applies the functions appeared in `functions` in order to the matched elements.

        The filtering criteria should only result in leaf nodes.
        The functions will be applied on the _value of the leaf node.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to Modification class for more info.
        functions : List[Callable[[Any], Any]]
            List of functions to be applied on the elements in order

        Returns
        -------
        An instance of DataMuncher class with the updated attribute

        """

        def function_chain_apply(x):
            """Helper function to chain all the functions given in the `functions` list and applies them in order to
            the given `x` input."""

            result = x

            # Apply the functions one-by-one and in order
            for function in functions:
                result = function(result)

            return result

        # Make the appropriate update dictionary to call the superclass update method
        update_dict = {'$function': {'_value': function_chain_apply}}

        # Do the update
        result = super().update(filter_dict=filter_dict, update_dict=update_dict)

        return result


class DataMuncherLeaf(PanaceaLeaf):
    """A leaf node class to contain all data in order."""

    # Set static variables
    item_begin_symbol = f'\u273f'
    item_color = f'\033[33m'
    begin_list_symbol = f'-'
    begin_list_color = f'\033[38;5;70m'
    begin_list_symbol = f'{begin_list_color}{begin_list_symbol}\033[0m'

    def __init__(self, value: Any):
        """Initializes the class based on the input value.

        Parameters
        ----------
        value : Any
            The value to store
        """

        super().__init__(value)
