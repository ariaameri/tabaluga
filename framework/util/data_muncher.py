from __future__ import annotations
from typing import List, Dict, Any, Callable, Union
from .panacea import Panacea, PanaceaLeaf


class DataMuncher(Panacea):
    """A class to contain all data in order."""

    # Set static variables
    item_begin_symbol = lambda _: f'\u2058'
    item_color = lambda _: f'\033[38;5;112m'

    def __init__(self, data_dict: Dict = None, leaf_class: PanaceaLeaf = None):
        """Initializes the class based on the input data dictionary.

        Parameters
        ----------
        data_dict : Dict
            A dictionary containing all the data
        leaf_class : PanaceaLeaf
            The pointer to the class that should be used as the leaf
        """

        # Check if no input was given
        if data_dict is None:
            data_dict = {}
        elif type(data_dict) is not dict:
            data_dict = {'data': data_dict}

        super().__init__(config_dict=data_dict, leaf_class=leaf_class or DataMuncherLeaf)

    def update_map(self,
                   filter_dict: Dict = None,
                   functions: Union[Callable[[Any], Any], List[Callable[[Any], Any]]] = None) \
            -> DataMuncher:
        """Applies the functions appeared in `functions` in order to the matched elements.

        The filtering criteria should only result in leaf nodes.
        The functions will be applied on the _value of the leaf node.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to Modification class for more info.
        functions : Union[Callable[[Any], Any], List[Callable[[Any], Any]]]
            (List of) functions to be applied on the elements in order

        Returns
        -------
        An instance of DataMuncher class with the updated attribute

        """

        # Update the filter criteria to make sure it selects only the leaf nodes
        self_selector = filter_dict.get('_self')
        if self_selector is not None:
            function_selector = self_selector.get('$function')
            if function_selector is not None:
                if isinstance(function_selector, list):
                    filter_dict['_self']['$function'] += [lambda x: x.is_leaf()]
                else:
                    filter_dict['_self']['$function'] = [filter_dict['_self']['$function'], lambda x: x.is_leaf()]
            else:
                filter_dict['_self']['$function'] = lambda x: x.is_leaf()
        else:
            filter_dict['_self'] = {'$function': lambda x: x.is_leaf()}

        # Make the appropriate update dictionary to call the superclass update method
        update_dict = {'$function': {'_value': functions}}

        # Do the update
        result = super().update(filter_dict=filter_dict, update_dict=update_dict)

        return result


class DataMuncherLeaf(PanaceaLeaf):
    """A leaf node class to contain all data in order."""

    # Set static variables
    item_begin_symbol = lambda _: f'\u1b75'
    item_color = lambda _: f'\x1b[38;5;32m'
    begin_list_symbol = lambda _: f'\x1b[38;5;81m-\033[0m'

    def __init__(self, value: Any):
        """Initializes the class based on the input value.

        Parameters
        ----------
        value : Any
            The value to store
        """

        super().__init__(value)
