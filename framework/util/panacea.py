from __future__ import annotations
from typing import Dict, Any, List, Type, Union, Callable
from types import FunctionType
import yaml
from enum import Enum
import re
from itertools import chain
from .option import Some, nothing, Option
from abc import ABC, abstractmethod
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


# a list of restricted names that are used and a key name should not collide with these
_restricted_names = ['_value', '_self', '_bc', '_key_name']


class PanaceaBase(ABC):
    """A class that will act as a base node class for the config tree classes."""

    def __init__(self):
        """Initializes the class."""

        # Create a dictionary to hold the data
        self._parameters: Dict[str, Any] = {}

    # Representation

    @abstractmethod
    def print(self, depth: int = -1) -> None:
        """Print the configuration up to certain depth.

        Parameters
        ----------
        depth : int, optional
            The depth until which the string representation should go down the configuration

        """

        pass

    @abstractmethod
    def dict_representation(self) -> Dict:
        """Method to help with visualization of the configurations.

        Returns
        -------
        dict
            Configurations in a dictionary such as the one with json/yaml
        """

        pass

    @abstractmethod
    def str_representation(self, name: str, depth: int = -1) -> str:
        """Helper function to create a string representation of the instance.

        Parameters
        ----------
        name : str
            The name of the current instance
        depth : int, optional
            The depth until which the string representation should go down the configuration

        Returns
        -------
        String containing the representation of the configuration given

        """

        pass

    @abstractmethod
    def _identity_str_representation(self, name) -> str:

        pass

    # Getters

    @abstractmethod
    def get_option(self, item: str) -> Option[PanaceaBase]:
        """Gets an item in the instance and return an Option value of that.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level

        Returns
        -------
        An Option value possibly containing the item

        """

        pass

    def get(self, item: str) -> Any:
        """Gets an item in the instance and return it or raise an error if not exists.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level or in the deeper level where levels are separated by '.'
                For example item1.item2 will look at item1 in the shallowest level and will look for item2 inside item1.

        Returns
        -------
        Value of the item

        """

        out = self.get_value_option(item)

        # Raise an error if the item does not exist
        if out.is_empty():
            raise AttributeError(f'{item} does not exist in this instance of {self.__class__.__name__}!')

        # in case we got a leaf, get the value

        return out.get()

    def get_value_option(self, item: str) -> Option[Any]:
        """
        Gets an item in the instance and return an Option value of that.
        This should be noted that this is the same as get_option method except in the case that the search result in a
        leaf node, in which the returned value will be an Option-wrapped of the value of the leaf.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level or in the deeper level where levels are separated by '.'
                For example item1.item2 will look at item1 in the shallowest level and will look for item2 inside item1.

        Returns
        -------
        An Option value possibly containing the item

        """

        # get the item
        value: Option[PanaceaBase] = self.get_option(item)

        # return the right thing!
        if value.filter(lambda x: x.is_leaf()).is_defined():
            return value.map(lambda x: x.get())
        else:
            return value

    def get_or_else(self, item: str, default_value: Any) -> Any:
        """Gets an item in the instance and return default_value if not found.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level
        default_value : Any
            A value to return if the item was not found

        Returns
        -------
        The item looking for or default value

        """

        return self.get_option(item).fold(lambda x: x.get() if isinstance(x, self.Leaf) else x, default_value)

    def get_parameters(self) -> Dict:
        """Gets the entire parameters.

        Returns
        -------
        A shallow copy of the instance parameters

        """

        # Create a shallow copy of the parameters
        out = {**self._parameters}

        return out

    # Operations

    def map(self, func: FunctionType) -> PanaceaBase:
        """Applies a function on the internal value and returns a new instance.

        Parameters
        ----------
        func : FunctionType
            The function to be applied on the internal value

        Returns
        -------
        A new instance with the updated value

        """

        pass

    # Checkers

    def is_branch(self) -> bool:
        """Method to return whether or not the current instance is a branch node or not."""

        return True if issubclass(type(self), Panacea) else False

    def is_leaf(self) -> bool:
        """Method to return whether or not the current instance is a leaf node or not."""

        return True if issubclass(type(self), PanaceaLeaf) else False

    def is_empty(self) -> bool:
        """A checker method that tells whether this instance of config is empty or not

        Returns
        -------
        bool
            False, as Leaf is not empty, this will be overriden in branch class
        """

        return False

    # Debugging mode

    def enable_debug_mode(self) -> PanaceaBase:
        """Enables debug mode, in this instance and all children, where parameters can be accessed with . notation."""

        # Reserve the current __dict__
        self.__old___dict__ = self.__dict__

        # Update the dictionary and pass the method down to children
        parameters_to_add = {
            **{key: value for key, value in self._parameters.items() if value.is_branch()},
            **{key: value.get() for key, value in self._parameters.items() if value.is_leaf()}
        }
        self.__dict__ = {**self.__dict__, **parameters_to_add}
        for key, item in self._parameters.items():
            if issubclass(type(item), Panacea):
                item.enable_debug_mode()

        return self

    def disable_debug_mode(self) -> PanaceaBase:
        """Disables debug mode, in this instance and all children, where parameters can be accessed with . notation."""

        # Check if enable_debug_mode has already been activated
        if getattr(self, '__old___dict__') is None:
            return self

        # Set the __dict__ to previous version and pass down
        self.__dict__ = self.__old___dict__
        for key, item in self._parameters.items():
            if issubclass(type(item), PanaceaBase):
                item.disable_debug_mode()

        # Delete the old dictionary
        del self.__old___dict__

        return self

    # Traversals

    def filter(self, filter_dict: Dict = None) -> Panacea:
        """Method to filter the current item based on the criteria given and return a new Panacea that satisfies
            the criteria.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to Modification class for more info.

        Returns
        -------
        A new instance of the Panacea class whose elements satisfy the given criteria.

        """

        # Make the Modification instance
        modifier = Modification(filter_dict=filter_dict)

        # Perform the filtering
        filtered: Option[(str, PanaceaBase)] = modifier.filter(panacea=self)

        # Return the result of an empty Panacea if the filtering result is empty
        return filtered.get_or_else(('', self.__class__()))[1]

    def find_one(self, filter_dict: Dict = None) -> Option:
        """Method to find the first item based on the criteria given and return an Option value of it.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to Modification class for more info.

        Returns
        -------
        An Option value of the first element found that satisfies the given criteria.

        """

        # Make the Modification instance
        modifier = Modification(filter_dict=filter_dict)

        # Perform the finding
        found_one: Option[Any] = modifier.find_one(panacea=self)

        # Return
        return found_one

    def update(self, filter_dict: Dict = None, update_dict: Dict = None, condition_dict: Dict = None) -> PanaceaBase:
        """Update an entry in the config and return a new Panacea.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to Modification class for more info.
        update_dict : dict
            Dictionary containing the updating rules.
                Refer to Modification class for more info.
        condition_dict : dict
            Dictionary containing the conditions of updating.
                Refer to Modification class for more info.

        Returns
        -------
        An instance of Panacea class with the updated attribute

        """

        # Make the Modification instance
        modifier = Modification(filter_dict=filter_dict, update_dict=update_dict, condition_dict=condition_dict)

        # Perform the update
        updated: Option[(str, PanaceaBase)] = modifier.update(panacea=self)

        # Return
        return updated.get_or_else(('', self))[1]


class Panacea(PanaceaBase):
    """A class that will contain config values and subclasses.

    This class is the non-leaf node class in the tree of configurations given.
    The leaf nodes will be of type PanaceaLeaf.
    """

    # Set static variables
    item_begin_symbol = lambda _: f'\u2022'
    item_color = lambda _: f'\033[38;5;209m'
    after_item_symbol = lambda _: f':'

    vertical_bar_with_color = lambda _: f'\033[37m\u22EE\033[0m'

    def __init__(self, config_dict: Dict = None, leaf_class: PanaceaLeaf = None):
        """Initializes the class based on the input config dictionary.

        Parameters
        ----------
        config_dict : Dict
            A dictionary containing all the configurations
        leaf_class : PanaceaLeaf
            The pointer to the class that should be used as the leaf
        """

        super().__init__()

        # Set the given leaf class or the default
        self.Leaf: PanaceaLeaf = leaf_class if leaf_class is not None else PanaceaLeaf

        # Create a dictionary to hold the data
        self._parameters: Dict[str, PanaceaBase] = {}

        # Check if no input was given
        if config_dict is None:
            return

        # Traverse the dictionary and set the parameters dictionary
        self._parameters = self.__init_helper(config_dict)

    @classmethod
    def create_from_file(cls, file_path: str):
        """A classmethod that creates an instance of the class by taking a yaml file and loading its data.

        Parameters
        ----------
        file_path : str
            The absolute (or relative) path to the yaml file to be read
        """

        # Read the file
        with open(file_path) as file:
            config_dict = yaml.full_load(file)

        # Return instance of the class
        return cls(config_dict)

    def __init_helper(self, config_dict: Dict) -> Dict:
        """Helper method for the constructor to recursively construct the parameters dictionary of the class

        Parameter
        ---------
        config_dict : dict
            A dictionary of the configuration to be turned into parameters

        Returns
        -------
        Processed parameters dictionary

        """

        def helper(value: Any) -> PanaceaBase:
            """Helper function to decide what to do with the incoming value.

            Parameters
            ----------
            value : Any
                The value to decide what to do about

            Returns
            -------
            Either the value unchanged or a Panacea instance if the value is a dictionary

            """

            # Check the type of the input
            # If it is already a PanaceaBase, let it be
            # If the type is a dictionary, go deep into it
            # Otherwise, make a leaf node of that
            if issubclass(type(value), PanaceaBase):
                return value
            elif isinstance(value, dict):
                return self.__class__(value)
            else:
                out = self.Leaf(value)

            return out

        # checking for validity
        for item in config_dict.keys():

            # check that the names do not contain `.`s
            if '.' in item:
                raise ValueError(f'cannot have a key of name {item}: key elements cannot have `.` in their names')

            # check for invalid key names
            if item in _restricted_names:
                restricted_names_str = '"' + '," "'.join(_restricted_names) + '"'
                raise ValueError(
                    f'cannot have a key of name {item}: '
                    f'key elements must be different than {restricted_names_str}'
                )

            # check if the key is string
            if not isinstance(item, str):
                raise ValueError(f'cannot have a key of {item}, keys have to be strings')

        # Create the dictionary of parameters
        final_parameters = {key: helper(value) for key, value in config_dict.items()}

        return final_parameters

    # Relations

    def __eq__(self, other):
        """Returns whether two Panaceas are the same or not."""

        # If other is not Panacea, then they are not the same
        if not isinstance(other, self.__class__):
            return False

        # If the same object
        if self is other:
            return True

        # Get the other instance parameters
        other_parameters = other.get_parameters()

        # All parameters must be equal
        return other_parameters == self._parameters

    def __ne__(self, other):
        """Returns whether two Panaceas are not the same."""

        return not (self == other)

    def __copy__(self):
        """Shallow copy of the current instance."""

        return self.__class__(self._parameters)

    def __deepcopy__(self, memodict={}):
        """Deep copy of the current instance."""

        return self.__class__(self.dict_representation())

    # Operations

    def map(self, func: FunctionType) -> PanaceaBase:
        """Applies a function to parameters and return a new Panacea of that.

        Parameters
        ----------
        func : FunctionType
            The function to be applied on the internal parameters

        Returns
        -------
        A new Panacea with the updated value

        """

        return self.__class__(func(self._parameters))

    # Checkers

    def contains(self, value: Any) -> bool:
        """Check if the given `value` exists in the shallowest level of the instance."""

        return value in self._parameters.values()

    def contains_key(self, key: str) -> bool:
        """Check if the given `key` exists as a name of a parameter in the shallowest level of the instance."""

        return key in self._parameters.keys()

    def exists(self, func: FunctionType) -> bool:
        """Checks if the function `func` holds true for any of the internal parameters."""

        return any(func(item) for item in self._parameters.values())

    def for_all(self, func: FunctionType) -> bool:
        """Checks if the function `func` holds true for all of the internal parameters."""

        return all(func(item) for item in self._parameters.values())

    # Representation

    def __str__(self) -> str:
        """Method to help with the visualization of the configuration in YAML style."""

        # Get dictionary representation of the current instance
        out_string = self.str_representation(name="", depth=2)

        return out_string

    def print(self, depth: int = -1) -> None:
        """Print the configuration up to certain depth.

        Parameters
        ----------
        depth : int, optional
            The depth until which the string representation should go down the configuration

        """

        print(self.str_representation(depth=depth, name=''))

    def dict_representation(self) -> Dict:
        """Method to help with visualization of the configurations.

        Returns
        -------
        dict
            Configurations in a dictionary such as the one with json/yaml
        """

        # Check for the type of the input and act accordingly
        out = {key: item.dict_representation() for key, item in self._parameters.items()}

        return out

    def str_representation(self, name: str, depth: int = -1) -> str:
        """Helper function to create a string representation of the instance.

        Parameters
        ----------
        name : str
            The name of the current instance
        depth : int, optional
            The depth until which the string representation should go down the configuration

        Returns
        -------
        String containing the representation of the configuration given

        """

        # Check if we have reach the root of the recursion, i.e. depth is zero
        if depth == 0:
            return ''

        # Create the resulting string
        out_string = ''
        out_string += self._identity_str_representation(name)
        out_string += self.after_item_symbol() if depth != 1 and name != '' else ''  # Only add after_item_symbol if we want to print anything in front
        out_string += f'\n'

        # Create the string from all the children
        out_substring = \
            ''.join(
                item.str_representation(name=key, depth=depth-1)
                for key, item
                in
                # first print the leaves and then the branches
                sorted(self._parameters.items(), key=lambda x: (not x[1].is_leaf(), x[0]))
            )

        # Indent the children result and add to the result
        out_string += re.sub(
            r'(^|\n)(?!$)',
            r'\1' + f'{self.vertical_bar_with_color()}' + r'\t',
            out_substring
        )

        return out_string

    def _identity_str_representation(self, name) -> str:

        return f'{self.item_begin_symbol()} {self.item_color()}{name}\033[0m'

    # Getters

    def get_option(self, item: str) -> Option[PanaceaBase]:
        """
        Gets a branch/leaf item in the instance and return an Option value of that.
        This should be noted that this method returns a PanaceaBase, so the search can result in a branch or leaf node,
        which will be wrapped in an Option and returned.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level or in the deeper level where levels are separated by '.'
                For example item1.item2 will look at item1 in the shallowest level and will look for item2 inside item1.

        Returns
        -------
        An Option value possibly containing the item

        """

        # Go deeper if we see the pattern 'item1.item2'
        if '.' in item:
            element = self.get_option(item.split('.')[0])
            if element.is_empty():
                return element
            else:
                return element.get()\
                    .get_option('.'.join(item.split('.')[1:]))

        # Look for the item and return it
        if item in self._parameters.keys():
            return Some(self._parameters.get(item))
        else:
            return nothing

    def get_leaf(self, item: str) -> PanaceaLeaf:
        """Gets a leaf node in the instance and return it.

        Parameters
        ----------
        item : str
            Leaf name to look for in the shallowest level

        Returns
        -------
        The leaf

        """

        # Get the item from the returned option
        out = self.get_option(item)

        # Check if the returned item is a leaf node
        if not out.exist(lambda x: isinstance(x, self.Leaf)):
            raise AttributeError(f'Leaf item {item} does not exist in this instance of {self.__class__.__name__}!')

        return out.get()

    def get_or_empty(self, item: str) -> Panacea:
        """
        Tries to get an item. If the item does not exist, returns an empty instance of the class.

        Parameters
        ----------
        item : str
            Item to look for. This item has to be a Panacea, otherwise, will panic.

        Returns
        -------
            Panacea that either contains the found result or empty

        """

        result = self.get_option(item)

        if result.is_defined():
            if not issubclass(type(result.get()), Panacea):
                raise ValueError(f"item of {item} requested resulted in not a branch node.")
            return result.get()
        else:
            return self.__class__()

    def get_parameters(self) -> Dict:
        """Gets the entire parameters.

        Returns
        -------
        A shallow copy of the instance parameters

        """

        # Create a shallow copy of the parameters
        out = {**self._parameters}

        return out

    def __getattr__(self, item):
        """Method to help with getting an attribute.

        Using this method, if the requested attribute does not exist,
        a default value of None will be returned.
        """

        return None

    # def __setattr__(self, name, value):
    #     """Method to help with setting an attribute.
    #
    #     Using this method, if the requested attribute does not exist,
    #     sets it and if it exists, raises an error.
    #     """
    #
    #     raise Exception(f"Cannot change or add value of {name} due to immutability. Use `update` method instead.")

    # State getters

    def is_empty(self) -> bool:
        """EXPERIMENTAL: A checker method that tells whether this instance of config is empty or not

        Returns
        -------
        bool
            True if this class is empty and False otherwise
        """

        return not bool(self._parameters)

    def _is_flat(self, contain__: bool = False) -> bool:
        """Returns whether or not the instance is flat, i.e. contains only leaf nodes."""

        parameter_names = list(self._parameters.keys())

        # Check if non of the parameters are leaf of subtype Panacea, i.e. the instance is flat
        check = all([
            not issubclass(type(self._parameters.get(parameter_name)), Panacea)
            for parameter_name
            in parameter_names
            if not parameter_name.startswith('_')
        ])

        return check

    # Modifications

    def reduce(self, function: Callable[[Any, Any], Any]) -> Any:
        """Reduces the parameters of this collection using the specified associative binary operator.

        It will return function(function(...function(x1, x2), x3), ..., xn) where xi is the i-th parameter.

        Parameters
        ----------
        function : Callable[[A, A], A]
            Function to do the reduction. Must take two arguments of type A and return value of type A

        Returns
        -------
        Result of the reduction

        """

        parameter_names = list(self._parameters.keys())

        # Assert the flat-ness of the instance
        assert self._is_flat() is True, \
            f'Object of type {self.__class__.__name__} is not flat, cannot perform reduction!'

        # Fill the result with the first element
        result = self.get(parameter_names[0])

        # Reduce the function `function` over all elements
        for parameter_name in parameter_names[1:]:
            result = function(result, self.get(parameter_name))

        return result

    def fold(self, zero_element: Any, function: Callable[[Any, Any], Any]) -> Any:
        """Folds the parameters of this collection using the specified associative binary operator and zero_elemnt.

        It will return function(function(...function(zero_element, x1), x2), ..., xn) where xi is the i-th parameter.

        Parameters
        ----------
        function : Callable[[B, A], B]
            Function to do the reduction. Must take two arguments of type B and A and return value of type B
        zero_element : Any
            The first element to be combined in the fold procedure

        Returns
        -------
        Result of the fold of type B

        """

        parameter_names = list(self._parameters.keys())

        # Assert the flat-ness of the instance
        assert self._is_flat() is True, \
            f'Object of type {self.__class__.__name__} is not flat, cannot perform fold!'

        # Fill the result with the first element
        result = zero_element

        # Reduce the function `function` over all elements
        for parameter_name in parameter_names:
            result = function(result, self.get(parameter_name))

        return result

    def union(self, new_config: Panacea) -> Panacea:
        """method to take the union of two config instances and replace the currently existing ones.

        Parameters
        ----------
        new_config : Panacea
            A config instance to union with (and update) the current one

        Returns
        -------
        A new instance of ConfigParser containing the union of the two config instances

        """

        def union_helper(new: Any, old: Any, this) -> PanaceaBase:
            """Helper function to create the union of two elements.

            Parameters
            ----------
            new : Any
                The new element
            old : Any
                The old element
            this
                The reference to the outer class

            Returns
            -------
            The result of union of the new and old elements

            """

            # If we are dealing with the same object, return it
            if new is old:
                return new
            # If the new and old items are ConfigParsers, traverse recursively, else return the new item
            if type(new) == type(old) == type(this):
                return old.union(new)
            else:
                return new

        # If None is passed, do nothing
        if new_config is None:
            return self

        assert type(new_config) is type(self), \
            f"The element to be union-ed must be the same type as {self.__class__.__name__} class."

        # Find the elements that exist in only one of the configs
        out_delta = {
            key: value
            for key, value in chain(self._parameters.items(), new_config._parameters.items())
            if (key in self._parameters.keys()) ^ (key in new_config._parameters.keys())
        }

        # Find the elements that exist in both of the configs
        out_intersection = {
            key: union_helper(value_new, self._parameters.get(key), self)
            for key, value_new in new_config._parameters.items()
            if (key in self._parameters) and (key in new_config._parameters)
        }

        # Concatenate the newly generated dictionaries to get a new one and create a Panacea from that
        final_parameters = {**out_delta, **out_intersection}

        # If the union is the current instance, return self
        if final_parameters == self._parameters:
            return self
        # If the union is the to-be-union-ed instance, return it
        elif final_parameters == new_config._parameters:
            return new_config
        else:
            return self.__class__(final_parameters)

    def intersection(self, new_config: Panacea) -> Panacea:
        """method to take the intersection of two config instances.

        Parameters
        ----------
        new_config : Panacea
            A config instance to intersect with the current one

        Returns
        -------
        A new instance of ConfigParser containing the intersection of the two config instances

        """

        return self.intersection_option(new_config).get_or_else(self.__class__())

    def intersection_option(self, new_config: Panacea) -> Option:
        """method to help to take the intersection of two config instances.

        Parameters
        ----------
        new_config : Panacea
            A config instance to intersect with the current one

        Returns
        -------
        An Option value containing the intersection of the two config instances

        """

        # Find the dictionary of intersection for the parameters
        out_intersection = {
            key: value.get()
            for key, value
            in {
                key: self._parameters.get(key).intersection_option(value_new)
                for key, value_new in new_config._parameters.items()
                if (key in self._parameters) and (key in new_config._parameters)
            }.items()
            if value.is_defined()
        }

        # If the intersection is the current instance, return self
        if out_intersection == self._parameters:
            return Some(self)
        # If the intersection is the to-be-intersected instance, return it
        elif out_intersection == new_config._parameters:
            return Some(new_config)
        # Concatenate the newly generated dictionaries to get a new one and create a Panacea from that
        elif out_intersection:
            return Some(self.__class__(out_intersection))
        # If the intersection is empty
        else:
            return nothing


class PanaceaLeaf(PanaceaBase):
    """A class that will contain a value in a config tree.

    This class is the leaf class in the tree of configurations given.
    The non-leaf nodes will be of type Panacea.
    """

    # Set static variables
    item_begin_symbol = lambda _: f'\u273f'
    item_color = lambda _: f'\033[33m'
    after_item_symbol = lambda _: f':'
    begin_list_symbol = lambda _: f'\033[38;5;70m-\033[0m'

    def __init__(self, value: Any):
        """Initializes the class based on the input value.

        Parameters
        ----------
        value : Any
            The value to store
        """

        super().__init__()

        # To conform with the dictionary initialization, read also from a dictionary whose value is stored in _value key
        if isinstance(value, dict) and '_value' in value.keys():
            self._value = value.get('_value')
        else:
            # Hold the value
            self._value = value

    # Relations

    def _leaf_reference_eq(self, other: PanaceaLeaf):
        """Check whether other has the same reference as the current instance."""

        return self is other

    def _leaf_value_eq(self, other: PanaceaLeaf):
        """Check whether other has the same internal value as the current instance."""

        if isinstance(other, self.__class__):
            return self.contains(other.get())
        else:
            return False

    def _any_eq(self, other: Any):
        """Check whether other has the same value as the internal value of the current instance."""

        return self.contains(other)

    def _leaf_value_lt(self, other: PanaceaLeaf):
        """Check whether other internal value is less than the current instance."""

        if isinstance(other, self.__class__):
            return self.get() < other.get()
        else:
            return False

    def _any_lt(self, other: Any):
        """Check whether other internal value is less than that of the current instance."""

        return self.get() < other

    def __eq__(self, other):
        """Defines the equality for the class.

        Two leaves are equal if their references are equal or if their values are equal.
        A leaf can also be equal to a value if its internal value is equal to it.

        """

        return self._leaf_reference_eq(other) or self._leaf_value_eq(other) or self._any_eq(other)

    def __ne__(self, other):
        """Defines the not equality for the class.

        Two leaves are not equal if their references are not equal or if their values are different.
        In other words, they are not equal if they are not ==-ly the same.

        """

        return not (self == other)

    def __lt__(self, other):
        """Defines the less than for the class.

        This comparison is based on the internal values.

        """

        return self._leaf_value_lt(other) or self._any_lt(other)

    def __gt__(self, other):
        """Defines the greater than for the class.

        This comparison is based on the internal values.

        """

        return (self < other) and (self != other)

    def __ge__(self, other):
        """Defines the greater than for the class.

        This comparison is based on the internal values.

        """

        return (self > other) or self == other

    def __le__(self, other):
        """Defines the less than for the class.

        This comparison is based on the internal values.

        """

        return (self < other) or (self == other)

    # Operations

    def __add__(self, other) -> PanaceaLeaf:
        """Returns a new Leaf with added value"""

        if isinstance(other, self.__class__):
            return self.__class__(self._value + other.get())
        else:
            return self.__class__(self._value + other)

    def __mul__(self, other) -> PanaceaLeaf:
        """Returns a new Leaf with multiple value"""

        if isinstance(other, self.__class__):
            return self.__class__(self._value * other.get())
        else:
            return self.__class__(self._value * other)

    def __copy__(self) -> PanaceaLeaf:
        """Returns a copy of itself"""

        return self.__class__(self._value)

    def map(self, func: FunctionType) -> PanaceaBase:
        """Applies a function on the internal value and returns a new Leaf.

        Parameters
        ----------
        func : FunctionType
            The function to be applied on the internal value

        Returns
        -------
        A new leaf with the updated value

        """

        return self.__class__(func(self._value))

    # Checkers

    def contains(self, value: Any) -> bool:
        """Check if the given `value` is the same as the internal one."""

        return self._value == value

    def exists(self, func: FunctionType) -> bool:
        """Returns the result of the applying the boolean function on the internal value."""

        return func(self._value)

    # Modification

    def intersection_option(self, new_config: PanaceaBase) -> Option:
        """method to help to take the intersection of two config instances.

        Parameters
        ----------
        new_config : Panacea
            A config instance to intersect with the current one

        Returns
        -------
        An Option value containing the intersection of the two config instances

        """

        return Some(self) if (self == new_config) else nothing

    # Representation

    def print(self, depth: int = -1) -> None:
        """Print the configuration up to certain depth.

        Parameters
        ----------
        depth : int, optional
            The depth until which the string representation should go down the configuration

        """

        print(self.str_representation(depth=depth, name=''))

    def __str__(self) -> str:
        """Method to help with the visualization of the configuration in YAML style."""

        # Get dictionary representation of the current instance
        out_string = self.str_representation(depth=1, name='')

        return out_string

    def dict_representation(self) -> Any:
        """Method to help with visualization of the configurations.

        Returns
        -------
        dict
            Configurations in a dictionary such as the one with json/yaml
        """

        result = self._value

        return result

    def str_representation(self, name: str, depth: int = -1) -> str:
        """Helper function to create a string representation of the instance.

        Parameters
        ----------
        name : str
            The name of the current instance
        depth : int, optional
            The depth until which the string representation should go down the configuration

        Returns
        -------
        String containing the representation of the configuration given

        """

        # Check if we should give anything
        if depth == 0:
            return ''

        out_string = self._identity_str_representation(name)

        # Check if we should go any further deep
        if depth == 1:
            return out_string + f'\n'

        out_string += f'{self.after_item_symbol()} '

        if isinstance(self._value, list):
            out_string += self._list_str_representation()
        else:
            out_string += self._item_str_representation()

        return out_string

    def _identity_str_representation(self, name) -> str:

        return f'{self.item_begin_symbol()}{self.item_color()} {name}\033[0m'

    def _item_str_representation(self) -> str:

        return f'{self._value}\n'

    def _list_str_representation(self) -> str:

        out_string = ''.join(f'\t{self.begin_list_symbol()} {value}\n' for value in self._value)

        out_string = f'\n{out_string}'

        return out_string

    # Getters

    def get_option(self, item: str = '_value') -> Option[PanaceaBase]:
        """
        Gets a branch/leaf item in the instance and return an Option value of that.
        Basically, return an Option value of self

        Parameters
        ----------
        item : str, optional
            extra!

        Returns
        -------
        An Option value possibly containing the item

        """

        return nothing

    def get_value_option(self, item: str = '_value') -> Option[Any]:
        """
        Gets an item in the instance and return an Option value of that.

        Parameters
        ----------
        item : str
            Item to look for

        Returns
        -------
        An Option value possibly containing the item

        """

        if item == '_value':
            return Some(self._value)
        else:
            return nothing

    def get(self, item: str = '_value') -> Any:
        """Returns the internal value

        Parameters
        ----------
        item : str
            Dummy variable, just to conform with the superclass

        Returns
        -------
        Value of the _value

        """

        return super().get(item)

    # Debugging mode

    def enable_debug_mode(self) -> PanaceaLeaf:
        """Enables debug mode, in this instance and all children, where parameters can be accessed with . notation."""

        # Do the general enable_debug_mode
        super().enable_debug_mode()

        # Update the dictionary and pass the method down to children
        self.__dict__ = {**self.__dict__, '_value': self._value}

        return self


class Modification:

    def __init__(self, filter_dict: Dict = None, update_dict: Dict = None, condition_dict: Dict = None):
        """Initializes the instance and create the modified filter and update dictionaries as Filter and Update
        class methods.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                The keys of the dictionary are among these options:
                    - the name of field: this will cause the filtering to be based on this specific field
                    - special names that start with '_': these will cause the filtering to be based on special
                        things specified by this key.
                        A complete list of them are:
                            - _bc: filter on the breadcrumb of the name of the fields
                            - _self: filter on the current item itself
                            - _value: filter based on the value of the leaf nodes only
                For the value of dictionary, refer to class Filter for more info.
        update_dict : dict
            Dictionary containing the update rules.
                The keys of the dictionary must contain the operators of the Update class. Refer to the Update class
                    for more info.
                The values of the dictionary should themselves be dictionaries with the folllwing structure.
                    The keys of this dictionary are among these options:
                        - the name of field: this will cause the updating to happen on the field
                        - special names that start with '_': these will cause the updating to be based on special
                            things specified by this key.
                            A complete list of them are:
                                - _self: update the current item itself
                                - _value: update the value of the leaf nodes only, that is if the filter matches a leaf

        condition_dict : dict
            Dictionary containing the condition of update/filter.
                The keys/value possible pair are among these options:
                    - 'deep': True/False: tells the update rule whether it should go down deep and update. Otherwise,
                            if set to False, will update only the shallowest level matched.
                    - 'thread_max_count': int: the maximum number of threads to use for multithreading updates.
                    - 'process_max_count': int: the maximum number of processes to use for multithreading updates.

        Examples
        --------

        ##### Filter #####

        Here, we will provide some examples for the `filter_dict` dictionary.
        In both the following examples, we want to filtering based on a field with name `foo` and is equal to 10
        >>> {'foo': 10}
        >>> {'foo': {'$equal': 10}}

        In this example, we want to filter if the field `foo` does not exist
        >>> {'foo': {'$exists': 0}}

        Now, we want to filter if the breadcrumb contains `foo` and is at depth 5. Note that the breadcrumb starts
        with an initial '.'.
        >>> {'_bc': {'$regex': r'(\.\w+){4}\.foo$'}}

        In this example, we want to filter all the nodes whose values are int
        >>> {'_value': {'$function': lambda a: isinstance(a, int)}}

        ##### Update #####

        Here, we will provide some examples for the `update_dict` dictionary.
        We want to set a value of 10 for the field `foo`
        >>> {'$set': {'foo': 10}}
        We want to set a value of 10 for the non-existing field `foo`
        >>> {'$set_only': {'foo': 10}}
        We want to set a value of 10 for the existing field `foo`
        >>> {'$update': {'foo': 10}}

        Remove a field named `foo`
        >>> {'$unset': {'foo': None}}

        Rename the field `foo` to `bar`
        >>> {'$rename': {'foo': 'bar'}}


        ##### Filter and Update #####

        Here, we will provide some examples for initializing the current class.
        Filter the Panacea if it has the field `foo` then add the field `bar` with value of 10 to that instance
        >>> Modification({'foo': {'$exists': 1}}, {'$set': {'bar': 10}})

        Filter the Panacea to meet the criteria when inside a leaf node uniqely named `foo`
            then change its value to `bar`
        >>> Modification({'_bc': {'$regex': r'foo$'}}, {'$set': {'_value': 'foo'}})

        """

        self.condition_dict = condition_dict or {}

        self.filter_dict = self.make_filter_dictionary(filter_dict or {})
        self.update_dict = self.make_update_dictionary(update_dict or {})

    class Filter:
        """A class that parses, holds and checks the filtering queries for Panacea."""

        # List of available operations
        class Operations(Enum):
            FUNCTION = '$function'
            EXISTS = '$exists'
            REGEX = '$regex'
            EQUAL = '$equal'
            OR = '$or'
            AND = '$and'

        def __init__(self, query: Dict):
            """Initializer to the class which will parse the query and create the corresponding actions.

            Parameters
            ----------
            query : Dict
                The query dictionary to do the filtering to be parsed
                `query` must be a dictionary with its keys being the operators of the Filter class and its values
                    being the corresponding expression.
                    Supported operators are:
                        - $exists: checks whether a field exists
                            the value can be 1 to denote the existence of the field and 0 otherwise
                        - $regex: checks a string against a regex expression
                            the value should be the regex expression
                        - $equal: checks for equality of the field with the given value
                            the value should be the value to check the field against
                        - $function: a (list of) boolean function to apply on the field
                            the value should be the function that gets a single parameter and returns a bool
                        - $or: a list of boolean queries to apply on the field and return the or of their results
                            the value should be a list of queries
                            * Currently, query of key '_bc' is not supported
                        - $and: a list of boolean queries to apply on the field and return the and of their results
                            the value should be a list of queries
                            * Currently, query of key '_bc' is not supported

            Examples
            --------
            Here, we will provide some examples for the `query` dictionary.
            In this example, want to filter based on the value being equal to 10
            >>> {'$equal': 10}

            In this example, we want to filter if value does not exist
            >>> {'$exists': 0}

            Now, we want to filter if the value is at depth 5.
            >>> {'$regex': r'(\.\w+){4}\.foo$'}

            In this example, we want to filter based on function that checks if the value is int
            >>> {'$function': lambda a: isinstance(a, int)}

            Let's have the value of `foo` to be 5 and the value of `bar` be either 6 or 8
            >>> {'$and': [{'foo': 5}, {'$or': [{'bar': 6}, {'bar': 8}]}]}

            """

            self.query = query

            # Parse the query and get the list of functions corresponding to the queries
            self._function_list = self._parser(self.query)

        def _parser(self, query: Dict) -> list:
            """Method to parse the query given and turn it into actions or a list of functions to be called.

            Parameters
            ----------
            query : Any
                A query to be parsed

            Returns
            -------
            A list of functions to be called

            """

            def helper(single_operator: str, value: Any) -> Callable[[Option], bool]:
                """Helper method to parse a single query by the single operator and its value given
                    and turn it into a function to be called.

                Parameters
                ----------
                single_operator : str
                    A single operator command string, starting with '$'
                value : Any
                    The value corresponding to the operator

                Returns
                -------
                A function corresponding to the operator

                """

                # If the operation is a function, return the wrapper with the function
                if single_operator == self.Operations.FUNCTION or single_operator == self.Operations.FUNCTION.value:
                    return self._function(value)
                # Do the rest of the operations
                elif single_operator == self.Operations.EXISTS or single_operator == self.Operations.EXISTS.value:
                    return self._exists(value)
                elif single_operator == self.Operations.REGEX or single_operator == self.Operations.REGEX.value:
                    return self._regex(value)
                elif single_operator == self.Operations.EQUAL or single_operator == self.Operations.EQUAL.value:
                    return self._equal(value)
                elif single_operator == self.Operations.OR or single_operator == self.Operations.OR.value:
                    return self._or(value)
                elif single_operator == self.Operations.AND or single_operator == self.Operations.AND.value:
                    return self._and(value)
                else:
                    raise AttributeError(f"Such operator {single_operator} does not exist for filtering/finding!")

            # Get the functions list
            function_list = [helper(operator, value) for operator, value in query.items()]

            return function_list

        def filter(self, x: Option) -> bool:
            """Method to perform the filtering on an Option value.

            This filtering is based on the query given in the constructor.

            Parameters
            ----------
            x : Option
                An Option value to perform the filtering

            Returns
            -------
            A boolean indicating whether or not all the filtering queries are satisfied

            """

            # Perform all the filtering on the current item
            filter_list = [func(x) for func in self._function_list]

            # Check if all the filters are satisfied
            satisfied = all(filter_list)

            return satisfied

        def _function(self, funcs: Union[FunctionType, List[FunctionType]]) -> Callable[[Option], bool]:
            """Wrapper function for a function to query on an Option value.

            Parameters
            ----------
            funcs : Union[FunctionType, List[FunctionType]]
                A (list of) functions to apply on the Option value. Must return boolean

            Returns
            -------
            A function that can be called on an Option value

            """

            def helper(x: Option) -> bool:
                """Function to be called on an Option value to apply an internal function.

                Parameters
                ----------
                x : Option
                    An Option value to apply the internal function to.

                Returns
                -------
                The boolean result of the application of function on x

                """

                result = x

                # Perform all the filtering functions
                for func in funcs:
                    result = result.filter(func)

                return result.is_defined()

            # Make sure the `funcs` is a list
            funcs = [funcs] if not isinstance(funcs, list) else funcs

            return helper

        def _exists(self, value: bool) -> Callable[[Option], bool]:
            """Operator for checking if a variable exists.

            Parameters
            ----------
            value : bool
                A boolean indicating whether we are dealing with existence or non-existence of the element

            Returns
            -------
            A function for filtering

            """

            def helper(x: Option) -> bool:
                """Helper function to decide whether or not Option x has an element.

                Parameters
                ----------
                x : Option
                    An Option value to check if exists or not

                Returns
                -------
                A boolean indicating whether or not the Option x has an element

                """

                # The result is the xnor of value and if the Option x is empty
                return \
                    not \
                        ((x.is_defined()) ^ value)

            return helper

        def _regex(self, regex: str) -> Callable[[Option], bool]:
            """Operator for checking a regex string on a value.

               Parameters
               ----------
               regex : str
                   A regex string to be checked on the Option value

               Returns
               -------
               A function for filtering

               """

            def helper(x: Option) -> bool:
                """Helper function to decide whether or not a regex satisfies the Option x element.

                Parameters
                ----------
                x : Option
                    An Option value to be checked

                Returns
                -------
                A boolean indicating whether or not the Option x satisfy the regex

                """

                # Check if the item is a string and then do regex filtering
                return \
                    x \
                        .filter(lambda d: isinstance(d, str)) \
                        .filter(lambda a: regex_checker.search(a) is not None) \
                        .is_defined()

            # Define the compiled regex
            regex_checker = re.compile(regex)

            return helper

        def _equal(self, value: Any) -> Callable[[Option], bool]:
            """Operator for checking if a variable is equal to the given value.

            Parameters
            ----------
            value : Any
                A value to check for equality against the current variable

            Returns
            -------
            A function for filtering

            """

            def helper(x: Option) -> bool:
                """Helper function to decide whether or not Option x is equal to `value`.

                Parameters
                ----------
                x : Option
                    An Option value to check for equality

                Returns
                -------
                A boolean indicating whether or not the Option x value is equal to `value`

                """

                # if a dictionary, convert and then compare
                # if a dictionary, it only makes sense to compare if x is a branch
                if isinstance(value, dict) and x.filter(lambda q: q.is_branch()).is_defined():
                    value_new = x.get().__class__(value)
                else:
                    value_new = value

                return x.exist(lambda item: item == value_new)

            return helper

        def _or(self, criteria: List) -> Callable[[Option], bool]:
            """Operator to 'or' the operators within.

            Parameters
            ----------
            criteria : List
                List containing the key/operators to be or-ed

            Returns
            -------
            A function for filtering

            """

            def helper(x: Option) -> bool:
                """Helper function to decide whether or not Option x satisfies or of some operators.

                Parameters
                ----------
                x : Option
                    An Option value containing the instance of the Panacea

                Returns
                -------
                A boolean indicating whether or not the Option x value satisfies any of the criteria

                """

                # Get the Panacea instance
                panacea = x.get()

                # Check all the criteria sequentially
                for criterion in modifiers:

                    # TODO: '_bc' does not work, fix it!
                    # If any of the results is true, the whole thing is true
                    if criterion.filter_check_self(panacea, '') is True:
                        return True

                return False

            # Make modification off of the list of filters
            modifiers: List[Modification] = [Modification(criterion) for criterion in criteria]

            return helper

        def _and(self, criteria: List) -> Callable[[Option], bool]:
            """Operator to 'and' the operators within.

            Parameters
            ----------
            criteria : List
                List containing the key/operators to be and-ed

            Returns
            -------
            A function for filtering

            """

            def helper(x: Option) -> bool:
                """Helper function to decide whether or not Option x satisfies and of some operators.

                Parameters
                ----------
                x : Option
                    An Option value containing the instance of the Panacea

                Returns
                -------
                A boolean indicating whether or not the Option x value satisfies all the criteria

                """

                # Get the Panacea instance
                panacea = x.get()

                # Check all the criteria sequentially
                for criterion in modifiers:

                    # TODO: '_bc' does not work, fix it!
                    # If any of the results is false, the whole thing is false
                    if criterion.filter_check_self(panacea, '') is False:
                        return False

                return True

            # Make modification off of the list of filters
            modifiers: List[Modification] = [Modification(criterion) for criterion in criteria]

            return helper

    class Update:
        """A class that parses, holds and updates the update rules for Panacea."""

        # List of available operations
        class Operations(Enum):
            UNSET = '$unset'
            SET = '$set'
            SET_ONLY = '$set_only'
            SET_ON_INSERT = '$set_on_insert'
            UPDATE_ONLY = '$update_only'
            UPDATE = '$update'
            RENAME = '$rename'
            INC = '$inc'
            MULT = '$mult'
            FUNCTION = '$function'
            MAP = '$map'
            MAP_ON_VALUE = '$map_on_value'
            MAP_ON_VALUE_THREAD = '$map_on_value_thread'

        # List of available conditionals
        class Conditionals(Enum):
            DEEP = 'deep'
            THREAD_MAX_COUNT = 'thread_max_count'
            PROCESS_MAX_COUNT = 'process_max_count'
            RECURSIVE = 'recursive'

        def __init__(self, query: Dict, condition_dict: Dict = None):
            """Initializer to the class which will parse the update rule/query and create the corresponding actions.

            Parameters
            ----------
            query : Dict
                The query to do the updating to be parsed
                Right now, only these sub-queries/operators are supported:
                    - set or update an entry with the operator $set
                    - only set a non-existing entry with the operator $set_only
                    - only update an existing entry with the operator $update
                    - rename an (existing) entry with the operator $rename
                    - increment an (existing) entry with the operator $inc
                    - multiply an (existing) entry with the operator $mult
                    - lambda functions with the operator $function
                The update rule/query has to be a dictionary with key containing the operator
                and value the corresponding value

                An example of the query is:
                    {'$set': {'epochs': 10, 'other_thing': 'hello'}, '$inc': {'batch': 5}}
                    which would set 'epochs' and 'other_thing' fields to 10 and 'hello'
                        and will increment 'batch' field by 5

                The query containing the update rules to be parsed
                    The keys of the dictionary are among these options:
                        - the name of field: this will cause the filtering to be based on this specific field
                        - special names that start with '_': these will cause the filtering to be based on special
                            things specified by this key.
                            A complete list of them are:
                                - _bc: filter on the breadcrumb of the name of the fields
                                - _self: filter on the current item itself
                                - _value: filter based on the value of the leaf nodes only
                    The value must be a dictionary with its keys being the operators of the Filter class and its values
                        be the corresponding expression.
                        If no dictionary is given for value, $equal operator is assumed.
                        Supported operators are:
                            - $exists: checks whether a field exists
                                the value can be 1 to denote the existence of the field and 0 otherwise
                            - $regex: checks a string against a regex expression
                                the value should be the regex expression
                            - $equal: checks for equality of the field with the given value
                                the value should be the value to check the field against
                            - $function: a boolean function to apply on the field
                                the value should be the function that gets a single parameter and returns a bool

            condition_dict : dict, optional
            Dictionary containing the condition of update.
                The keys/value possible pair are among these options:
                    - 'thread_max_count': int: the maximum number of threads to use for multithreading updates.
                    - 'process_max_count': int: the maximum number of processes to use for multithreading updates.

            """

            self.query = query

            # Parse the query and get the dictionary of functions corresponding to the queries
            self._query_dict: Dict[str, Callable[[Option], PanaceaBase]] = self._parser(self.query)

            self.condition_dict = self._process_condition(condition_dict or {})

        def _process_condition(self, condition_dict):

            condition_dict[self.Conditionals.PROCESS_MAX_COUNT.value] = \
                min(multiprocessing.cpu_count(), condition_dict.get(self.Conditionals.PROCESS_MAX_COUNT.value) or 5)

            condition_dict[self.Conditionals.THREAD_MAX_COUNT.value] = \
                min(multiprocessing.cpu_count() * 5, condition_dict.get(self.Conditionals.THREAD_MAX_COUNT.value) or 5)

            return condition_dict

        def _parser(self, query: Dict) -> Dict[str, Callable[[Option], PanaceaBase]]:
            """Method to parse the query given and turn it into actions or a list of functions to be called.

            Parameters
            ----------
            query : Any
                A query to be parsed

            Returns
            -------
            A dictionary of modified query with functions to be called

            """

            # If query dictionary is empty, return it
            if not query:
                return query

            def helper(single_operator: str, update_dict: Dict) -> Dict[str, Callable[[Option], PanaceaBase]]:
                """Helper method to parse a single query by the single operator and its value given
                    and turn it into a function to be called.

                Parameters
                ----------
                single_operator : str
                    A single operator command string, starting with '$'
                update_dict : dict
                    The update corresponding to the operator
                        It should have field names as keys and corresponding operator value as value

                Returns
                -------
                A function corresponding to the operator

                """

                # If the operation is $operator, set the function to its corresponding wrapper
                if single_operator == self.Operations.UNSET.value or \
                        single_operator == self.Operations.UNSET:
                    function = self._unset
                elif single_operator == self.Operations.SET.value or \
                        single_operator == self.Operations.SET:
                    function = self._set
                elif single_operator == self.Operations.SET_ONLY.value or \
                        single_operator == self.Operations.SET_ONLY:
                    function = self._set_only
                elif single_operator == self.Operations.SET_ON_INSERT.value or \
                        single_operator == self.Operations.SET_ON_INSERT:
                    function = self._set_on_insert
                elif single_operator == self.Operations.UPDATE_ONLY.value or \
                        single_operator == self.Operations.UPDATE_ONLY:
                    function = self._update_only
                elif single_operator == self.Operations.UPDATE.value or \
                        single_operator == self.Operations.UPDATE:
                    function = self._update
                elif single_operator == self.Operations.RENAME.value or \
                        single_operator == self.Operations.RENAME:
                    function = self._rename
                elif single_operator == self.Operations.INC.value or \
                        single_operator == self.Operations.INC:
                    function = self._inc
                elif single_operator == self.Operations.MULT.value or \
                        single_operator == self.Operations.MULT:
                    function = self._mult
                elif single_operator == self.Operations.FUNCTION.value or \
                        single_operator == self.Operations.FUNCTION:
                    function = self._function
                elif single_operator == self.Operations.MAP.value or \
                        single_operator == self.Operations.MAP:
                    function = self._map
                elif single_operator == self.Operations.MAP_ON_VALUE.value or \
                        single_operator == self.Operations.MAP_ON_VALUE:
                    function = self._map_on_value
                elif single_operator == self.Operations.MAP_ON_VALUE_THREAD.value or \
                        single_operator == self.Operations.MAP_ON_VALUE_THREAD:
                    function = self._map_on_value_thread
                # TODO: FIx multiprocessing, it does not currently work
                # elif single_operator == '$map_on_value_process':
                #     function = self._map_on_value_process
                else:
                    raise AttributeError(f"Such operator '{single_operator}' does not exist for updating!")

                # Modify the update dictionary with the corresponding function
                modified_update_dict = {key: function(value) for key, value in update_dict.items()}

                return modified_update_dict

            # Check if all update rules are mutually exclusive
            from collections import Counter
            from functools import reduce
            count = Counter(
                reduce(lambda x, y: x + y,
                       # Go over the value of each of the items in the query dictionary
                       # Then, take the keys of each of the elements, which is a dictionary
                       [list(value.keys()) for _, value in query.items()]
                       )
            )
            # Check if all the fields are declared only once
            for key, value in count.items():
                if value >= 2:
                    raise ValueError(f"Conflict in update dictionary for key {key}: duplicate update rules.")

            # Get the functions list
            function_list: List[Dict[str, Callable[[Option], PanaceaBase]]] = \
                [helper(operator, value) for operator, value in query.items()]

            # Turn all the lists, into a single dictionary with instructions
            function_dict = reduce(lambda x, y: {**x, **y}, function_list)

            return function_dict

        def get_modified_query(self) -> Dict[str, Callable[[Option], PanaceaBase]]:
            """Returns the modified update query dictionary constructed in the initializer.

            Returns
            -------
            Modified dictionary of the update query

            """

            return self._query_dict

        def _unset(self, value: Any) -> Callable[[str, Option], Option]:
            """Wrapper function for unsetting a value on an Option value.

            Parameters
            ----------
            value : Any
                A value to set to the Option value.
                    If Option value exists, i.e. it is a PanaceaBase, remove it
                    If Option value does not exist, do nothing

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> nothing:
                """Function to be called on an Option value to unset a value.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to unset the value

                Returns
                -------
                Option, nothing, to remove the value

                """

                return nothing

            return helper

        def _set(self, value: Any) -> Callable[[str, Option], Some]:
            """Wrapper function for setting a value on an Option value.

            Parameters
            ----------
            value : Any
                A value to set to the Option value.
                    If Option value exists, i.e. it is a PanaceaBase, update it
                    If Option value does not exist, set the value

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Some:
                """Function to be called on an Option value to set a value.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to set the value

                Returns
                -------
                Option, Some, (key, value) pair with the set value

                """

                result = Some((key, value))

                return result

            return helper

        def _set_only(self, value: Any) -> Callable[[str, Option], Some]:
            """Wrapper function for setting a value on an Option value that does not exist.

            Parameters
            ----------
            value : Any
                A value to set to the Option value.
                    If Option value exists, i.e. it is a PanaceaBase, raise error
                    If Option value does not exist, set the value

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Some:
                """Function to be called on an Option value, that has to be nothing, to set a value.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value, which has to be nothing

                Returns
                -------
                Option, Some, (key, value) pair with the set value

                """

                # if the value exists
                if x.is_defined():
                    raise ValueError(f"The value '{key}' to `{self.Operations.SET_ONLY.value}` exists!")

                # the good case!
                else:
                    result = Some((key, value))

                return result

            return helper

        def _set_on_insert(self, value: Any) -> Callable[[str, Option], Option]:
            """Wrapper function for setting a value on an Option value.

            Parameters
            ----------
            value : Any
                A value to set to the Option value.
                    If Option value exists, i.e. it is a PanaceaBase, raise error
                    If Option value does not exist, ignore

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value to set a value.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value

                Returns
                -------
                Option of (key, value) pair with the set value

                """

                result = x.map(lambda a: (key, a)).or_else(Some((key, value)))

                return result

            return helper

        def _update_only(self, value: Any) -> Callable[[str, Option], Some]:
            """Wrapper function for updating a value on an Option value that does exist.

            Parameters
            ----------
            value : Any
                A value to update the Option value with.
                    If Option value exists, i.e. it is a PanaceaBase, update its value
                    If Option value does not exist, raise an error

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Some:
                """Function to be called on an Option value, that has to be Some, to update it.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value, which has to be Some

                Returns
                -------
                Option, Some, (key, value) pair with the updated value

                """

                # if the value does not exist
                if x.is_empty():
                    raise ValueError(f"The value '{key}' to `{self.Operations.UPDATE_ONLY.value}` does not exist!")

                # the good case!
                result = Some((key, value))

                return result

            return helper

        def _update(self, value: Any) -> Callable[[str, Option], Option]:
            """Wrapper function for updating a value on an Option value that does exist and ignore if not exist.

            Parameters
            ----------
            value : Any
                A value to update the Option value with.
                    If Option value exists, i.e. it is a PanaceaBase, update its value
                    If Option value does not exist, ignore

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value to update it.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value

                Returns
                -------
                Option of (key, value) pair with the updated value

                """

                result = x.map(lambda a: (key, value))

                return result

            return helper

        def _rename(self, value: str) -> Callable[[str, Option], Option]:
            """Wrapper function for renaming a value name on an Option value.

            Parameters
            ----------
            key_new : str
                The old name of the value to be renamed

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value to rename its key if available.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value, which has to be Some

                Returns
                -------
                Option of (key, value) pair with updated key.

                """

                result = x.map(lambda z: (value, z))

                return result

            return helper

        def _inc(self, value: Any) -> Callable[[str, Option], Option]:
            """Wrapper function for updating a value on an Option value by adding `value` to it.

            Parameters
            ----------
            value : Any
                A value to add to the Option value.

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value, to add `value` to it.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to add `value` to

                Returns
                -------
                Option value with the updated value

                """

                result = x.map(lambda a: (key, a + value))

                return result

            return helper

        def _mult(self, value: Any) -> Callable[[str, Option], Option]:
            """Wrapper function for updating a value on an Option value by multiplying it by `value`.

            Parameters
            ----------
            value : Any
                A value to multiply by the Option value.

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value, to multiply `value` by it.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to multiply `value` by

                Returns
                -------
                Option value with the updated value

                """

                result = x.map(lambda a: (key, a * value))

                return result

            return helper

        def _function(self, value: Union[FunctionType, List[FunctionType]]) -> Callable[[str, Option], Option]:
            """Wrapper function for updating a value on an Option value by applying the function `func` to it.

            Parameters
            ----------
            funcs: Union[FunctionType, List[FunctionType]]
                The (list of) function to apply to the Option value

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value, to apply `func` to its value.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to apply `func` to

                Returns
                -------
                Option value with the updated value

                """

                result = x

                # Apply all the functions in order
                for func in funcs:
                    result = result.map(func)

                result = result.map(lambda a: (key, a))

                return result

            # Make sure the `funcs` is a list
            funcs = [value] if not isinstance(value, list) else value

            return helper

        def _map(self, funcs: Union[FunctionType, List[FunctionType]]) -> Callable[[str, Option], Option]:
            """Wrapper function for updating a value on an Option value 'by mapping' the function `func` to the
            PanaceaBase instance within the Option value.

            Parameters
            ----------
            funcs: Union[FunctionType, List[FunctionType]]
                The (list of) function to map to the Option value

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value, to 'map' `func` to it

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to 'map' `func` to, the object within the Option has to have a map method, such as
                        a PanaceaBase

                Returns
                -------
                Option value with the updated value

                """

                result = x

                # Apply all the functions in order
                for func in funcs:
                    result = result.map(lambda a: a.map(func))

                return result.map(lambda a: (key, a))

            # Make sure the `funcs` is a list
            funcs = [funcs] if not isinstance(funcs, list) else funcs

            return helper

        def _map_on_value(self, funcs: Union[FunctionType, List[FunctionType]]) -> Callable[[str, Option], Option]:
            """Wrapper function for updating a value on an Option value by mapping the functions `funcs` to the
            iterable element within the Option value. Thus, the given value has to be a leaf.

            Note that the returned value is always a LIST.

            The difference of this method and `_map` is that `_map` calls the method `map` on the received object
                whereas the current method `_map_on_value` does the mapping on ITERABLE object itself within the
                received object, which must be a PanaceaLeaf.

                For example, if the input is PanacecLeaf([45, 46, 47]]) and we want to apply a function `func` to it,
                    the method `map` will call `PanaceaLeaf([45, 46, 47]).map(func)`
                    the method `map_on_value` will perform `map(func, [45, 46, 47])`

            Parameters
            ----------
            funcs: Union[FunctionType, List[FunctionType]]
                The (list of) function to map to the iterable element within Option value

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value, to 'map' `func` to it. It should be noted that the object
                    entrapped within the Option value has to be iterable.

                    Note that the returned value is always a LIST.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to 'map' `func` to, the value within has to be either
                        - a leaf containing an iterable
                        - an iterable
                        If the value inside is neither, will ignore

                Returns
                -------
                Option value with the updated value which is a LIST

                """

                def extra_help(item):
                    """Additional helper function to be passed as a new function to the `_map` method.

                    Parameters
                    ----------
                    item : Any
                        An iterable item (that is inside a leaf node)

                    Returns
                    -------
                    A list of the result with the closure-ized functions applied

                    """

                    result = item

                    # Apply all the functions in order
                    for func in funcs:
                        result = (func(item) for item in result)

                    return list(result)

                # If the internal value for Option is a leaf
                if x.is_defined():
                    if issubclass(type(x.get), PanaceaLeaf):

                        # Check if the item is a leaf and get its internal value
                        # result = x.filter(lambda a: a.is_leaf())

                        # Pose this problem as a map problem on the leaf
                        return self._map(extra_help)(key, x)

                    # If the internal value is not a node but an iterable
                    else:

                        return Some((key, extra_help(x.get())))

                return x

            # Make sure the `funcs` is a list
            funcs = [funcs] if not isinstance(funcs, list) else funcs

            return helper

        # TODO: Try to fix this, the following does not work!
        def _map_on_value_process(self, funcs: Union[FunctionType, List[FunctionType]]) -> Callable[
            [str, Option], Option]:
            """Wrapper function for updating a value on an Option value by mapping the functions `funcs` to the
            iterable element within the Option value using multiprocessing. Thus, the given value has to be a leaf.

            This method is just like the method `map_on_value` but utilizes multithreading.
            The max number of workers can be given using the `condition_dict` in the initializer.

            For more info, refer to the method `map_on_value`.

            Parameters
            ----------
            funcs: Union[FunctionType, List[FunctionType]]
                The (list of) function to map to the iterable element within Option value

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value, to 'map' `func` to it using multiprocessing.
                    It should be noted that the object entrapped within the Option value has to be iterable.

                    Note that the returned value is always a LIST.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to 'map' `func` to, the value within has to be a leaf containing an iterable
                        If the value inside is not a leaf, will ignore

                Returns
                -------
                Option value with the updated value which is a LIST

                """

                def extra_help(item):
                    """Additional helper function to be passed as a new function to the `_map` method.

                    Parameters
                    ----------
                    item : Any
                        An iterable item (that is inside a leaf node)

                    Returns
                    -------
                    A list of the result with the closure-ized functions applied

                    """

                    result = item

                    with multiprocessing.Pool(self.condition_dict.get(self.Conditionals.PROCESS_MAX_COUNT.value)) \
                            as executor:
                        # Apply all the functions in order
                        for func in funcs:
                            result = executor.map(func, result)

                    return list(result)

                # Check if the item is a leaf and get its internal value
                result = x.filter(lambda a: a.is_leaf())

                # Pose this problem as a map problem on the leaf
                return self._map(extra_help)(key, result)

            # Make sure the `funcs` is a list
            funcs = [funcs] if not isinstance(funcs, list) else funcs

            return helper

        def _map_on_value_thread(self, funcs: Union[FunctionType, List[FunctionType]]) -> Callable[
            [str, Option], Option]:
            """Wrapper function for updating a value on an Option value by mapping the functions `funcs` to the
            iterable element within the Option value using multithreading. Thus, the given value has to be a leaf.

            This method is just like the method `map_on_value` but utilizes multithreading.
            The max number of workers can be given using the `condition_dict` in the initializer.

            For more info, refer to the method `map_on_value`.

            Parameters
            ----------
            funcs: Union[FunctionType, List[FunctionType]]
                The (list of) function to map to the iterable element within Option value

            Returns
            -------
            A function that can be called on an Option (key, value) pair, where value is PanaceaBase

            """

            def helper(key: str, x: Option) -> Option:
                """Function to be called on an Option value, to 'map' `func` to it using multithreading.
                    It should be noted that the object entrapped within the Option value has to be iterable.

                    Note that the returned value is always a LIST.

                Parameters
                ----------
                key : str
                    Name of the Option value
                x : Option
                    An Option value to 'map' `func` to, the value within has to be either
                        - a leaf containing an iterable
                        - an iterable
                        If the value inside is neither, will ignore

                Returns
                -------
                Option value with the updated value which is a LIST

                """

                def extra_help(item):
                    """Additional helper function to be passed as a new function to the `_map` method.

                    Parameters
                    ----------
                    item : Any
                        An iterable item (that is inside a leaf node)

                    Returns
                    -------
                    A list of the result with the closure-ized functions applied

                    """

                    result = item

                    with ThreadPoolExecutor(self.condition_dict.get(self.Conditionals.THREAD_MAX_COUNT.value)) \
                            as executor:
                        # Apply all the functions in order
                        for func in funcs:
                            result = executor.map(func, result)

                    return list(result)

                # If the internal value for Option is a leaf
                if x.is_defined():
                    if issubclass(type(x.get), PanaceaLeaf):

                        # Check if the item is a leaf and get its internal value
                        # result = x.filter(lambda a: a.is_leaf())

                        # Pose this problem as a map problem on the leaf
                        return self._map(extra_help)(key, x)

                    # If the internal value is not a node but an iterable
                    else:

                        return Some((key, extra_help(x.get())))

                return x

            # Make sure the `funcs` is a list
            funcs = [funcs] if not isinstance(funcs, list) else funcs

            return helper

    # General traversals

    def traverse(self,
                 panacea: PanaceaBase,
                 bc: str,
                 do_after_satisfied: Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]],
                 propagate: Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]
                 ) \
            -> Option[(str, PanaceaBase)]:
        """Traverses the tree of panacea, checks if it meets the criteria and does operations after that, or propagates
        the operation to its children.

        Parameters
        ----------
        panacea : PanaceaBase
            The PanaceaBase instance to traverse through
        bc : str
            The breadcrumb so far
        do_after_satisfied : : Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]
            The function to be called if panacea meets the filtering criteria
        propagate: Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]
            The function to be called if panacea did not meet the filtering criteria.
            This function should propagate the desired operation to the children of the instance or return

        Returns
        -------
        An Option value of type Option[(str, PanaceaBase)] where (str, PanaceaBase) is the key/value pair of the result.

        """

        # Get the current instance key
        key = bc.split('.')[-1]

        # Check if self is satisfied
        if self.filter_check_self(panacea, bc) is True:
            return do_after_satisfied(key, panacea)  # Provide (key, value) pair as input

        # If self is not satisfied, propagate
        else:
            return propagate(key, panacea)

    def propagate_all(self,
                      function_to_call_for_each_element: Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]) \
            -> Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]:
        """Propagation function that goes through all the children of the panacea instance.

        Parameters
        ----------
        function_to_call_for_each_element : Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]
            A function to be called on each of the children elements during the propagation

        Returns
        -------
        A function of type Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]] that can be called to propagate the
        operation through all children of the panacea instance

        """

        def helper(key: str, panacea: PanaceaBase) -> Option[(str, PanaceaBase)]:
            """Helper method to do the propagation and takes care of closure of function_to_call_for_each_element.

            It calls function_to_call_for_each_element on the (key, panacea) pair given and returns the result as an
            Option value of the PanaceaBase class.

            Parameters
            ----------
            key : str
                The name of the key of the given panacea
            panacea : PanaceaBase
                The panacea element

            Returns
            -------
            An Option value of the type Option[(str, PanaceaBase)] containing the propagation result of all the children

            """

            # Construct a new dictionary for making a new class whose values are generated by
            # function_to_call_for_each_element function
            new_dict = \
                {
                    item.get()[0]: item.get()[1]  # Each returned element is (key, value) pair
                    for item
                    in
                    [  # Process each of the parameters, results in Option value containing (key, value) pairs
                        function_to_call_for_each_element(key, value)
                        for key, value
                        in panacea.get_parameters().items()
                    ]
                    if item.is_defined()
                }

            # TODO: Check if the dictionary has not changed, do not produce a new instance

            # If the processing resulted in a valid case, make a new class and return it, otherwise, nothing
            # this is the same if panacea was empty in the first place! otherwise, we would remove empty panaceas
            if new_dict or panacea.is_empty():
                return Some((key, panacea.__class__(new_dict)))
            else:
                return nothing

        return helper

    def propagate_one(self,
                      function_to_call_for_each_element: Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]) \
            -> Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]:
        """Propagation function that goes through the children of the panacea instance and returns the first one that
        matches function_to_call_for_each_element.

        Parameters
        ----------
        function_to_call_for_each_element : Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]]
            A function to be called on each of the children elements during the propagation

        Returns
        -------
        A function of type Callable[[str, PanaceaBase], Option[(str, PanaceaBase)]] that can be called to propagate the
        operation through the children of the panacea instance to return the first child that matches according to
        function_to_call_for_each_element.

        """

        def helper(key: str, panacea: PanaceaBase) -> Option[(str, PanaceaBase)]:
            """Helper method to do the propagation and takes care of closure of function_to_call_for_each_element.

            It calls function_to_call_for_each_element on the (key, panacea) pair given and returns the first result
            found as an Option value

            Parameters
            ----------
            key : str
                The name of the key of the given panacea, not currently used
            panacea : PanaceaBase
                The panacea element

            Returns
            -------
            An Option value of the type Option[(str, PanaceaBase)] containing the propagation result of the first child
            that matches according to function_to_call_for_each_element.

            """

            # Go over each children of the panacea instance and return the result as soon as it is a match
            for key, value in panacea.get_parameters().items():

                # Get the results of finding in the current parameter
                result: Option = function_to_call_for_each_element(key, value)

                # If a result is found, break and do not go over other parameters
                if result.is_defined():
                    return result

            # If none of the children matches, return nothing
            return nothing

        return helper

    # Filter

    def make_filter_dictionary(self, filter_dict: Dict) -> Dict:
        """Method to create a dictionary from the given `filter_dict` whose values are instances of Filter class.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to `filter` method for more information

        Returns
        -------
        A dictionary with the same keys as the input dictionary but values of Filter class instance

        """

        # Replace the key/value pairs whose value is not a dictionary with the '$equal' operator for the value
        filter_dict_modified = {
            # Elements that are already in dictionary form that Filter class accept
            **{key: value for key, value in filter_dict.items()
               if isinstance(value, dict) and not key.startswith('$')},
            # Elements that are not in dictionary form are set to '$equal' operator for the Filter class
            **{key: {self.Filter.Operations.EQUAL.value: value} for key, value in filter_dict.items()
               if not isinstance(value, dict) and not key.startswith('$')}
        }
        # Add operator keys
        operators_dict = {key: value for key, value in filter_dict.items() if key.startswith('$')}
        if operators_dict:
            self_dict = filter_dict_modified.get('_self') or {}
            filter_dict_modified['_self'] = {**self_dict, **operators_dict}

        # Process the criteria for each of the filter_dict fields into an instance of the Filter class
        processed_filter_dict: Dict = {key: self.Filter(value) for key, value in filter_dict_modified.items()}

        # Split the dictionary into two keys: `field` and `_special`
        # The `field` key contains all the selectors corresponding to the name of the fields
        # The `_special` key contains all the special selectors that start with '_'
        processed_filter_dict = \
            {
                'field': {key: value for key, value in processed_filter_dict.items() if not key.startswith('_')},
                '_special': {key: value for key, value in processed_filter_dict.items() if key.startswith('_')}
            }

        return processed_filter_dict

    def filter_check_self(self, panacea: PanaceaBase, bc: str = '') -> bool:
        """Method to check if `panacea` satisfies the criteria addressed by self.filter_dict.

        Parameters
        ----------
        panacea : PanaceaBase
            PanaceaBase instance to check for meeting the criteria
        bc : str
            The breadcrumb so far

        Returns
        -------
        A boolean indicating the result

        """

        # Load the filter dictionary
        filter_dict = self.filter_dict

        # By default, self satisfies all the criteria
        satisfied = True

        # Check if all filter's field items are satisfied
        for key, filter in filter_dict.get('field').items():
            satisfied &= filter.filter(panacea.get_option(key))

            # If filter is not satisfied, return false
            if satisfied is False:
                return False

        # Check special items that start with _
        satisfied &= \
            filter_dict \
                .get('_special') \
                .get('_bc') \
                .filter(Some(bc)) \
                if filter_dict.get('_special').get('_bc') is not None \
                else True
        satisfied &= \
            filter_dict \
                .get('_special') \
                .get('_self') \
                .filter(Some(panacea)) \
                if filter_dict.get('_special').get('_self') is not None \
                else True
        satisfied &= \
            filter_dict \
                .get('_special') \
                .get('_key_name') \
                .filter(Some(bc.split('.')[-1])) \
                if filter_dict.get('_special').get('_key_name') is not None \
                else True

        # For special item '_value', `panacea` has to be a leaf
        satisfied &= \
            filter_dict \
                .get('_special') \
                .get('_value') \
                .filter(panacea.get_option('_value').filter(lambda x: issubclass(type(panacea), PanaceaLeaf))) \
                if filter_dict.get('_special').get('_value') is not None \
                else True

        return satisfied

    def filter(self, panacea: PanaceaBase, bc: str = '') -> Option[(str, PanaceaBase)]:
        """Method to perform the filtering on `panacea`.

        It prunes the `panacea` tree (in case of branch node) and returns only its branches that satisfy the filtering
        criteria

        Parameters
        ----------
        panacea : PanaceaBase
            The PanaceaBase instance to perform the filtering on
        bc : str, optional
            The breadcrumb so far

        Returns
        -------
        The result of filtering in form of Option value of type Option[(str, PanaceaBase)] where (str, PanaceaBase) is
        the key/value pair of the result.

        """

        # For each of the elements in the propagation, we should do the filtering again with updated bc
        for_each_element = \
            lambda key, value: self.filter(panacea=value, bc=f'{bc}.{key}')

        # For propagation
        # Remember that propagation happens when the instance panacea did not meet the filtering criteria
        # If the given `panacea` is a branch node, propagate again
        # If the given `panacea` is a leaf node, return nothing, as there is no more children to propagate and
        # the `panacea` instance did not meet the filtering criteria
        if issubclass(type(panacea), Panacea):
            propagate = self.propagate_all(for_each_element)
        elif issubclass(type(panacea), PanaceaLeaf):
            propagate = lambda x, key: nothing
        else:
            raise AttributeError(
                f"What just happened?! The tree has to have only nodes or leaves, got type {type(panacea)}"
            )

        # If the `panacea` instance met the filtering criteria
        # After that, do nothing, just return it in the compatible form
        do_after_satisfied = lambda key, panacea: Some((key, panacea))

        # Do the traversing with the correct functions
        return self.traverse(panacea=panacea, bc=bc, do_after_satisfied=do_after_satisfied, propagate=propagate)

    def find_one(self, panacea: PanaceaBase, bc: str = '') -> Option[Any]:
        """Method to find the very first single element on `panacea` that satisfies the filtering criteria and returns
        it.

        Parameters
        ----------
        panacea : PanaceaBase
            The PanaceaBase instance to search for
        bc : str, optional
            The breadcrumb so far

        Returns
        -------
        The result of finding the very single element that satisfies the filtering criteria in form of Option value.

        """

        # For each of the elements in the propagation, we should do the finding one again with updated bc
        for_each_element = \
            lambda key, value: self.find_one(panacea=value, bc=f'{bc}.{key}')

        # For propagation
        # Remember that propagation happens when the instance panacea did not meet the filtering criteria
        # If the given `panacea` is a branch node, propagate again
        # If the given `panacea` is a leaf node, return nothing, as there is no more children to propagate and
        # the `panacea` instance did not meet the filtering criteria
        if issubclass(type(panacea), Panacea):
            propagate = self.propagate_one(for_each_element)
        elif issubclass(type(panacea), PanaceaLeaf):
            propagate = lambda x, key: nothing
        else:
            raise AttributeError(f"What just happened?! got of type {type(panacea)}")

        # If the `panacea` instance met the filtering criteria
        # After that, if it is a branch node, return it as Option value, and if it is a leaf node, return its value
        # as Option value
        do_after_satisfied = lambda key, x: Some(x) if issubclass(type(x), Panacea) else Some(x.get())

        # Do the traversing with the correct functions
        return self.traverse(panacea=panacea, bc=bc, do_after_satisfied=do_after_satisfied, propagate=propagate)

    # Update

    def make_update_dictionary(self, update_dict: Dict) -> Dict:
        """Method to create a dictionary from the given `update_dict` whose values are instances of Update class.

        Parameters
        ----------
        update_dict : dict
            Dictionary containing the updating rules

        Returns
        -------
        A dictionary with keys as fields and values as Update class methods

        """

        # placeholder
        processed_update_dict = {}

        # Replace the key/value pairs whose value is not a dictionary with '$set' operator
        modified_update_dict = {
            # Elements that are already in dictionary form that Update class accept assuming operators are fine
            **{key: value for key, value in update_dict.items() if isinstance(value, dict)}
        }
        # Elements that are not in dictionary form are set to '$set' operator for the Update class
        set_dict = {
            key: value for key, value in update_dict.items() if not isinstance(value, dict)
        }
        # If the '$set' operator already exists, extend it, otherwise, put it there
        modified_update_dict[self.Update.Operations.SET] = \
            {**(modified_update_dict.get(self.Update.Operations.SET) or {}), **set_dict}

        # if doing recursive, extract the recursive parts
        if self.condition_dict.get('recursive') is True:
            recursive_dict = {}
            for operation, fields in modified_update_dict.items():
                for field, operand in fields.items():
                    if isinstance(operand, dict):
                        recursive_dict[operation] = recursive_dict.get(operation) or {}
                        recursive_dict[operation][field] = operand

            for operation, fields in recursive_dict.items():
                for field, operand in fields.items():
                    if isinstance(operand, dict):
                        del modified_update_dict[operation][field]

            processed_update_dict['_recursive'] = recursive_dict

        # Process the update_dict and get a modified update dictionary
        processed_update_dict: Dict = \
            {
                **processed_update_dict,
                **self.Update(modified_update_dict, condition_dict=self.condition_dict).get_modified_query()
            }

        # Split the dictionary into two keys: `field` and `_special`
        # The `field` key contains all the selectors corresponding to the name of the fields
        # The `_special` key contains all the special selectors that start with '_'
        processed_update_dict = \
            {
                'field': {key: value for key, value in processed_update_dict.items() if not key.startswith('_')},
                '_special': {key: value for key, value in processed_update_dict.items() if key.startswith('_')}
            }

        return processed_update_dict

    def update_self(self, key: str, panacea: PanaceaBase, update_dict: Dict = None) -> (str, PanaceaBase):
        """Method to update the `panacea` instance based on the update rules and returns the result.

        Parameters
        ----------
        key : str
            the key name
        panacea : PanaceaBase
            PanaceaBase instance to update
        update_dict : Dict, optional
            Optional update dictionary to process, if not given, the one set in constructor will be used

        Returns
        -------
        A new copy of the `panacea` instance with updated properties

        """

        # Load the update dictionary
        update_dict = self.update_dict if update_dict is None else update_dict

        # Dummy variable
        new_panacea = panacea
        new_key = key

        # First the field update rules are applied
        # Then, on the result, _special rules are applied

        if issubclass(type(panacea), Panacea):

            # Get the updated field items
            modified_panacea_dictionary = {
                item.get()[0]: item.get()[1]  # Each returned element is (key, value) pair
                for item
                in
                [  # Process each of the parameters, results in Option value containing (key, value) pairs
                    update(key, panacea.get_option(key))
                    for key, update
                    in update_dict.get('field').items()  # items that should select specific fields
                    if '.' not in key
                ]
                if item.is_defined()
            }

            # Get the updates on \w*(\.\w+)+ update rules
            modified_panacea_deep_dictionary = {
                key: value
                for key, value
                in
                {
                    key.split('.')[0]: self.update_self(
                        key,
                        panacea.get_option(key.split('.')[0]).get_or_else(panacea.__class__()),
                        {'field': {key[(key.index('.') + 1):]: update}, '_special': {}}
                    )[1]
                    for key, update
                    in update_dict.get('field').items()
                    if '.' in key
                }.items()
                if not value.is_empty()
            }

            # Create a new dictionary in order to make a new instance
            # The items that are not modified by the update rules should reappear as-is
            new_panacea_dict = {
                # Items not modified by the update rules
                **{key: value for key, value in panacea.get_parameters().items() if
                   key not in update_dict.get('field').keys()},
                # Items modified by the update rules
                **modified_panacea_dictionary,
                **modified_panacea_deep_dictionary
            }

            # Generate a new class from the modified parameters
            new_panacea = panacea.__class__(new_panacea_dict)

        # Apply the special updates
        if update_dict.get('_special').get('_recursive') is not None:
            # go over every field and update
            for operation, fields in update_dict.get('_special').get('_recursive').items():
                for field, operand in fields.items():
                    # we go one level down in nested given dictionary as operand, get the result, update
                    updated_panacea = \
                        new_panacea\
                        .get_option(field)\
                        .get_or_else(new_panacea.__class__())
                    name = key if not isinstance(operand, dict) else field
                    updated_key, updated_panacea = \
                        self.update_self(name, updated_panacea, self.make_update_dictionary({operation: operand}))
                    new_panacea = \
                        new_panacea\
                        .update(
                            {},
                            {
                                self.Update.Operations.UNSET: {name: 1},
                            }
                        )
                    if not updated_panacea.is_empty():
                        new_panacea = \
                            new_panacea \
                            .update(
                                {},
                                {
                                    self.Update.Operations.SET: {updated_key: updated_panacea},
                                }
                            )
        if update_dict.get('_special').get('_key_name') is not None:
            # Apply the update rule
            new_key: str = update_dict.get('_special').get('_key_name')('', Some(key)).get()[1]
            if not isinstance(new_key, str):
                raise ValueError('operation on key name resulted in a non string variable.')
        if update_dict.get('_special').get('_self') is not None:
            # Apply the update rule
            # Note that the result can be anything, anything that the user says, not necessarily a leaf
            new_panacea: Any = update_dict.get('_special').get('_self')('', Some(new_panacea)).get()[1]

        # If we have a `_value' item and new_panacea (after all the updates yet) is a leaf, update it
        if update_dict.get('_special').get('_value') is not None and issubclass(type(new_panacea), PanaceaLeaf):

            # Get the result of applying the update rule to the internal value
            result = update_dict.get('_special').get('_value')('', Some(new_panacea.get())).get()[1]

            # Make a new leaf from the new value
            new_panacea = new_panacea.map(lambda x: result)

        return new_key, new_panacea

    def update(self, panacea, bc: str = '') -> Option[(str, PanaceaBase)]:
        """Method to (filter and) update the `panacea` instance.
         It updates the locations that met the filtering criteria and leave the rest untouched.

        Parameters
        ----------
        panacea : PanaceaBase
            The PanaceaBase instance to (filter and) update
        bc : str, optional
            The breadcrumb so far

        Returns
        -------
        The result of updating in form of Option value of type Option[(str, PanaceaBase)] where (str, PanaceaBase) is
        the key/value pair of the result.

        """

        # For each of the elements in the propagation, we should do the updating again with updated bc
        for_each_element = \
            lambda key, value: self.update(panacea=value, bc=f'{bc}.{key}')

        # For propagation
        # Remember that propagation happens when the instance panacea did not meet the filtering criteria
        # If the given `panacea` is a branch node, propagate again
        # If the given `panacea` is a leaf node, return itself, as there is no more children to propagate and
        # the `panacea` instance did not meet the filtering criteria but we want to leave it untouched
        if issubclass(type(panacea), Panacea):
            propagate = self.propagate_all(for_each_element)
        elif issubclass(type(panacea), PanaceaLeaf):
            propagate = lambda key, x: Some((key, x))
        else:
            raise AttributeError(
                f"What just happened?! The tree has to have only nodes or leaves, got type {type(panacea)}"
            )

        # If the `panacea` instance met the filtering criteria
        # After that, do the updating and return the result in a correct way
        # If we should go even deeper than the shallowest level matched, go deeper!
        if self.condition_dict.get(self.Update.Conditionals.DEEP.value) is True:
            do_after_satisfied = lambda key, panacea: propagate(*self.update_self(key, panacea))
        else:
            do_after_satisfied = lambda key, panacea: Some(self.update_self(key, panacea))

        # Do the traversing with the correct functions
        return self.traverse(panacea=panacea, bc=bc, do_after_satisfied=do_after_satisfied, propagate=propagate)


# constants
FILTER_OPERATIONS = Modification.Filter.Operations
UPDATE_OPERATIONS = Modification.Update.Operations
UPDATE_CONDITIONALS = Modification.Update.Conditionals
