from __future__ import annotations
from typing import Dict, Any, List, Type, Union, Callable
from types import FunctionType
import yaml
import re
from itertools import chain
from .option import Some, nothing, Option
from abc import ABC, abstractmethod


class PanaceaBase(ABC):
    """A class that will act as a base node class for the config tree classes."""

    def __init__(self):
        """Initializes the class."""

        # Create a dictionary to hold the data
        self._parameters: Dict[str, Any] = {}

        pass

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
    def get_option(self, item: str) -> Option:
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
            Item to look for in the shallowest level

        Returns
        -------
        Value of the item

        """

        # Get the item from the returned option
        out = self.get_option(item)

        # Raise an error if the item does not exist
        if out.is_empty():
            raise AttributeError(f'{item} does not exist in this instance of {self.__class__.__name__}!')

        return out.get()

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

        return self.get_option(item).get_or_else(default_value)

    # Debugging mode

    def enable_debug_mode(self) -> PanaceaBase:
        """Enables debug mode, in this instance and all children, where parameters can be accessed with . notation."""

        # Reserve the current __dict__
        self.__old___dict__ = self.__dict__

        # Update the dictionary and pass the method down to children
        self.__dict__ = {**self.__dict__, **self._parameters}
        for key, item in self._parameters.items():
            if issubclass(type(item), PanaceaBase):
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

    # Modification

    class Filter:
        """A class that parses, holds and checks the filtering queries for Panacea."""

        def __init__(self, query: Dict):
            """Initializer to the class which will parse the query and create the corresponding actions.

            Parameters
            ----------
            query : Dict
                The query to do the filtering to be parsed
                Right now, only these sub-queries/operators are supported:
                    lambda functions with operator $function, whether it exists with operator $exist
                The query has to be a dictionary with key containing the operator

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
                if single_operator == '$function':
                    return self._function(value)
                # Do the rest of the operations
                elif single_operator == '$exist':
                    return self._exist(value)
                elif single_operator == '$regex':
                    return self._regex(value)
                elif single_operator == '$equal':
                    return self._equal(value)

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

        def _function(self, func: FunctionType) -> Callable[[Option], bool]:
            """Wrapper function for a function to query on an Option value.

            Parameters
            ----------
            func : FunctionType
                A function to apply on the Option value. Must return boolean

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

                return \
                    x\
                    .filter(func)\
                    .is_defined()

            return helper

        def _exist(self, value: bool) -> Callable[[Option], bool]:
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
                    x\
                    .filter(lambda d: isinstance(d, str))\
                    .filter(lambda a: re.search(regex, a) is not None)\
                    .is_defined()

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

                return x.exist(lambda item: item == value)

            return helper


class Panacea(PanaceaBase):
    """A class that will contain config values and subclasses.

    This class is the non-leaf node class in the tree of configurations given.
    The leaf nodes will be of type PanaceaLeaf.
    """

    # Set static variables
    item_begin_symbol = f'\u2022'
    item_color = f'\033[38;5;209m'

    vertical_bar_symbol = f'\u22EE'
    # vertical_bar_color = f'{CCC.foreground.set_8_16.light_gray}'
    vertical_bar_color = f'\033[37m'
    vertical_bar_with_color = f'{vertical_bar_color}{vertical_bar_symbol}\033[0m'

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

        # Create the dictionary of parameters
        final_parameters = {key: helper(value) for key, value in config_dict.items()}

        return final_parameters

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
        out_string += f':' if depth != 1 and name != '' else ''  # Only add ':' if we want to print anything in front
        out_string += f'\n'

        # Create the string from all the children
        out_substring = \
            ''.join(
                item.str_representation(name=key, depth=depth-1)
                for key, item
                in sorted(self._parameters.items())
            )

        # Indent the children result and add to the result
        out_string += re.sub(
            r'(^|\n)(?!$)',
            r'\1' + f'{self.vertical_bar_with_color}' + r'\t',
            out_substring
        )

        return out_string

    def _identity_str_representation(self, name) -> str:

        return f'{self.item_begin_symbol} {self.item_color}{name}\033[0m'

    # Getters

    def get_option(self, item: str) -> Option:
        """Gets an item in the instance and return an Option value of that.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level

        Returns
        -------
        An Option value possibly containing the item

        """

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

    def get(self, item: str) -> Any:
        """Gets an item in the instance and return it or raise an error if not exists.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level

        Returns
        -------
        Value of the item

        """

        # Get the item from the returned option
        out = super().get(item)

        # If the result is a leaf, return its value
        if isinstance(out, self.Leaf):
            return out.get('_value')
        # If the result is a non-leaf, return itself
        else:
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

        # Check if non of the parameters are leaf of subtype PanaceaLeaf, i.e. the instance is flat
        check = all([
            not issubclass(type(self._parameters.get(parameter_name)), PanaceaLeaf)
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
        result = self._parameters.get(parameter_names[0])

        # Reduce the function `function` over all elements
        for parameter_name in parameter_names[1:]:
            result = function(result, self._parameters.get(parameter_name))

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
            result = function(result, self._parameters.get(parameter_name))

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

        def intersection_helper(new: Any, old: Any, this) -> Option:
            """Helper function to create the intersection of two elements.

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
            The result of intersection of the new and old elements as an Option value

            """

            # If we are dealing with the same object, return it
            if new is old:
                return Some(new)
            # If the new and old items are ConfigParsers, traverse recursively, else return the new item
            if type(new) == type(old) == type(this):
                intersection = old.intersection(new)
                return Some(intersection) if not intersection.is_empty() else nothing
            else:
                return nothing

        # If None is passed, return empty Panacea
        if new_config is None:
            return self.__class__()

        assert type(new_config) is type(self), \
            f"The element to be intersected must be the same type as {self.__class__.__name__} class."

        # Find the elements that exist in both of the configs
        out_intersection = {
            key: value.get()
            for key, value
            in {
                key: intersection_helper(value_new, self._parameters.get(key), self)
                for key, value_new in new_config._parameters.items()
                if (key in self._parameters) and (key in new_config._parameters)
            }.items()
            if value.is_defined()
        }

        # If the intersection is the current instance, return self
        if out_intersection == self._parameters:
            return self
        # If the intersection is the to-be-intersected instance, return it
        elif out_intersection == new_config._parameters:
            return new_config
        # Concatenate the newly generated dictionaries to get a new one and create a Panacea from that
        else:
            return self.__class__(out_intersection)

    def _filter_checker(self, filter_dict: Dict, bc: str = '', bc_meta: str = '', this: Option = nothing) -> bool:

        # Check if all filters are satisfied
        satisfied = all(
            filter.filter(self.get_option(key))
            for key, filter
            in filter_dict.get('field').items()  # items that should select specific fields
        )

        # Check special items that start with _
        satisfied &= \
            filter_dict\
                .get('_special')\
                .get('_bc')\
                .filter(Some(bc)) \
                if filter_dict.get('_special').get('_bc') is not None \
                else True
        satisfied &= \
            filter_dict\
                .get('_special')\
                .get('_bc_meta')\
                .filter(Some(bc_meta)) \
                if filter_dict.get('_special').get('_bc_meta') is not None \
                else True
        satisfied &= \
            filter_dict\
                .get('_special')\
                .get('_self')\
                .filter(this.or_else(Some(self))) \
                if filter_dict.get('_special').get('_self') is not None \
                else True

        return satisfied

    def _make_filter_dictionary(self, filter_dict: Dict) -> Dict:
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
        filter_dict = {
            # Elements that are already in dictionary form that Filter class accept
            **{key: value for key, value in filter_dict.items() if isinstance(value, dict)},
            # Elements that are not in dictionary form are set to '$equal' operator for the Filter class
            **{key: {'$equal': value} for key, value in filter_dict.items() if not isinstance(value, dict)}
        }

        # Process the criteria for each of the filter_dict fields into an instance of the Filter class
        processed_filter_dict: Dict = {key: self.Filter(value) for key, value in filter_dict.items()}

        # Split the dictionary into two keys: `field` and `_special`
        # The `field` key contains all the selectors corresponding to the name of the fields
        # The `_special` key contains all the special selectors that start with '_'
        processed_filter_dict = \
            {
                'field': {key: value for key, value in processed_filter_dict.items() if not key.startswith('_')},
                '_special': {key: value for key, value in processed_filter_dict.items() if key.startswith('_')}
            }

        return processed_filter_dict

    def filter(self, filter_dict: Dict = None) -> Panacea:
        """Method to filter the current item based on the criteria given and return a new Panacea that satisfies
            the criteria.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                The keys of the dictionary are among these options:
                    - the name of field: this will cause the filtering to be based on this specific field
                    - special names that start with '_': these will cause the filtering to be based on special things
                                                            specified by this key.
                                                            A complete list of them are:
                                                                - _bc: filter on the breadcrumb of the name of the
                                                                        fields
                                                                - _bc_meta: filter on the breadcrumb of the name of the
                                                                        fields
                                                                - _self: filter on the current item itself
                The value must be a dictionary with its keys being the operators of the Filter class and its values
                    be the corresponding expression.

        Returns
        -------
        A new instance of the Panacea class whose elements satisfy the given criteria.

        """

        # Return itself if no filtering is provided
        if filter_dict is None:
            return self

        # Make the filter dictionary whose elements are instances of the Filter class
        processed_filter_dict = self._make_filter_dictionary(filter_dict)

        # Perform the filtering
        filtered: Option = self._filter_helper(processed_filter_dict, '', '')

        # Return the result of an empty Panacea if the filtering result is empty
        return filtered.get_or_else(self.__class__())

    def _filter_helper(self, filter_dict: Dict, bc: str, bc_meta: str) -> Option:
        """Method to help with filtering.
            Performs filtering on each of the parameters of the current instance.

        Parameters
        ----------
        filter_dict : dict[str, dict[str, Filter]]
            Dictionary containing the filtering criteria whose values are instances of the Filter class
            It consists of two keys:
                'field,' whose element select existing fields, and
                '_special,' whose element select special generated items
        bc : str
            The breadcrumb string so far
        bc_meta : str
            The meta breadcrumb string so far

        Returns
        -------
        An Option value containing the results

        """

        def helper(name: str, value: Any) -> Option:
            """Helper method to filter a given parameter.

            Parameters
            ----------
            name : str
                The name of the parameter
            value : Any
                The value of the parameter

            Returns
            -------
            An Option instance containing the result of filtering based on the closure-ized filter_dict

            """

            # Set a placeholder for the result
            out = nothing

            # If the parameter is Panacea, call its own filtering with the updated name
            if type(value) is type(self):
                out: Option = value._filter_helper(filter_dict, bc + f'.{name}', bc_meta)

            # If the parameter is a leaf, see if it matches the filter and return the result
            # It should be noted that there must exist only one field selector in field_dict.field in order
            # to be able to match a specific leaf, otherwise it will definitely not be a match
            # moreover, that field should be the same as the name of the leaf to be examined
            # the other way to a possible match is to not have any field and have only _special selectors
            elif list(filter_dict.get('field').keys()) in [[], [name]]:  # Empty or the name, respectively
                # Create a new filter dictionary specific to this leaf
                # Check if we should filter the internal value of the leaf or not
                # If yes, populate the element `field` dictionary by '_value' and its corresponding filter else empty
                field_dict = \
                    {'_value': filter_dict.get('field').get(name)} \
                        if filter_dict.get('field').get(name) is not None \
                        else {}
                modified_filter_dict = \
                    {
                        'field': field_dict,
                        '_special': filter_dict.get('_special'),
                    }
                out: Option = value._filter_helper(modified_filter_dict, bc + f'.{name}', bc_meta)

            return out

        # Construct the meta breadcrumb
        bc_meta: str = bc_meta + f'.' + self.get_option('_meta').fold(lambda x: x.get('_value'), '')

        # Check if the current instance satisfies the filtering
        if self._filter_checker(filter_dict, bc, bc_meta) is True:
            return Some(self)
        else:
            new_dict = \
                {  # Keep only the filtered parameters that are not empty, i.e. that are not Nothing
                    key: value.get()
                    for key, value
                    in
                        {  # Process and filter each of the self._parameters
                            key: helper(key, value)
                            for key, value
                            in self._parameters.items()
                        }.items()
                    if value.is_defined()  # filter out the Nothing ones
                }

        # If nothing has changed, return self
        if new_dict == self._parameters:
            return Some(self)
        # if something has changed, create a new Panacea
        elif new_dict:
            return Some(self.__class__(new_dict))
        # If filtering resulted in nothing
        else:
            return nothing

    def find_one(self, filter_dict: Dict = None) -> Option:
        """Method to find the first item based on the criteria given and return an Option value of it.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to `filter` method for more information

        Returns
        -------
        An Option value of the first element found that satisfies the given criteria.

        """

        # Return None if no filtering is provided
        if filter_dict is None:
            return nothing

        # Process the criteria for each of the filter_dict fields into an instance of the Filter class
        processed_filter_dict: Dict = self._make_filter_dictionary(filter_dict)

        # Perform the finding
        found_one: Option = self._find_one_helper(processed_filter_dict, '', '')

        # Return
        return found_one

    def _find_one_helper(self, filter_dict: Dict, bc: str, bc_meta: str) -> Option:
        """Method to help with finding the first element that satisfies criteria.
            Performs filtering on each of the parameters of the current instance and returns the first that satisfies
            the criteria.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria whose values are instances of the Filter class
        bc : str
            The breadcrumb string so far
        bc_meta : str
            The meta breadcrumb string so far

        Returns
        -------
        An Option value containing the results

        """

        def helper(name: str, value: Any) -> Option:
            """Helper method to filter a given parameter.

            Parameters
            ----------
            name : str
                The name of the parameter
            value : Any
                The value of the parameter

            Returns
            -------
            An Option instance containing the result of filtering based on the closure-ized filter_dict

            """

            # Set a placeholder for the result
            out = nothing

            # If the parameter is Panacea, call its own filtering with the updated name
            if type(value) is type(self):
                out: Option = value._find_one_helper(filter_dict, bc + f'.{name}', bc_meta)

            # If the parameter is a leaf, see if it matches the filter and return the result
            # It should be noted that there must exist only one field selector in field_dict.field in order
            # to be able to match a specific leaf, otherwise it will definitely not be a match
            # moreover, that field should be the same as the name of the leaf to be examined
            # the other way to a possible match is to not have any field and have only _special selectors
            elif list(filter_dict.get('field').keys()) in [[], [name]]:  # Empty or the name, respectively
                # Create a new filter dictionary specific to this leaf
                # Check if we should filter the internal value of the leaf or not
                # If yes, populate the element `field` dictionary by '_value' and its corresponding filter else empty
                field_dict = \
                    {'_value': filter_dict.get('field').get(name)} \
                        if filter_dict.get('field').get(name) is not None \
                        else {}
                modified_filter_dict = \
                    {
                        'field': field_dict,
                        '_special': filter_dict.get('_special'),
                    }
                out: Option = value._find_one_helper(modified_filter_dict, bc + f'.{name}', bc_meta)

            return out

        # Construct the meta breadcrumb
        bc_meta: str = bc_meta + f'.' + self.get_or_else('_meta', '')

        # Construct placeholder for the final result
        result: Option = nothing

        # Check if the current instance satisfies the filtering
        if self._filter_checker(filter_dict, bc, bc_meta) is True:
            result = Some(self)
        else:
            # Go over each of the parameters and return the one that satisfies the criteria
            for key, value in self._parameters.items():

                # Get the results of finding in the current parameter
                result: Option = helper(key, value)

                # If a result is found, break and do not go over other parameters
                if result.is_defined():
                    break

        return result

    def update(self, name: str, value) -> Panacea:
        """Update an entry in the config and return a new Panacea.

        Parameters
        ----------
        name : str
            The name of the attribute to change
                For example, this can be 'some_param' or can be 'parent.parent.some_child_param'
        value : Any
            The value that must be set for the updated attribute

        Returns
        -------
        An instance of Panacea class with the updated attribute

        """

        # Split the name to see if should go deeper in the attributes
        split: List[str] = name.split('.')
        # list_entry = re.search(r'(\w+)\[(\d+)\]', split[0])

        # In case we have to change an attribute here at depth zero
        if len(split) == 1:
            parameters = {**self._parameters, **{name: value}}  # Either update or create new attribute
            return self.__class__(parameters)
        # In case we have to go deeper to change an attribute
        else:
            root: str = split[0]
            parameters = {key: value for key, value in self._parameters.items() if key != root}
            # Find if we should update or create new attribute
            chooser = self._parameters[root] if root in self._parameters.keys() else self.__class__({})
            parameters = {**parameters, root: chooser.update('.'.join(split[1:]), value)}
            return self.__class__(parameters)


class PanaceaLeaf(PanaceaBase):
    """A class that will contain a value in a config tree.

    This class is the leaf class in the tree of configurations given.
    The non-leaf nodes will be of type Panacea.
    """

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

        super().__init__()

        # To conform with the dictionary initialization, read also from a dictionary whose value is stored in _value key
        if isinstance(value, dict) and '_value' in value.keys():
            self._value = value.get('_value')
        else:
            # Hold the value
            self._value = value

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

        result = {'_value': self._value}

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

        out_string += f': '

        if isinstance(self._value, list):
            out_string += self._list_str_representation()
        else:
            out_string += self._item_str_representation()

        return out_string

    def _identity_str_representation(self, name) -> str:

        return f'{self.item_begin_symbol}{self.item_color} {name}\033[0m'

    def _item_str_representation(self) -> str:

        return f'{self._value}\n'

    def _list_str_representation(self) -> str:

        out_string = ''.join(f'\t{self.begin_list_symbol} {value}\n' for value in self._value)

        out_string = f'\n{out_string}'

        return out_string

    # Getters

    def get_option(self, item: str) -> Option:
        """Gets an item in the instance and return an Option value of that.

        Parameters
        ----------
        item : str
            Item to look for in the shallowest level

        Returns
        -------
        An Option value possibly containing the item

        """

        if item == '_value':
            return Some(self._value)
        else:
            return nothing

    # Debugging mode

    def enable_debug_mode(self) -> PanaceaLeaf:
        """Enables debug mode, in this instance and all children, where parameters can be accessed with . notation."""

        # Do the general enable_debug_mode
        super().enable_debug_mode()

        # Update the dictionary and pass the method down to children
        self.__dict__ = {**self.__dict__, '_value': self._value}

        return self

    # def filter(self, filter) -> bool:
    #     """Method to filter the current leaf based on the Filter instance, `filter`, and return the boolean result.
    #
    #     Parameters
    #     ----------
    #     filter : Filter
    #         Filter instance to be used for filtering the current leaf node
    #
    #     Returns
    #     -------
    #     A new instance of the Panacea class whose elements satisfy the given criteria.
    #
    #     """
    #
    #     # Find the result of the filtering, i.e. whether this instance satisfies the filter
    #     satisfied = filter.filter(self.get_option('_value'))
    #
    #     return satisfied

    # Modifications

    def _filter_helper(self, filter_dict, bc: str, bc_meta: str) -> Option:
        """Method to help with filtering.
            Performs filtering on internal value of the current instance.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria whose values are instances of the Filter class
        bc : str
            The breadcrumb string so far
        bc_meta : str
            The meta breadcrumb string so far

        Returns
        -------
        An Option value containing the results

        """

        # A placeholder for the result
        result = nothing

        # Check if the current instance satisfies the filtering
        if self._filter_checker(filter_dict, bc, bc_meta) is True:
            result = Some(self)

        return result

    def _filter_checker(self, filter_dict: Dict, bc: str = '', bc_meta: str = '', this: Option = nothing) -> bool:

        # Check if all filters are satisfied
        satisfied = all(
            filter.filter(self.get_option(key))
            for key, filter
            in filter_dict.get('field').items()  # items that should select specific fields
        )

        # Check special items that start with _
        satisfied &= \
            filter_dict\
                .get('_special')\
                .get('_bc')\
                .filter(Some(bc)) \
                if filter_dict.get('_special').get('_bc') is not None \
                else True
        satisfied &= \
            filter_dict\
                .get('_special')\
                .get('_bc_meta')\
                .filter(Some(bc_meta)) \
                if filter_dict.get('_special').get('_bc_meta') is not None \
                else True
        satisfied &= \
            filter_dict\
                .get('_special')\
                .get('_self')\
                .filter(this.or_else(Some(self))) \
                if filter_dict.get('_special').get('_self') is not None \
                else True

        return satisfied

    def _find_one_helper(self, filter_dict: Dict, bc: str, bc_meta: str) -> Option:
        """Method to help with finding the first element that satisfies criteria.
            Performs filtering on internal value of the current instance.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria whose values are instances of the Filter class
        bc : str
            The breadcrumb string so far
        bc_meta : str
            The meta breadcrumb string so far

        Returns
        -------
        An Option value containing the results

        """

        # A placeholder for the result
        result = nothing

        # Check if the current instance satisfies the filtering
        if self._filter_checker(filter_dict, bc, bc_meta) is True:
            result = Some(self._value)

        return result
