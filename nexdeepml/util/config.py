from __future__ import annotations
from typing import Dict, Any, List, Type, Union, Callable
from types import FunctionType
import yaml
import re
from itertools import chain
from .option import Some, nothing, Option


class ConfigParser:
    """A class that will contain config values and subclasses."""

    # Set static variables
    item_begin_symbol = f'\u2022'
    item_color = f'\033[38;5;209m'
    begin_list_symbol = f'-'
    # begin_list_color = f'{CCC.foreground.set_88_256.chartreuse4}'
    begin_list_color = f'\033[38;5;70m'
    begin_list_symbol = f'{begin_list_color}{begin_list_symbol}\033[0m'
    vertical_bar_symbol = f'\u22EE'
    # vertical_bar_color = f'{CCC.foreground.set_8_16.light_gray}'
    vertical_bar_color = f'\033[37m'
    vertical_bar_with_color = f'{vertical_bar_color}{vertical_bar_symbol}\033[0m'

    def __init__(self, config_dict: Dict = None):
        """Initializes the class based on the input config dictionary.

        Parameters
        ----------
        config_dict : Dict
            A dictionary containing all the configurations
        """

        # Create a dictionary to hold the data
        self._parameters = {}

        # Check if no input was given
        if config_dict is None:
            return

        # Take care of the special _meta field
        if '_meta' in config_dict.keys():
            self._parameters['_meta'] = config_dict.pop('_meta')

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

        def helper(value: Any) -> Any:
            """Helper function to decide what to do with the incoming value.

            Parameters
            ----------
            value : Any
                The value to decide what to do about

            Returns
            -------
            Either the value unchanged or a ConfigParser instance if the value is a dictionary

            """

            # Check the type of the input
            # If the type is a dictionary, go deep into it
            # Otherwise, just leave the item be there
            if type(value) == dict:
                return self.__class__(value)
            else:
                out = value

            return out

        # Create the dictionary of parameters
        final_parameters = {key: helper(value) for key, value in config_dict.items()}

        return final_parameters

    def __str__(self) -> str:
        """Method to help with the visualization of the configuration in YAML style."""

        # Get dictionary representation of the current instance
        out_string = self.str_representation(depth=2)

        return out_string

    def print(self, depth: int = -1) -> None:
        """Print the configuration up to certain depth.

        Parameters
        ----------
        depth : int, optional
            The depth until which the string representation should go down the configuration

        """

        print(self.str_representation(depth=depth))

    def dict_representation(self, config: Any = None) -> Dict:
        """Method to help with visualization of the configurations.

        Parameters
        ----------
        config : Any
            Configuration to be processed

        Returns
        -------
        dict
            Configurations in a dictionary such as the one with json/yaml
        """

        # If no config is given, perform everything on the current instance
        if config is None:
            config = self

        # Check for the type of the input and act accordingly
        if type(config) is type(self):
            out = {key: self.dict_representation(item) for key, item in config._parameters.items()}
        else:
            out = config

        return out

    def str_representation(self, config: Any = None, depth: int = -1) -> str:
        """Helper function to create a string representation of the instance.

        Parameters
        ----------
        config : {ConfigParser, list, str}, optional
            Configuration reference to create string from
        depth : int, optional
            The depth until which the string representation should go down the configuration

        Returns
        -------
        String containing the representation of the configuration given

        """

        def config_dict_string(key, item):
            """Helper function to create the string from elements of a dictionary.

            Parameters
            ----------
            key
                Key of the dictionary element
            item
                Value of the dictionary element

            Returns
            -------
            String representation of the item

            """

            out_string = ''
            out_string += f'{self.item_begin_symbol} {self.item_color}{key}\033[0m'
            out_string += f':' if depth != 1 else ''  # Only add ':' if we want to print anything in front
            out_string += f'\n'

            # Find the corresponding string for the item
            out_substring = self.str_representation(item, depth - 1)

            # Indent the result
            out_substring = re.sub(
                r'(^|\n)(?!$)',
                r'\1' + f'{self.vertical_bar_with_color}' + r'\t',
                out_substring
            )

            out_string += out_substring

            return out_string

        def config_list_string(item):
            """Helper function to create the string from elements of a list.

            Parameters
            ----------
            item
                Value of the list element

            Returns
            -------
            String representation of the item

            """

            out_string = self.str_representation(item, depth)

            # Write begin_list at the beginning in green
            out_string = \
                f'{self.begin_list_symbol} {out_string}'\
                if type(item) != type(self) \
                else re.sub(r'(^|\n)' + r'(' + f'{self.item_begin_symbol}' + r')',
                            r'\1' + f'{self.begin_list_color}' + r'\2' + f'\033[0m',
                            out_string)

            return out_string

        # Check if we have reach the root of the recursion, i.e. depth is zero
        if depth == 0:
            return ''

        # If no config is given, perform everything on the current instance
        if config is None:
            config = self

        if type(config) is type(self):
            out_string = ''.join(config_dict_string(key, item) for key, item in sorted(config._parameters.items()))
        elif type(config) == list:
            out_string = ''.join(config_list_string(item) for item in config)
        else:
            return self._identity_str_representation(config) + f'\n'

        # The final result
        return out_string

    def _identity_str_representation(self, value):

        return str(value)

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

    def is_empty(self) -> bool:
        """EXPERIMENTAL: A checker method that tells whether this instance of config is empty or not

        Returns
        -------
        bool
            True if this class is empty and False otherwise
        """

        return not bool(self._parameters)

    def enable_debug_mode(self) -> ConfigParser:
        """Enables debug mode, in this instance and all children, where parameters can be accessed with . notation."""

        # Reserve the current __dict__
        self.__old___dict__ = self.__dict__

        # Update the dictionary and pass the method down to children
        self.__dict__ = {**self.__dict__, **self._parameters}
        for key, item in self._parameters.items():
            if issubclass(type(item), ConfigParser):
                item.enable_debug_mode()

        return self

    def disable_debug_mode(self) -> ConfigParser:
        """Disables debug mode, in this instance and all children, where parameters can be accessed with . notation."""

        # Check if enable_debug_mode has already been activated
        if getattr(self, '__old___dict__') is None:
            return self

        # Set the __dict__ to previous version and pass down
        self.__dict__ = self.__old___dict__
        for key, item in self._parameters.items():
            if issubclass(type(item), ConfigParser):
                item.disable_debug_mode()

        # Delete the old dictionary
        del self.__old___dict__

        return self

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

        # Check if non of the parameters are of type ConfigParse, i.e. the instance is flat
        check = all([
            not issubclass(type(self._parameters.get(parameter_name)), ConfigParser)
            for parameter_name
            in parameter_names
            if not parameter_name.startswith('_')
        ])

        # Assert the flat-ness of the instance
        assert check is True, f'Object of type {self.__class__.__name__} is not flat, cannot perform reduction!'

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

        # Check if non of the parameters are of type ConfigParse, i.e. the instance is flat
        check = all([
            not issubclass(type(self._parameters.get(parameter_name)), ConfigParser)
            for parameter_name
            in parameter_names
            if not parameter_name.startswith('_')
        ])

        # Assert the flat-ness of the instance
        assert check is True, f'Object of type {self.__class__.__name__} is not flat, cannot perform fold!'

        # Fill the result with the first element
        result = zero_element

        # Reduce the function `function` over all elements
        for parameter_name in parameter_names:
            result = function(result, self._parameters.get(parameter_name))

        return result

    def update(self, name: str, value) -> ConfigParser:
        """Update an entry in the config and return a new ConfigParser.

        Parameters
        ----------
        name : str
            The name of the attribute to change
                For example, this can be 'some_param' or can be 'parent.parent.some_child_param'
        value : Any
            The value that must be set for the updated attribute

        Returns
        -------
        An instance of ConfigParser class with the updated attribute

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

    def union(self, new_config: ConfigParser) -> ConfigParser:
        """method to take the union of two config instances and replace the currently existing ones.

        Parameters
        ----------
        new_config : ConfigParser
            A config instance to union with (and update) the current one

        Returns
        -------
        A new instance of ConfigParser containing the union of the two config instances

        """

        def union_helper(new: Any, old: Any, this) -> Any:
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
            if (key in self._parameters) ^ (key in new_config._parameters)
        }

        # Find the elements that exist in both of the configs
        out_intersection = {
            key: union_helper(value, self._parameters.get(key), self)
            for key, value in new_config._parameters.items()
            if (key in self._parameters) and (key in new_config._parameters)
        }

        # Concatenate the newly generated dictionaries to get a new one and create a ConfigParser from that
        new_config = self.__class__({**out_delta, **out_intersection})

        return new_config

    def _filter_checker(self, filter_dict: Dict, bc: str = '', bc_meta: str = '', this: Option = nothing) -> bool:

        # Check if all filters are satisfied
        satisfied = all(
            filter.filter(self.get_option(key))
            for key, filter
            in filter_dict.items()
            if not key.startswith('_')  # special items start with '_'
        )

        # Check special items that start with _
        satisfied &= \
            filter_dict.get('_bc').filter(Some(bc)) if filter_dict.get('_bc') is not None else True
        satisfied &= \
            filter_dict.get('_bc_meta').filter(Some(bc_meta)) if filter_dict.get('_bc_meta') is not None else True
        satisfied &= \
            filter_dict.get('_self').filter(this.or_else(Some(self))) if filter_dict.get('_self') is not None else True


        # bc_regex = filter_dict.get('_bc_regex')
        # if bc_regex is not None:
        #     satisfied &= bool(re.search(bc_regex, bc))
        # if satisfied is False:
        #     return False
        #
        # bc_meta_regex = filter_dict.get('_bc_meta_regex')
        # if bc_meta_regex is not None:
        #     satisfied &= bool(re.search(bc_meta_regex, bc_meta))
        # if satisfied is False:
        #     return False
        #
        # for key, function in filter_dict.items():
        #     if key not in ['_bc_regex', '_bc_meta_regex']:
        #         satisfied &= function(self.__dict__.get(key))
        #     if satisfied is False:
        #         return False

        return satisfied

    def filter(self, filter_dict: Dict = None) -> ConfigParser:
        """Method to filter the current item based on the criteria given and return a new ConfigParser that satisfies
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
        A new instance of the ConfigParser class whose elements satisfy the given criteria.

        """

        # Return itself if no filtering is provided
        if filter_dict is None:
            return self

        # Process the criteria for each of the filter_dict fields into an instance of the Filter class
        processed_filter_dict: Dict = {key: self.Filter(value) for key, value in filter_dict.items()}

        # return processed_filter_dict

        # Perform the filtering
        filtered: Option = self.__filter_helper(processed_filter_dict, '', '')

        # Return the result of an empty ConfigParser if the filtering result is empty
        return filtered.get_or_else(self.__class__())

    def __filter_helper(self, filter_dict: Dict, bc: str, bc_meta: str) -> Option:
        """Method to help with filtering.
            Performs filtering on each of the parameters of the current instance.

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

            # If the parameter is ConfigParser, call its own filtering with the updated name
            if type(value) is type(self):
                out: Option = value.__filter_helper(filter_dict, bc + f'.{name}', bc_meta)
            # If the parameter is anything else, see if it matches the filter and return the result
            else:
                out: Option = Some(value) \
                    if self._filter_checker(filter_dict, bc + f'.{name}', bc_meta, Some(value)) is True \
                    else nothing

            return out

        # Construct the meta breadcrumb
        bc_meta: str = bc_meta + f'.' + self.get_or_else('_meta', '')

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

        return Some(self.__class__(new_dict)) if new_dict else nothing

    def find_one(self, filter_dict: Dict = None) -> Any:
        """Method to find the first item based on the criteria given and return it.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria.
                Refer to `filter` method for more information

        Returns
        -------
        The first element found that satisfies the given criteria.

        """

        # Return None if no filtering is provided
        if filter_dict is None:
            return None

        # Process the criteria for each of the filter_dict fields into an instance of the Filter class
        processed_filter_dict: Dict = {key: self.Filter(value) for key, value in filter_dict.items()}

        # Perform the finding
        found_one: Option = self.__find_one_helper(processed_filter_dict, '', '')

        # Return None if the no result is found
        return found_one.get_or_else(None)

    def __find_one_helper(self, filter_dict: Dict, bc: str, bc_meta: str) -> Option:
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
            An Option instance containing the result of filtering and finding one element
                based on the closure-ized filter_dict

            """

            # If the parameter is ConfigParser, call its own find_one with the updated name
            if type(value) is type(self):
                out: Option = value.__find_one_helper(filter_dict, bc + f'.{name}', bc_meta)
            # If the parameter is anything else, see if it matches the filter and return the result
            else:
                out: Option = Some(value) \
                    if self._filter_checker(filter_dict, bc + f'.{name}', bc_meta, Some(value)) is True \
                    else nothing

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

    class Filter:
        """A class that parses, holds and checks the filtering queries for ConfigParser."""

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
                    not \
                    x\
                    .filter(func)\
                    .is_empty()

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
                    not \
                    x\
                    .filter(lambda d: isinstance(d, str))\
                    .filter(lambda a: re.search(regex, a) is not None)\
                    .is_empty()

            return helper
