from __future__ import annotations
from typing import Dict, Any, List, Type, Union, Callable
from types import FunctionType
import yaml
import re
from itertools import chain
from .option import Some, Nothing, Option


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
        self._parameters = self._init_helper(config_dict)

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

    def _init_helper(self, config_dict: Dict) -> Dict:
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

    def get(self, item: str) -> Type[Option]:
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
            return Nothing()

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

        return self.get(item).get_or_else(default_value)

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

    def update(self, name: str, value):
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
            parameters = {**self.__dict__, **{name: value}}  # Either update or create new attribute
            return self.__class__(parameters)
        # In case we have to go deeper to change an attribute
        else:
            root: str = split[0]
            parameters = {key: value for key, value in self.__dict__.items() if key != root}
            # Find if we should update or create new attribute
            chooser = self.__dict__[root] if root in self.__dict__.keys() else self.__class__({})
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

    def find_one(self, filter_dict: Dict = None):

        if filter_dict is None:
            return self
        else:
            found = self._find_one_helper(filter_dict, '', '')
            return found or self.__class__()

    def _find_one_helper(self, filter_dict, bc, bc_meta):

        bc_meta = bc_meta + self.__dict__.get('_meta', '')

        if self._filter_checker(filter_dict, bc, bc_meta) is True:
            return self

        else:
            for key, value in self.__dict__.items():
                if type(value) is type(self):
                    out = value._find_one_helper(filter_dict, bc + f'.{key}', bc_meta)
                    if out is not None:
                        return out
                elif type(value) is list:
                    for item in value:
                        if type(item) is type(self):
                            out = item._find_one_helper(filter_dict, bc + f'.{key}', bc_meta)
                            if out is not None:
                                return out
                else:
                    out = self._filter_checker(filter_dict, bc + f'.{key}', bc_meta) and value
                    if out is not False:
                        new_dict = {**{'_meta': self.__dict__['_meta']}, **{key: out}} \
                            if self.__dict__.get('_meta') \
                            else {key: out}
                        return self.__class__(new_dict)

        return None

    def _filter_checker(self, filter_dict: Dict, bc: str = '', bc_meta: str = '', this=None):

        # Check if all filters are satisfied
        satisfied = all(
            filter.filter(self.get(key))
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
            filter_dict.get('_self').filter(Some(this or self)) if filter_dict.get('_self') is not None else True


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

    def filter(self, filter_dict: Dict = None):
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
        filtered: Type[Option] = self._filter_helper(processed_filter_dict, '', '')

        # Return the result of an empty ConfigParser if the filtering result is empty
        return filtered.get_or_else(self.__class__())

    def _filter_helper(self, filter_dict: Dict, bc: str, bc_meta: str) -> Type[Option]:
        """Method to help with filtering.
            Performs filtering on each of the parameters of the current instance.

        Parameters
        ----------
        filter_dict : dict
            Dictionary containing the filtering criteria whose values are instances of the Filter class
        bc : str

        """

        def helper(name: str, value: Any) -> Type[Option]:
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
                out = value._filter_helper(filter_dict, bc + f'.{name}', bc_meta)
            # If the parameter is anything else, see if it matches the filter and return the result
            else:
                out = Some(value) if self._filter_checker(filter_dict, bc + f'.{name}', bc_meta, value) else Nothing()

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
                    if not value.is_empty()  # filter out the Nothing ones
                }

        return Some(self.__class__(new_dict)) if new_dict else Nothing()

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

        def _parser(self, query: Dict) -> Any:
            """Method to parse the query given and turn it into actions or a list of functions to be called.

            Parameters
            ----------
            query : Any
                A query to be parsed

            Returns
            -------
            A list of functions to be called

            """

            def helper(single_operator: str, value: Any) -> FunctionType:
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

        def filter(self, x: Type[Option]) -> bool:
            """Method to perform the filtering on an Option value.

            This filtering is based on the query given in the constructor.

            Parameters
            ----------
            x : Type[Option]
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

        def _function(self, func: FunctionType) -> Callable[[Type[Option]], bool]:
            """Wrapper function for a function to query on an Option value.

            Parameters
            ----------
            func : FunctionType
                A function to apply on the Option value. Must return boolean

            Returns
            -------
            A function that can be called on an Option value

            """

            def helper(x: Type[Option]) -> bool:
                """Function to be called on an Option value to apply an internal function.

                Parameters
                ----------
                x : Type[Option]
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

        def _exist(self, value: bool) -> Callable[[Type[Option]], bool]:
            """Operator for checking if a variable exists.

            Parameters
            ----------
            value : bool
                A boolean indicating whether we are dealing with existence or non-existence of the element

            Returns
            -------
            A function for filtering

            """

            def helper(x: Type[Option]) -> bool:
                """Helper function to decide whether or not Option x has an element.

                Parameters
                ----------
                x : Type[Option]
                    An Option value to check if exists or not

                Returns
                -------
                A boolean indicating whether or not the Option x has an element

                """

                # The result is the xnor of value and if the Option x is empty
                return \
                    not \
                    ((not x.is_empty()) ^ value)

            return helper

        def _regex(self, regex: str) -> Callable[[Type[Option]], bool]:
            """Operator for checking a regex string on a value.

               Parameters
               ----------
               regex : str
                   A regex string to be checked on the Option value

               Returns
               -------
               A function for filtering

               """

            def helper(x: Type[Option]) -> bool:
                """Helper function to decide whether or not a regex satisfies the Option x element.

                Parameters
                ----------
                x : Type[Option]
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
