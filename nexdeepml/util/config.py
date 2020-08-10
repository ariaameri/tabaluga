from __future__ import annotations
from typing import Dict, Any, List
import yaml
import re
from itertools import chain


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

        # Check if no input was given
        if config_dict is None:
            return

        _meta = config_dict.get('_meta', '')
        self.__dict__['_meta'] = _meta
        _value = config_dict.get('_value')
        if _value is not None:
            self.__dict__['_value'] = _value

        for key, value in config_dict.items():
            if key not in ['_meta', '_value']:
                self.__dict__[key] = self._init_helper(value)

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

    def _init_helper(self, config: Any, add_value: bool = True) -> Any:
        """Helper method for the constructor to recursively construct the config class.

        Parameter
        ---------
        config : Any
            A sub-config of the main config dictionary.
        add_value : bool, optional
            A boolean indicating whether we should add the _value record

        """

        if type(config) is type(self):
            return config
        elif type(config) == dict:
            return self.__class__(config)
        elif type(config) == list:
            out = [self._init_helper(item, False) for item in config]
        else:
            out = self.__class__({'_value': config}) \
                if (add_value is True and '_value' not in self.__dict__.keys()) \
                else config

        return out

    def __str__(self) -> str:
        """Method to help with the visualization of the configuration in YAML style."""

        # Get dictionary representation of the current instance
        out_string = self.str_representation(depth=2)

        return out_string

    def print(self, depth: int = -1):
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
            out = {self.dict_representation(item) for key, item in config.__dict__.items()}
        elif type(config) == list:
            out = [self.dict_representation(item) for item in config]
        else:
            out = config

        return out

    def str_representation(self, config: Any = None, depth: int = -1):
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
                r'((^|\n)(?!$)|^$)',
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

        # The final string to be returned
        # out_string = ''

        # If no config is given, perform everything on the current instance
        if config is None:
            config = self

        if type(config) is type(self):
            out_string = ''.join(config_dict_string(key, item) for key, item in sorted(config.__dict__.items()))
        elif type(config) == list:
            out_string = ''.join(config_list_string(item) for item in config)
        else:
            return self._identity_str_representation(config) + f'\n'

        # The final result
        return out_string

    def _identity_str_representation(self, value):

        return str(value)

    def __getattr__(self, item):
        """Method to help with getting an attribute.

        Using this method, if the requested attribute does not exist,
        a default value of None will be returned.
        """

        return None

    def __setattr__(self, name, value):
        """Method to help with setting an attribute.

        Using this method, if the requested attribute does not exist,
        sets it and if it exists, raises an error.
        """

        raise Exception(f"Cannot change or add value of {name} due to immutability. Use `update` method instead.")

    def is_empty(self) -> bool:
        """EXPERIMENTAL: A checker method that tells whether this instance of config is empty or not

        Returns
        -------
        bool
            True if this class is empty and False otherwise
        """

        return not bool(self.__dict__)

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

        # TODO: optimize

        def union_helper(new, old, this):

            if type(new) == type(old) == type(this):
                return old.union(new)
            else:
                return new

        if new_config is None:
            return new_config

        if type(new_config) is type(self):

            out_delta = {
                key: value
                for key, value in chain(self.__dict__.items(), new_config.__dict__.items())
                if (key in self.__dict__) ^ (key in new_config.__dict__)
            }

            out_intersection = {
                key: union_helper(value, self.get(key), self)
                for key, value in new_config.__dict__.items()
                if (key in self.__dict__) and (key in new_config.__dict__)
            }

            new_config = self.__class__({**out_delta, **out_intersection})

        return new_config

    def get(self, item: str, default_value: Any = None) -> Any:
        """Gets an item in the instance or return the default_value if not found.

        Parameters
        ----------
        item : str
            Item to be search for possibly containing the hierarchy
        default_value : str, optional
            Value to be returned if the item is not found. If not give, None will be returned

        Returns
        -------
        The value of the found item or the default_value if not found.

        """

        # TODO: Improve

        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return default_value

    def find_one(self, filter_dict: Dict = None):

        if filter_dict is None:
            return self
        else:
            found = self._find_one_helper(filter_dict, '', '')
            return found or self.__class__()

    def _find_one_helper(self, filter_dict, bc, bc_meta):

        def checker():

            satisfied = True

            bc_regex = filter_dict.get('_bc_regex')
            if bc_regex is not None:
                satisfied &= bool(re.search(bc_regex, bc))
            if satisfied is False:
                return False

            bc_meta_regex = filter_dict.get('_bc_meta_regex')
            if bc_meta_regex is not None:
                satisfied &= bool(re.search(bc_meta_regex, bc_meta))
            if satisfied is False:
                return False

            for key, function in filter_dict.items():
                if key not in ['_bc_regex', '_bc_meta_regex']:
                    satisfied &= function(self.__dict__.get(key))
                if satisfied is False:
                    return False

            return satisfied

        bc_meta = bc_meta + f'.{self.__dict__["_meta"]}'

        if checker() is True:
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

        return None

    def filter(self, filter_dict: Dict = None):

        if filter_dict is None:
            return self
        else:
            filtered = self._filter_helper(filter_dict, '', '')
            return filtered or self.__class__()

    def _filter_helper(self, filter_dict, bc, bc_meta):

        def checker():

            satisfied = True

            bc_regex = filter_dict.get('_bc_regex')
            if bc_regex is not None:
                satisfied &= bool(re.search(bc_regex, bc))
            if satisfied is False:
                return False

            bc_meta_regex = filter_dict.get('_bc_meta_regex')
            if bc_meta_regex is not None:
                satisfied &= bool(re.search(bc_meta_regex, bc_meta))
            if satisfied is False:
                return False

            for key, function in filter_dict.items():
                if key not in ['_bc_regex', '_bc_meta_regex']:
                    satisfied &= function(self.__dict__.get(key))
                if satisfied is False:
                    return False

            return satisfied

        bc_meta = bc_meta + f'.{self.__dict__["_meta"]}'

        if checker() is True:
            return self

        else:

            new_dict = {}

            for key, value in self.__dict__.items():
                if type(value) is type(self):
                    out = value._filter_helper(filter_dict, bc + f'.{key}', bc_meta)
                    if out is not None:
                        new_dict = {**{key: out}, **new_dict}
                elif type(value) is list:
                    new_list = []
                    for item in value:
                        if type(item) is type(self):
                            out = item._filter_helper(filter_dict, bc + f'.{key}', bc_meta)
                            if out is not None:
                                new_list.append(out)
                    if len(new_list) > 0:
                        new_dict = {**{key: new_list}, **new_dict}

        if len(new_dict) > 0:
            return self.__class__({**{'_meta': self.__dict__['_meta']}, **new_dict})
        else:
            return None

