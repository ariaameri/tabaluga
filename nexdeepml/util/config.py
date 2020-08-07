from typing import Dict, Any
import yaml
import re
import numpy as np


class ConfigParser:
    """A class that will contain config values and subclasses."""

    # Set static variables
    begin_configparser = f'\u2022'
    config_color = f'\033[38;5;209m'
    begin_list = f'-'
    # begin_list_color = f'{CCC.foreground.set_88_256.chartreuse4}'
    begin_list_color = f'\033[38;5;70m'
    begin_list = f'{begin_list_color}{begin_list}\033[0m'
    vertical_bar = f'\u22EE'
    # vertical_bar_color = f'{CCC.foreground.set_8_16.light_gray}'
    vertical_bar_color = f'\033[37m'
    vertical_bar_with_color = f'{vertical_bar_color}{vertical_bar}\033[0m'

    def __init__(self, config_dict: Dict = None):
        """Initializes the class based on the input config dictionary.

        Parameters
        ----------
        config_dict : Dict
            A dictionary containing all the configurations
        """

        # Check if no input was given
        if config_dict is None:
            config_dict = {}

        for key, value in config_dict.items():
            setattr(self, key, self._init_helper(value))

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

    def _init_helper(self, config: Any) -> Any:
        """Helper method for the constructor to recursively construct the config class.

        Parameter
        ---------
        config : Any
            A sub-config of the main config dictionary.
        """

        if type(config) == dict:
            return self.__class__(config)
        elif type(config) == list:
            out = [self._init_helper(item) for item in config]
        else:
            out = config

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
            out = {}
            for key, item in config.__dict__.items():
                out[key] = self.dict_representation(item)
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

        # Check if we have reach the root of the recursion, i.e. depth is zero
        if depth == 0:
            return ''

        # The final string to be returned
        out_string = ''

        # If no config is given, perform everything on the current instance
        if config is None:
            config = self

        if issubclass(type(config), type(self)):
            for key, item in config.__dict__.items():

                out_string += f'{self.begin_configparser} {self.config_color}{key}\033[0m'
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

        elif type(config) == list:

            # The final string of this section
            out_substring = ''
            for item in config:
                out_subsubstring = self.str_representation(item, depth)
                # Write begin_list at the beginning in green
                out_substring += \
                    f'{self.begin_list} {out_subsubstring}' \
                    if type(item) != type(self) \
                    else f'{self.begin_list_color}{self.begin_configparser}\033[0m {out_subsubstring[2:]}'

            out_string += out_substring

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

        if name in self.__dict__:
            raise Exception(f"Value of {name} exists. Cannot change value of {name} due to immutability.")
        else:
            self.__dict__[name] = value

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
        split = name.split('.')
        # list_entry = re.search(r'(\w+)\[(\d+)\]', split[0])

        # In case we have to change an attribute here at depth zero
        if len(split) == 1:
            parameters = {**self.__dict__, **{name: value}}  # Either update or create new attribute
            return self.__class__(parameters)
        # In case we have to go deeper to change an attribute
        else:
            parameters = {i: d for i, d in self.__dict__.items() if i != split[0]}
            # Find if we should update or create new attribute
            chooser = self.__dict__[split[0]] if split[0] in self.__dict__.keys() else self.__class__({})
            parameters = {**parameters, split[0]: chooser.update('.'.join(split[1:]), value)}
            return self.__class__(parameters)
