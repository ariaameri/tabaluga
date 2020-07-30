from typing import Dict, Any
import yaml
import re


class ConfigParser:
    """A class that will contain config values and subclasses."""

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
            return ConfigParser(config)
        elif type(config) == list:
            out = []
            for item in config:
                out.append(self._init_helper(item))
        else:
            out = config

        return out

    def __str__(self) -> str:
        """Method to help with the visualization of the configuration in YAML style."""

        # Get dictionary representation of the current instance
        config = self._str_helper(self)

        # Get yaml string representation
        out_string = yaml.dump(config)

        # Add bullet at the beginning of each class, indent the lists, and make indentations with tabs
        out_string = out_string.replace("-", "  -")
        out_string = re.sub(r' {2}', '\t', out_string)
        out_string = re.sub(r'(\n|^)(\s*)(\w)', r'\1\2' + f'\u2022 ' + r'\3', out_string)

        return out_string

    def _str_helper(self, config: Any) -> Dict:
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

        # Check for the type of the input and act accordingly
        if type(config) == ConfigParser:
            out = {}
            for key, item in config.__dict__.items():
                out[key] = self._str_helper(item)
        elif type(config) == list:
            out = []
            for item in config:
                out.append(self._str_helper(item))
        else:
            out = config

        return out

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
