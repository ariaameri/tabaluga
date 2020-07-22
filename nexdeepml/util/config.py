from typing import Dict, Any
import yaml


class ConfigParser:
    """A class that will contain config values and subclasses."""

    def __init__(self, config_dict: Dict):
        """Initializes the class based on the input config dictionary.

        Parameters
        ----------
        config_dict : Dict
            A dictionary containing all the configurations
        """

        for key, value in config_dict.items():
            setattr(self, key, self._init_helper(value))

    def _init_helper(self, config: Any) -> Any:
        """Helper method for the constructor to recursively construct the config class.

        Parameter
        ---------
        config : Any
            A sub-config of the main config dictionary.
        """

        if type(config) == dict:
            return ConfigParser(config)
        if type(config) == list:
            out = []
            for item in config:
                out.append(self._init_helper(item))
        else:
            out = config

        return out

    def __str__(self) -> str:
        """Method to help with the visualization of the configuration in YAML style."""

        config = self._str_helper(self)

        return yaml.dump(config)

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
