import pathlib
from abc import ABC, abstractmethod
from ..util.config import ConfigParser
from ..base import base


class ModelManager(base.BaseEventManager, ABC):
    """Abstract class that manages neural network models."""

    def __init__(self, config: ConfigParser = None):
        """Initialize the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance

        """

        super().__init__(config)


class Model(base.BaseWorker, ABC):
    """Abstract class that contains the neural network model."""

    def __init__(self, config: ConfigParser = None):
        """Initializes the instance.

        Creates the model and takes care of initial tasks.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance

        """

        super().__init__(config)

    @abstractmethod
    def save(self, path: pathlib.Path) -> bool:
        """
        Saves the model.

        Parameters
        ----------
        path : pathlib.Path
            The path at which the model should be saved

        Returns
        -------
        bool
            whether the save was successful

        """

        raise NotImplementedError

    @abstractmethod
    def load(self, path: pathlib.Path) -> bool:
        """
        Loads the model.

        Parameters
        ----------
        path : pathlib.Path
            The path at which the model should be loaded

        Returns
        -------
        bool
            whether the loading was successful

        """

        raise NotImplementedError
