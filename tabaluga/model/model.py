import pathlib
from abc import ABC, abstractmethod
from typing import List
from ..util.config import ConfigParser
from ..base import base
from ..util.result import Result


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

    @abstractmethod
    def save(self, path: pathlib.Path) -> Result[List[pathlib.Path], Exception]:
        """
        Takes care of saving its models.

        Parameters
        ----------
        path : pathlib.Path
            The path at which the models should be saved, this can be used any how>

        Returns
        -------
        Result[List[pathlib.Path], Exception]
            paths of the saved models wrapped in Result

        """

        raise NotImplementedError

    @abstractmethod
    def load(self, path: pathlib.Path) -> Result[List[pathlib.Path], Exception]:
        """
        Takes care of loading its models.

        Parameters
        ----------
        path : pathlib.Path
            The path at which the model should be loaded, this can be used any how>

        Returns
        -------
        Result[List[pathlib.Path], Exception]
            paths of the loaded models wrapped in Result

        """

        raise NotImplementedError


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
    def save(self, path: pathlib.Path) -> Result[pathlib.Path, Exception]:
        """
        Saves the model.

        Parameters
        ----------
        path : pathlib.Path
            The path at which the model should be saved

        Returns
        -------
        Result[pathlib.Path, Exception]
            the saved path wrapped in result

        """

        raise NotImplementedError

    @abstractmethod
    def load(self, path: pathlib.Path) -> Result[pathlib.Path, Exception]:
        """
        Loads the model.

        Parameters
        ----------
        path : pathlib.Path
            The path at which the model should be loaded

        Returns
        -------
        Result[pathlib.Path, Exception]
            the loaded path wrapped in result

        """

        raise NotImplementedError
