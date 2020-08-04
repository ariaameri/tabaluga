from abc import ABC, abstractmethod
from ..util.config import ConfigParser
from ..base import base
import torch


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


class ModelPyTorchManager(ModelManager, torch.nn.Module, ABC):
    """Abstract class that manages pyTorch neural network models."""

    def __init__(self, config: ConfigParser = None):
        """Initialize the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance

        """

        ModelManager.__init__(self, config)
        torch.nn.Module.__init__(self)

    @abstractmethod
    def forward(self, x):
        """Implements the feedforward method of the neural net.

        Parameters
        ----------
        x
            Input to the neural net

        """

        raise NotImplementedError


class ModelPyTorch(Model, torch.nn.Module, ABC):
    """Class to serve as the base class of pyTorch models."""

    def __init__(self, config: ConfigParser = None):
        """Initializes the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance

        """

        Model.__init__(self, config)
        torch.nn.Module.__init__(self)

    @abstractmethod
    def forward(self, x):
        """The feedforward of the model.

        Parameters
        ----------
        x
            Input to the neural net

        """

        raise NotImplementedError
