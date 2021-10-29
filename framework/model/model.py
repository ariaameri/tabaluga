from abc import ABC
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
