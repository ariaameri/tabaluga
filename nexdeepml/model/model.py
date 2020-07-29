from ..util.config import ConfigParser
from ..base import base


class ModelManager(base.BaseEventManager):
    """Abstract class that manages neural network models."""

    def __init__(self):
        """Initializes this instance."""

        super().__init__()


class Model(base.BaseEventWorker):
    """Abstract class that contains the neural network model."""

    def __init__(self):
        """Initializes the instance.

        Creates the model and takes care of initial tasks.
        """

        super().__init__()
