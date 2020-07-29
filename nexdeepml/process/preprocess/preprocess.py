from abc import ABC

from ...util.config import ConfigParser
from ...base import base


class PreprocessManager(base.BaseEventManager, ABC):
    """This abstract class manages the Pre-Process instances."""

    def __init__(self, config: ConfigParser):
        """Initializes the pre-process manager.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance and its workers.

        """

        super().__init__(config)


class Preprocess(base.BaseEventWorker):
    """A class that is the base/parent class of any pre-process to be defined."""

    def __init__(self, config: ConfigParser):
        """Initializes the pre-process instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)


