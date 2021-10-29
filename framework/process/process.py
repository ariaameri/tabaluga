from abc import ABC, abstractmethod
from ..base import base
from ..util.config import ConfigParser


class ProcessManager(base.BaseEventManager, ABC):
    """This abstract class manages the Pre- and Post-Process instances or Managers."""

    def __init__(self, config: ConfigParser):
        """Initializes the class.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance and its workers.

        """

        super().__init__(config)


class Process(base.BaseWorker, ABC):
    """Abstract class for Process instances."""

    def __init__(self, config: ConfigParser):
        """Initializes the class.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance.

        """

        super().__init__(config)

    @abstractmethod
    def process(self, data):
        """Does the act of processing.

        Parameters
        ----------
        data
            Input data

        Returns
        -------
        Processed data

        """

        pass