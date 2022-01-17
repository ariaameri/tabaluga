from ..base.base import BaseWorker, BaseEventManager
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from abc import ABC, abstractmethod


class Loss(BaseWorker, ABC):
    """Abstract class to calculate the loss function value."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

    @abstractmethod
    def calculate(self, x: DataMuncher):
        """
        Calculates the loss function value.

        Parameters
        ----------
        x : DataMuncher
            data given in a DataMuncher

        Returns
        -------
        The value of the loss function

        """

        pass


class LossManager(BaseEventManager, ABC):
    """Abstract class to manage instances of Loss class."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)
