from ..base.base import BaseWorker, BaseEventManager
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from abc import ABC, abstractmethod


class Optimizer(BaseWorker, ABC):
    """Abstract class to do the optimization of the neural net."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

    @abstractmethod
    def optimize(self, x: DataMuncher):
        """
        Optimize the neural net.

        Parameters
        ----------
        x : DataMuncher
            data given in a DataMuncher

        Returns
        -------

        """

        pass


class OptimizerManager(BaseEventManager, ABC):
    """Abstract class to manage instances of Optimizer class."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)
