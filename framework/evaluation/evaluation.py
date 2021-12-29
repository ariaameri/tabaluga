from ..base.base import BaseWorker, BaseEventManager
from ..util.config import ConfigParser
from abc import ABC, abstractmethod


class Evaluation(BaseWorker, ABC):
    """Abstract class to calculate the loss function value."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

    @abstractmethod
    def calculate(self, x: dict):
        """
        Calculates the evaluation function value.

        Parameters
        ----------
        x : dict
            data given in a dictionary

        Returns
        -------
        The value of the eval matrix

        """

        pass


class EvaluationManager(BaseEventManager, ABC):
    """Abstract class to manage instances of Eval class."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)
