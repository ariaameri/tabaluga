from ..loss.loss import Loss, LossManager
from ..util.config import ConfigParser
from abc import ABC, abstractmethod


class Evaluation(Loss):
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

    @abstractmethod
    def _init_metric(self, x: dict):
        pass

    @abstractmethod
    def _update_metric(self, x: dict):
        pass

    @abstractmethod
    def _reset_metric(self, x: dict):
        pass

    @abstractmethod
    def save_metric(self, location: str):
        pass

    @abstractmethod
    def _compute_metric(self):
        pass


class EvaluationManager(LossManager):
    """Abstract class to manage instances of Eval class."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)
