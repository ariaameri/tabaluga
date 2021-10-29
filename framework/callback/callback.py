"""
This file contains the classes that are used as callbacks.
Classes here are abstract, base classes, and non-abstract, which are the ones often used.

The layout is like this:

- We have only a single instance of the CallbackManager non-abstract chile whose job is to contain workers and call them.
This is the instance that the trainer will call and its job is to distribute the event.
The workers of the CallbackManager are of type Callback or ManagerCallback.

- Callback is the base class to be extended for every standalone callback that does not manage any other callbacks and
does the job by itself, rather than propagating it through its workers.

- ManagerCallback is the base class to be extended for every callback that manages other callbacks. Its job is to
distribute the event received to other Callback or ManagerCallback s. This is just a practice to group similar callbacks
together.

"""

from ..base import base
from ..util.config import ConfigParser
from typing import Dict, List
from abc import ABC
from ..util.calculation import Calculation


class Callback(base.BaseEventWorker):
    """An abstract class that is the base/parent class of any callbacks to be defined."""

    def __init__(self, config: ConfigParser = None, trainer=None):
        """Initializes the callback class.

        Parameters
        ----------
        config : ConfigParser
            The configuration for this instance and the rest of the instances it will initialize
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config)

        # Set the attributes
        self.trainer = trainer


class CallbackManager(base.BaseEventManager, ABC):
    """"An abstract class that manages Callback instances and calls their events on the occurrence of events."""

    def __init__(self, config: ConfigParser, trainer):
        """Initializes the callback manager class.

        Parameters
        ----------
        config : ConfigParser
            The configuration for this instance and the rest of the instances it will initialize
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config)

        # Set the attributes
        self.trainer = trainer


class ManagerCallback(Callback, base.BaseEventManager):
    """This abstract class initializes a single Manager and calls its events."""

    def __init__(self, config: ConfigParser, trainer):
        """Initializes the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this callback instance and the manager class.
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config, trainer)

        self._config = config

        self.workers: base.Workers = base.Workers()

        self.create_workers()

    def create_workers(self):
        """Creates and initializes a Manager instance."""

        pass


class TrainStatExpAverage(Callback):
    """Class to take the exponential moving average of the training metrics."""

    def __init__(self, config: ConfigParser = None, trainer=None):
        """Initializes the callback class.

        Parameters
        ----------
        config : ConfigParser
            The configuration for this instance and the rest of the instances it will initialize
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config, trainer)

    def on_batch_end(self, info: Dict = None):

        # Get the train statistics
        train_stat = self.trainer.train_statistics.find_one({'_bc': {'$regex': 'Train$'}})

        if train_stat.is_empty():
            return

        train_stat = train_stat.get()

        # Update the train statistics
        self.trainer.train_statistics = \
            self.trainer.train_statistics.update(
                {'Train': {'$exists': 1}},
                {'Train': self.get_exp_average(train_stat)}
            )

    def on_val_batch_end(self, info: Dict = None):

        # Get the train statistics
        train_stat = self.trainer.train_statistics.find_one({'_bc': {'$regex': 'Validation$'}})

        if train_stat.is_empty():
            return

        train_stat = train_stat.get()

        # Update the train statistics
        self.trainer.train_statistics = \
            self.trainer.train_statistics.update(
                {'Validation': {'$exists': 1}},
                {'Validation': self.get_exp_average(train_stat)}
            )

    def get_exp_average(self, stat):
        """Goes over the leaves (in depth) in `stat` and makes/updates 'Exp Average' node within that is the exponential
        average of eac leaf.

        Parameters
        ----------
        stat : LogHug
            The LogHug item to traverse through

        Return
        ------
        LogHug instance with updated values.

        """

        # Take the leaves and the branches
        parameters_leaf = \
            {key: value for key, value in stat.get_parameters().items() if value.is_leaf()}
        parameters_branch = \
            {key: value for key, value in stat.get_parameters().items() if value.is_branch() and key != 'Exp Average'}

        # Make or get the Exp Average node
        exp_average = stat.get_option('Exp Average')
        if exp_average.is_empty():
            from ..logger.log_hug import LogHug
            exp_average = LogHug()
        else:
            exp_average = exp_average.get()

        # Update the exponential average for the shallow leaves
        for key, value in parameters_leaf.items():
            exp_average = exp_average.update(
                {},
                {key: Calculation.exp_average(item=exp_average.get_or_else(key, None), d_item=value.get())}
            )

        # Update the exponential average for the deeper leaves
        sub_exp_average = {key: self.get_exp_average(value) for key, value in parameters_branch.items()}

        return stat.update({}, {'$set': {'Exp Average': exp_average, **sub_exp_average}})
