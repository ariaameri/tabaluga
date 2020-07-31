from ..base import base
from ..util.config import ConfigParser
from typing import Dict, List
from ..logger.logger import TQDMLogger
from abc import ABC
from ..dataloader import dataloader
# from ..trainer import trainer
from collections import OrderedDict


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


class TQDMCallback(Callback):
    """Creates and manages an instance of tqdm to take care of progress bar for training/testing the network."""

    def __init__(self, tqdm_config: ConfigParser):
        """Initialize the callback for tqdm progress bar.

        Parameters
        ----------
        tqdm_config : ConfigParser
            The configuration for this instance

        """

        super().__init__(tqdm_config)

        # Save tqdm config
        self._tqdm_config: ConfigParser = tqdm_config

        # Set the tqdm instances
        self._train_tqdm: TQDMLogger
        self._val_tqdm: TQDMLogger

    def on_train_begin(self, info: Dict = None):
        """Creates the tqdm train instance."""

        self._train_tqdm: TQDMLogger = TQDMLogger(self._tqdm_config)

    def on_train_end(self, info: Dict = None):
        """Closes the tqdm train instance."""

        self._train_tqdm.close()

    def on_val_begin(self, info: Dict = None):
        """Creates the tqdm validation instance."""

        self._val_tqdm: TQDMLogger = TQDMLogger(self._tqdm_config)

    def on_val_end(self, info: Dict = None):
        """Closes the tqdm validation instance."""

        self._val_tqdm.close()

    def on_epoch_begin(self, info: Dict = None):
        """Sets the total number of iterations and resets the tqdm train progress bar.

        Parameters
        ----------
        info : Dict
            Dictionary containing the info:
                total_iterations : int
                    Total number of iterations in each epoch.
        """

        total_iterations: int = info['total_iterations']
        self._train_tqdm.reset(total_iterations)

    def on_batch_end(self, info: Dict = None):
        """Updates the tqdm train progress bar.

        Parameters
        ----------
        info : Dict
            Dictionary containing the info:
                batch_size : int
                    Batch size to update the progress.
                For other entries in info, see TQDMLogger.
        """

        batch_size: int = info.pop('batch_size')
        self._train_tqdm.update(batch_size, info)

    def on_val_epoch_begin(self, info: Dict = None):
        """Sets the total number of iterations and resets the tqdm validation progress bar.

        Parameters
        ----------
        info : Dict
            Dictionary containing the info:
                total_iterations : int
                    Total number of iterations in each epoch.
        """

        total_iterations: int = info['total_iterations']
        self._val_tqdm.reset(total_iterations)

    def on_val_batch_end(self, info: Dict = None):
        """Updates the tqdm validation progress bar.

        Parameters
        ----------
        info : Dict
            Dictionary containing the info:
                batch_size : int
                    Batch size to update the progress.
                For other entries in info, see TQDMLogger.
        """

        batch_size: int = info.pop('batch_size')
        self._val_tqdm.update(batch_size, info)


# TODO: Does this have to extend Callback or CallbackManager?
class ManagerCallback(Callback):
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

