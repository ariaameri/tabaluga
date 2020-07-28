from ..base import base
from ..util.config import ConfigParser
from typing import Dict
from ..logger.logger import TQDMLogger


class Callback(base.BaseWorker):
    """A class that is the base/parent class of any callbacks to be defined."""

    def __init__(self):
        """Initializes the callback class."""

        super().__init__()


class CallbackManager(base.BaseManager):
    """"A class that manages Callback instances and calls their events on the occurrence of events."""

    def __init__(self):
        """Initializes the callback manager class."""

        super().__init__()


class TQDMCallback(Callback):
    """Creates and manages an instance of tqdm to take care of progress bar for training/testing the network."""

    def __init__(self, tqdm_config: ConfigParser):
        """Initialize the callback for tqdm progress bar."""
        super().__init__()

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
