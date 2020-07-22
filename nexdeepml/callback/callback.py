from typing import Dict
from ..logger.logger import TQDMLogger
from ..base import base


class Callback(base.BaseWorker):
    """A class that is the base/parent class of any callbacks to be defined."""

    def __init__(self):
        """Initializes the callback class."""

        super().__init__()

# TODO: Do we get to save a config?


class CallbackManager(base.BaseManager):
    """"A class that manages Callback instances and calls their events on the occurrence of events."""

    def __init__(self):
        """Initializes the callback manager class."""

        super().__init__()


class TQDMCallback(Callback):
    """Creates and manages an instance of tqdm to take care of progress bar for training/testing the network."""

    def __init__(self, tqdm_config):
        """Initialize the callback for tqdm progress bar."""
        super().__init__()

        self._tqdm = TQDMLogger(tqdm_config)

    def on_epoch_begin(self, info: Dict):
        """Sets the total number of iterations and resets the tqdm progress bar.

        Parameters
        ----------
        info : Dict
            Dictionary containing the info:
                total_iterations : int
                    Total number of iterations in each epoch.
        """

        total_iterations: int = info['total_iterations']
        self._tqdm.reset(total_iterations)

    def on_batch_end(self, info: Dict):
        """Updates the tqdm progress bar.

        Parameters
        ----------
        info : Dict
            Dictionary containing the info:
                batch_size : int
                    Batch size to update the progress.
                For other entries in info, see TQDMLogger.
        """

        batch_size: int = info.pop('batch_size')
        self._tqdm.update(batch_size, info)
