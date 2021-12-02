from __future__ import annotations
import time
from ..util.config import ConfigParser
from ..util.panacea import Panacea
from ..base.base import BaseEventManager
from .the_progress_bar import TheProgressBar, TheProgressBarParallelManager
from ..communicator import mpi
from .logger_sole import Logger
from abc import ABC
from typing import Dict, Callable, List
import numpy as np


class LoggerManager(BaseEventManager, ABC):
    """"An abstract class that manages Logger instances and calls their events on the occurrence of events."""

    def __init__(self, config: ConfigParser = None):
        """Initializes the logger manager class.

        Parameters
        ----------
        config : ConfigParser
            The configuration for this instance and the rest of the instances it will initialize

        """

        super().__init__(config)


class TheProgressBarLogger(Logger):
    """A logger for the progress bar that takes the control of stdout.

    WARNING: It does not exactly behave the same way as the Logger class. It is just a wrapper/manager
                class for the TheProgressBarLogger class.
    """

    def __init__(self, config: ConfigParser = None):
        """Initialize the logger and the TheProgressBar instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this callback instance and the data manager class.

        """

        # Making sure the logger is going to write to the console
        # Make sure it does not write any prefix
        config = config.update({}, {'$set': {'console': True, 'format': ''}})

        super().__init__(config)

        # Create the instance of the TheProgressBar
        self._the_progress_bar = TheProgressBar(self.console_file, self._config)
        self._the_progress_bar_manager = \
            TheProgressBarParallelManager(self.console_file, self._config) \
            if mpi.mpi_communicator.is_distributed() \
            else None

        # The number of total items and epochs
        self._total: int = -1
        self._n_epochs: int = -1

    def __enter__(self):

        self.activate()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()

    def terminate(self):
        """Method to get called upon termination."""

        # if we are terminating, we just want to pause the printing and all but not let go of the console
        self._the_progress_bar.pause()
        if self._the_progress_bar_manager is not None:
            self._the_progress_bar_manager.pause()

    def activate(self) -> TheProgressBarLogger:
        """Activates the TheProgressBar instance.

        Has to be called to initiate its running.

        """

        self._the_progress_bar.activate()
        if mpi.mpi_communicator.is_distributed():
            self._the_progress_bar_manager.activate()

        return self

    def set_number_epochs(self, epochs: int) -> None:
        """Sets the total number of epochs.

        Parameters
        ----------
        epochs : int
            Number of epochs

        """

        # The number of total epochs
        self._n_epochs: int = epochs

    def set_number_items(self, items: int) -> None:
        """Sets the total number of items in this progress.

        Parameters
        ----------
        items : int
            Number of items

        """

        # set it!
        self._the_progress_bar.set_number_items(items)

    def _custom_reset(self, total: int = -np.inf, return_to_line_number: int = 0) -> TheProgressBarLogger:
        """Set the total number of iterations and resets the the_progress_bar.

        Parameters
        ----------
        total : int, optional
            the total number of items expected. if not set, will be set to the default value.
        return_to_line_number: int, optional
            The number of the line to return to the beginning of, after the progress bar

        """

        self._the_progress_bar.reset(return_to_line_number=return_to_line_number)
        self._the_progress_bar.set_number_items(total)

        # wait for all workers to be reset
        if mpi.mpi_communicator.is_distributed():
            mpi.mpi_communicator.barrier()
            # wait a random time, just to be sure :D
            time.sleep(1)
            # now, reset the manager
            self._the_progress_bar_manager.reset(return_to_line_number=return_to_line_number)

        self._total = total

        return self

    def reset(self, total: int = -np.inf) -> TheProgressBarLogger:
        """Set the total number of iterations and prints and resets the the_progress_bar.

        Parameters
        ----------
        total : int, optional
            the total number of items expected. if not set, will be set to the default value.

        """

        return self._custom_reset(total=total, return_to_line_number=-1)

    def reset_bar_only(self, total: int = -np.inf) -> TheProgressBarLogger:
        """Set the total number of iterations and resets only the bar of the the_progress_bar.

        Parameters
        ----------
        total : int, optional
            the total number of items expected. if not set, will be set to the default value.

        """

        return self._custom_reset(total=total, return_to_line_number=0)

    def reset_to_next_line(self, total: int = -np.inf) -> TheProgressBarLogger:
        """Set the total number of iterations and resets only the bar of the the_progress_bar.

        Parameters
        ----------
        total : int, optional
            the total number of items expected. if not set, will be set to the default value.

        """

        return self._custom_reset(total=total, return_to_line_number=1)

    def close(self) -> None:
        """Finishes and closes the TheProgressBar instance."""

        self._the_progress_bar.deactivate()
        if mpi.mpi_communicator.is_distributed():
            self._the_progress_bar_manager.deactivate()

    def pause(self) -> None:
        """Pauses the TheProgressBar instance."""

        self._the_progress_bar.pause()
        if mpi.mpi_communicator.is_distributed():
            self._the_progress_bar_manager.pause()

    def resume(self) -> None:
        """Resumes the TheProgressBar instance."""

        self._the_progress_bar.resume()
        if mpi.mpi_communicator.is_distributed():
            self._the_progress_bar_manager.resume()

    def update(self, update_count: int, msg_dict: Dict = None) -> None:
        """Update the TheProgressBar progress bar with description set to message.

        Parameters
        ----------
        update_count : int
            The amount that should be added to the TheProgressBar instance.
        msg_dict : Dict, optional
            Contains the dictionary message to be set as the progress bar description.
        """

        # first, update the bar
        self._the_progress_bar.update(update_count)

        # # generate the message and aggregation data
        # message, data = self._generate_message_from_loghug(msg_dict)
        #
        # # set all
        # self._the_progress_bar.set_aggregation_data(data)
        #
        # self._the_progress_bar.set_description_after(message)
        #
        # self._the_progress_bar.set_description_short_after("yellow")
        # self._the_progress_bar.set_description_short_before(f'At epoch {msg_dict["epoch"]}/{self._n_epochs} of {msg_dict["mode"]}:')

    def set_aggregator_function_data(self, data: Panacea):
        """
        Sets the aggregator data that is used to process the aggregated string.

        Parameters
        ----------
        data : Panacea
            Aggregation data, here we require the data to be a Panacea subclass

        """

        self._the_progress_bar.set_aggregation_data(data)

    def set_aggregator_function_full(self, func: Callable[[List[Panacea]], str]):
        """
        Sets the aggregator function that is used to process the aggregated data and give full output.

        Parameters
        ----------
        func : Callable[[list[Panacea, str]]]
            function that is given a list of to-be-aggregated data and should return the full aggregation.

        """

        self._the_progress_bar.set_aggregator_function_full(func)
        if mpi.mpi_communicator.is_distributed():
            self._the_progress_bar_manager.set_aggregator_function_full(func)

    def set_aggregator_function_short(self, func: Callable[[List[Panacea]], str]):
        """
        Sets the aggregator function that is used to process the aggregated data and give short output.

        Parameters
        ----------
        func : Callable[[list[Panacea, str]]]
            function that is given a list of to-be-aggregated data and should return the short aggregation.

        """

        self._the_progress_bar.set_aggregator_function_short(func)
        if mpi.mpi_communicator.is_distributed():
            self._the_progress_bar_manager.set_aggregator_function_short(func)
