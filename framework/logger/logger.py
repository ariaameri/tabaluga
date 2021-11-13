from __future__ import annotations
from ..util.config import ConfigParser
from .log_hug import LogHug
from ..util.symbols_unicode import SYMBOL_UNICODE_CONFIG as SUC
from ..base.base import BaseEventManager
from .the_progress_bar import TheProgressBarColored
from .logger_sole import LoggerConsoleFile, Logger
from abc import ABC
from typing import Dict, List, Union, Any
import numpy as np
import signal
import re
import colored


class LoggerManager(BaseEventManager, ABC):
    """"An abstract class that manages Logger instances and calls their events on the occurrence of events."""

    def __init__(self, config: ConfigParser = None):
        """Initializes the logger manager class.

        Parameters
        ----------
        config : ConfigParser
            The configuration for this instance and the rest of the instances it will initialize

        """

        # Add the console handler
        # Update each and every logger config that we have within our config
        self.console_file = LoggerConsoleFile().activate()
        config = config.update({'_bc': {'$regex': r'\.\w+$'}}, {'$set': {'console_handler': self.console_file}})

        super().__init__(config)

    def on_os_signal(self, info: Dict = None):

        os_signal = info['signal']

        if os_signal == signal.SIGINT or os_signal == signal.SIGTERM:
            self.console_file.deactivate()


class TheProgressBarLogger(Logger):
    """A logger for the progress bar that takes the control of stdout.

    WARNING: It does not exactly behave the same way as the Logger class. It is just a wrapper/manager
                class for the TheProgressBarLogger class.
    """
    # TODO: Conform the TheProgressBarLogger or a new class with Logger class

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
        self._the_progress_bar = TheProgressBarColored(self.console_file)

        # The number of total items and epochs
        self._total: int = -1
        self._n_epochs: int = -1

    def __enter__(self):

        self.activate()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()

    def activate(self) -> TheProgressBarLogger:
        """Activates the TheProgressBar instance.

        Has to be called to initiate its running.

        """

        self._the_progress_bar.activate()

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

    def _custom_reset(self, total: int, return_to_line_number: int = 0) -> TheProgressBarLogger:
        """Set the total number of iterations and resets the the_progress_bar.

        Parameters
        ----------
        total : int
            the total number of items expected.
        return_to_line_number: int, optional
            The number of the line to return to the beginning of, after the progress bar

        """

        self._the_progress_bar.reset(return_to_line_number=return_to_line_number)
        self._the_progress_bar.set_number_items(total)
        self._total = total

        return self

    def reset(self, total: int) -> TheProgressBarLogger:
        """Set the total number of iterations and prints and resets the the_progress_bar.

        Parameters
        ----------
        total : int
            the total number of items expected.

        """

        return self._custom_reset(total=total, return_to_line_number=-1)

    def reset_bar_only(self, total: int) -> TheProgressBarLogger:
        """Set the total number of iterations and resets only the bar of the the_progress_bar.

        Parameters
        ----------
        total : int
            the total number of items expected.

        """

        return self._custom_reset(total=total, return_to_line_number=0)

    def reset_to_next_line(self, total: int) -> TheProgressBarLogger:
        """Set the total number of iterations and resets only the bar of the the_progress_bar.

        Parameters
        ----------
        total : int
            the total number of items expected.

        """

        return self._custom_reset(total=total, return_to_line_number=1)

    def close(self) -> None:
        """Finishes and closes the TheProgressBar instance."""

        self._the_progress_bar.deactivate()

    def pause(self) -> None:
        """Pauses the TheProgressBar instance."""

        self._the_progress_bar.pause()

    def resume(self) -> None:
        """Resumes the TheProgressBar instance."""

        self._the_progress_bar.resume()

    def update(self, update_count: int, msg_dict: Dict = None) -> None:
        """Update the TheProgressBar progress bar with description set to message.

        Parameters
        ----------
        update_count : int
            The amount that should be added to the TheProgressBar instance.
        msg_dict : Dict, optional
            Contains the dictionary message to be set as the progress bar description.
            Will be passed to the _generate_message method, read there for more info.
        """

        self._the_progress_bar.update(update_count)

        # message = self._generate_message(msg_dict)
        message = self._generate_message_from_loghug(msg_dict)
        self._the_progress_bar.set_description_after(message)

    def _generate_message(self, msg_dict: Dict) -> str:
        """Generates a string based on the input to be used as tqdm bar description.

        If msg is None, empty string will be returned.

        Parameters
        ---------
        msg_dict : Dict
            Dictionary containing the information to be used. Contains:
                epoch: int
                loss: float
                val_loss: float, optional
        """

        # Make a copy of the dictionary to modify it
        msg_dict_copy = {**msg_dict}

        title = ''

        if msg_dict is None:
            return title

        # Find the length of the total epochs
        # get and remove the 'epoch' item from the dictionary
        # and reformat the string accordingly
        ep_len = int(np.ceil(np.log10(self._n_epochs)))
        epoch = msg_dict_copy.pop('epoch')
        title = f'{colored.fg("green_4")}{SUC.heavy_teardrop_spoked_asterisk} '
        title += f'{colored.fg("chartreuse_3a")}Epoch ' \
                 f'{colored.fg("green_3a")}{epoch:{ep_len}d}' \
                 f'{colored.fg("grey_27")}/' \
                 f'{colored.fg("dark_green")}{self._n_epochs}' \
                 f'{colored.attr("reset")}'

        message_dict = {}

        # Check if we have training values for logging, starting with 'train_'
        # get the item from the dictionary and delete it
        # Generate the message set and add it to the total message
        train_items = {key[len('train_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
                       if key.startswith('train_')}
        if train_items:
            message_dict = {**message_dict, 'Train': train_items}

        # Check if we have validation values for logging, starting with 'train_'
        # get the item from the dictionary and delete it
        # Generate the message set and add it to the total message
        val_items = {key[len('val_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
                     if key.startswith('val_')}
        if val_items:
            message_dict = {**message_dict, 'Validation': val_items}

        if msg_dict_copy:
            message_dict = {**message_dict, 'Others': msg_dict_copy}

        # Generate the message
        message = LogHug(message_dict).str_representation_with_title(title=title)

        # Indent the message
        message = '\n\t' + '\n\t'.join(message.split(f'\n'))

        return message

    def _generate_message_from_loghug(self, msg_dict: Dict) -> str:
        """Generates a string based on the input to be used as tqdm bar description.

        If msg is None, empty string will be returned.

        Parameters
        ---------
        msg_dict : Dict
            Dictionary containing the information to be used. Contains:
                epoch: int
                loss: float
                val_loss: float, optional
        """

        # Find the length of the total epochs
        # get and remove the 'epoch' item from the dictionary
        # and reformat the string accordingly
        ep_len = int(np.ceil(np.log10(self._n_epochs)))
        epoch = msg_dict.get('epoch')
        title = f'\n' \
                f'{colored.fg("green_4")}{SUC.heavy_teardrop_spoked_asterisk} '
        title += f'{colored.fg("chartreuse_3a")}Epoch ' \
                 f'{colored.fg("green_3a")}{epoch:{ep_len}d}' \
                 f'{colored.fg("grey_27")}/' \
                 f'{colored.fg("dark_green")}{self._n_epochs}' \
                 f'{colored.attr("reset")}'

        # If there is a stat entry, use it
        if msg_dict.get('stat') is not None:
            message = msg_dict.get('stat').str_representation_with_title(title=title)
        else:
            message = title

        # Indent the messages once
        message = re.sub(r'(^|\n)', r'\1\t', message)

        return message
