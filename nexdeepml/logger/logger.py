from __future__ import annotations
from ..util.config import ConfigParser
from ..util.console_colors import CONSOLE_COLORS_CONFIG as CCC
from ..util.symbols_unicode import SYMBOL_UNICODE_CONFIG as SUC
from ..base.base import BaseWorker, BaseEventManager
from .the_progress_bar import TheProgressBarColored
from abc import ABC
from typing import Dict, List, Union
import logging
import sys
from tqdm import tqdm
import numpy as np
import io
from datetime import datetime
from copy import deepcopy


class Logger(BaseWorker):
    """An abstract base/parent class for all logger classes."""

    # Keep track of how many logger instances we have
    _counter: List[int] = [0]

    # Keep track of supported logger functionality/abilities
    log_abilities = ['debug', 'report', 'info', 'warning', 'error']

    # TODO: Figure out the way configurations have to be passed to the class

    def __init__(self, config: ConfigParser):
        """Initializes the logger class

        Parameters
        ----------
        config : ConfigParser
            The configuration for the logger class
            It should consist of the following attributes
                name : str, optional
                    The name of the logger
                level : {logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}, optional
                    The level at which logging should happen.
                format : str, optional
                    The format at which the logger should log
                console : boolean, optional
                    Whether or not to log to console. If false, will write to file whose path is provided by file_name
                file_name : str, optional
                    The path of the file to write the log file to. Can be omitted if console is False

        """

        super().__init__(config)

        # The level at which we log
        self._level: int = config.level or logging.INFO

        # Get the logger
        self._logger = logging.getLogger(config.name or str(self._counter[0]))
        self._counter[0] += 1
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Suppress _logger output to stdout

        # TODO: Should we have logging to both the console and the file?
        # Determine whether to write to file or console and get the handlers
        self.console = config.console
        if self.console is True:
            # Get the handler
            self.console_file: Union[LoggerConsoleFile, io.TextIOWrapper] = config.console_handler or sys.stdout
            self._handler = logging.StreamHandler(self.console_file)
        else:
            file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f'{file_name}.txt'
            self._handler = logging.FileHandler(config.file_name or file_name)

        # Set the level, format, and attach
        self._handler.setLevel(config.level or logging.INFO)
        self._format = config.format or self._create_format()
        self._handler.setFormatter(
            logging.Formatter(
                self._format
            ))
        self._logger.addHandler(self._handler)

    def get_abilities(self) -> List[str]:
        """Method to return the list of abilities for the logger.

        Returns
        -------
        A list of all the levels at which this logger can perform

        """

        return self.log_abilities

    def _create_format(self) -> str:
        """Creates a custom logger format and returns the string.

        Returns
        -------
        The string containing the customized logger format

        """

        format = ''

        if self.console is False:
            format += f'{CCC.foreground.set_88_256.grey50}' + '%(asctime)s '
            format += f'{CCC.foreground.set_88_256.grey42}' + '- '
        else:
            format += f'{CCC.foreground.set_88_256.dodgerblue3}' + f'{SUC.rightwards_arrow_to_bar} '
        format += f'{CCC.foreground.set_88_256.gold1}' + '%(name)s'
        format += f'{CCC.foreground.set_88_256.grey42}' + ' '
        format += f'{CCC.reset.all}' + '%(message)s'

        return format

    def log(self, msg: str, level: str = 'debug') -> None:
        """Logs the given message at the given level.

        Parameters
        ----------
        msg : str
            String message to log
        level : str
            The level at which the message should be logged
        """

        # Check if the logger has the mentioned level
        assert level in self.log_abilities, \
            f'Level of logging, {level}, not accepted.\nSupported levels are {", ".join(self.get_abilities())}.'

        # Log
        getattr(self, level)(msg)

    def report(self, msg: str) -> None:
        """Writes the message given as a report.

        A report is logged the same as info but with different declaration

        Parameters
        ----------
        msg : str
            The message to be written as report

        """

        # Adds colored 'report: ' to the beginning of the message
        report_message = f'{CCC.foreground.set_88_256.green4}'\
                         f'report: '\
                         f'{CCC.reset.all}'
        report_message += msg

        self._logger.info(report_message)

    def info(self, msg: str) -> None:
        """Writes the message given as an info.

        Parameters
        ----------
        msg : str
            The message to be written as info

        """

        # Adds colored 'info: ' to the beginning of the message
        info_message = f'{CCC.foreground.set_88_256.green3}' \
                       f'info: ' \
                       f'{CCC.reset.all}'
        info_message += msg

        self._logger.info(info_message)

    def warning(self, msg: str) -> None:
        """Writes the message given as a warning.

        Parameters
        ----------
        msg : str
            The message to be written as warning

        """

        # Adds colored 'warning: ' to the beginning of the message
        warning_message = f'{CCC.foreground.set_88_256.red1}'\
                          f'warning: '\
                          f'{CCC.reset.all}'
        warning_message += msg

        self._logger.warning(warning_message)

    def error(self, msg: str) -> None:
        """Writes the message given as an error.

        Parameters
        ----------
        msg : str
            The message to be written as error

        """

        # Adds colored 'ERROR: ' to the beginning of the message and color the message as well
        error_message = f'{CCC.background.set_8_16.red}{CCC.foreground.set_8_16.white}'\
                        f'ERROR: '
        error_message += msg
        error_message += f'{CCC.reset.all}'

        self._logger.error(error_message)

    def debug(self, msg: str) -> None:
        """Writes the message given as an debug.

        Parameters
        ----------
        msg : str
            The message to be written as debug

        """

        # Adds colored 'debug: ' to the beginning of the message
        debug_message = f'{CCC.foreground.set_88_256.indianred2}'\
                        f'debug: '\
                        f'{CCC.reset.all}'
        debug_message += msg

        self._logger.debug(debug_message)

    def set_level(self, level: int) -> None:
        """Sets the level of logging.

        Parameters
        ----------
        level : {logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
            The level at which logging should happen.

        """

        self._level = level
        self._handler.level = level


class LoggerConsoleFile:
    """Class to hold the console handler and act as one."""

    def __init__(self):
        
        # Keep the stack list of console (file) handlers to write to
        self.handler = [sys.stdout]

    def add_handler(self, handler):
        """Add a console (file) handler to the stack.

        Parameters
        ----------
        handler
            A console handler

        """

        # Add the handler
        self.handler.append(handler)

    def revert(self):
        """Reverts the current handler to the previous one."""

        # Revert only when it has more than 1 item in the stack
        if len(self.handler) > 1:
            del self.handler[-1]

    def __getattr__(self, item):

        # Get the attribute from the latest file
        attr = getattr(self.handler[-1], item)

        return attr


class LoggerManager(BaseEventManager, ABC):
    """"An abstract class that manages Logger instances and calls their events on the occurrence of events."""

    def __init__(self, config: ConfigParser):
        """Initializes the logger manager class.

        Parameters
        ----------
        config : ConfigParser
            The configuration for this instance and the rest of the instances it will initialize

        """

        # Add the console handler
        self.console_file = LoggerConsoleFile()
        config = config.update('console_handler', self.console_file)

        super().__init__(config)


class TQDMLogger(Logger, io.StringIO):

    # TODO: Write the doc for the config argument

    def __init__(self, config: ConfigParser):
        """Initialize the tqdm logger instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this callback instance and the data manager class.

        """

        # Making sure the logger is going to write to the console
        # Make sure it does not write any prefix
        config = config.update('console', True).update('format', '')

        self._config = config

        Logger.__init__(self, config)
        io.StringIO.__init__(self)

        self._tqdm = tqdm(
            position=self._counter[0],
            bar_format=''
                       f'{CCC.foreground.set_88_256.grey74}'
                       '{percentage:3.0f}% '
                       f'{CCC.foreground.set_88_256.cornflowerblue}'
                       '|'
                       f'{CCC.foreground.set_88_256.steelblue4}'
                       '{bar}'
                       f'{CCC.foreground.set_88_256.cornflowerblue}'
                       '| '
                       f'{CCC.foreground.set_88_256.gold1}'
                       '{n_fmt}'
                       f'{CCC.foreground.set_88_256.grey}'
                       '/'
                       f'{CCC.foreground.set_88_256.orange2}'
                       '{total_fmt} '
                       f'{CCC.foreground.set_88_256.grey50}' 
                       '[{elapsed}<{remaining}, ' '{rate_fmt}{postfix}] '
                       f'{CCC.reset.all}'
                       '{desc}',
            file=self  # Write to this log handler instead of stderr
        )
        # Bookkeeping for tqdm
        self.buf: str = ''

        # The number of total iterations
        self._total: int = -1

        self._n_epochs: int = -1

    def set_number_epochs(self, epochs: int):
        """Sets the total number of epochs.

        Parameters
        ----------
        epochs : int
            Number of epochs

        """

        # The number of total epochs
        self._n_epochs: int = epochs

    def reset(self, total: int) -> None:
        """Set the total number of iterations and resets the tqdm.

        Parameters
        ----------
        total : int
            the total number of iterations expected."""

        self._tqdm.reset(total=total)
        self._total = total

    def close(self) -> None:
        """Finishes and closes the tqdm instance."""

        self._tqdm.close()

    def update(self, update_count: int, msg_dict: Dict = None) -> None:
        """Update the tqdm progress bar with description set to message.

        Parameters
        ----------
        update_count : int
            The amount that should be added to the tqdm instance.
        msg_dict : Dict, optional
            Contains the dictionary message to be set as the progress bar description.
            Will be passed to the _generate_message method, read there for more info.
        """

        self._tqdm.update(update_count)

        message = self._generate_message(msg_dict)
        self._tqdm.set_description_str(message)

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

        def _generate_message_set(msg_set_dict: Dict) -> str:
            """Function to help generate the set of messages for training, validation, ... .

            Parameters
            ----------
            msg_set_dict : dict
                The dictionary containing the information needed. Some are
                    _title: containing the title of the set

            Returns
            -------
            The generated string of the information set
            """

            message = ''

            if len(msg_set_dict) > 1:
                message += f'\t' * 2
                title = str(msg_set_dict.pop('_title'))
                message += f'{CCC.foreground.set_88_256.deepskyblue5}{SUC.right_facing_armenian_eternity_sign} '
                message += f'{CCC.foreground.set_88_256.deepskyblue3}{title}'
                message += f'{CCC.reset.all}'
                message += f'\n'
                for key, value in sorted(msg_set_dict.items()):
                    message += f'\t' * 3
                    message += f'{SUC.horizontal_bar} '
                    message += f'{CCC.foreground.set_88_256.lightsalmon1}{key}' \
                               f'{CCC.reset.all}: ' \
                               f'{CCC.foreground.set_88_256.orange1}{value: .5e}' \
                               f'{CCC.reset.all}' \
                               f'\n'

            return message

        # Make a copy of the dictionary to modify it
        msg_dict_copy = deepcopy(msg_dict)

        message = ''

        if msg_dict is None:
            return message

        # Find the length of the total epochs
        # get and remove the 'epoch' item from the dictionary
        # and reformat the string accordingly
        ep_len = int(np.ceil(np.log10(self._n_epochs)))
        epoch = msg_dict_copy.pop('epoch')
        message += f'\n'
        message += f'\t' * 1
        message += f'{CCC.foreground.set_88_256.green4}{SUC.heavy_teardrop_spoked_asterisk} '
        message += f'{CCC.foreground.set_88_256.chartreuse4}Epoch ' \
                   f'{CCC.foreground.set_88_256.green3}{epoch:{ep_len}d}' \
                   f'{CCC.foreground.set_88_256.grey}/' \
                   f'{CCC.foreground.set_88_256.darkgreen}{self._n_epochs}' \
                   f'{CCC.reset.all}'
        message += f'\n'

        # Check if we have training values for logging, starting with 'train_'
        # get the item from the dictionary and delete it
        # Generate the message set and add it to the total message
        train_items = {key[len('train_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
                       if key.startswith('train_')}

        message += _generate_message_set({"_title": "Training", **train_items})

        # Check if we have validation values for logging, starting with 'train_'
        # get the item from the dictionary and delete it
        # Generate the message set and add it to the total message
        val_items = {key[len('val_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
                     if key.startswith('val_')}

        message += _generate_message_set({"_title": "Validation", **val_items})

        # Print the rest of the message set
        message += _generate_message_set({"_title": "Others", **msg_dict_copy})

        return message

    def write(self, buf: str) -> None:
        """For tqdm to write to the buffer."""

        self.buf = buf.strip('\r\n\t ')

    def flush(self) -> None:
        """For tqdm.

        Will write tqdm messages as infos."""

        self._logger.info(self.buf)


class TheProgressBarLogger(Logger):
    """A logger for the progress bar that takes the control of stdout.

    WARNING: It does not exactly behave the same way as the Logger class. It is just a wrapper/manager
                class for the TheProgressBarLogger class.
    """
    # TODO: Conform the TheProgressBarLogger or a new class with Logger class

    def __init__(self, config: ConfigParser):
        """Initialize the logger and the TheProgressBar instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this callback instance and the data manager class.

        """

        # Making sure the logger is going to write to the console
        # Make sure it does not write any prefix
        config = config.update('console', True).update('format', '')

        super().__init__(config)

        # Create the instance of the TheProgressBar
        self._the_progress_bar = TheProgressBarColored()

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

        # Redirect the stream handler of the this logger to use TPB
        # self._handler.setStream(self._the_progress_bar)


        self.console_file.add_handler(self._the_progress_bar)

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

    def reset(self, total: int) -> TheProgressBarLogger:
        """Set the total number of iterations and resets the the_progress_bar.

        Parameters
        ----------
        total : int
            the total number of items expected.

        """

        self._the_progress_bar.reset()
        self._the_progress_bar.set_number_items(total)
        self._total = total

        return self

    def close(self) -> None:
        """Finishes and closes the TheProgressBar instance."""

        self._the_progress_bar.deactivate()

        # Redirect the stream handler of the this logger to use stdout
        self._handler.setStream(sys.stdout)

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

        message = self._generate_message(msg_dict)
        self._the_progress_bar.set_description(message)

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

        def _generate_message_set(msg_set_dict: Dict) -> str:
            """Function to help generate the set of messages for training, validation, ... .

            Parameters
            ----------
            msg_set_dict : dict
                The dictionary containing the information needed. Some are
                    _title: containing the title of the set

            Returns
            -------
            The generated string of the information set
            """

            message = ''

            if len(msg_set_dict) > 1:
                message += f'\t' * 2
                title = str(msg_set_dict.pop('_title'))
                message += f'{CCC.foreground.set_88_256.deepskyblue5}{SUC.right_facing_armenian_eternity_sign} '
                message += f'{CCC.foreground.set_88_256.deepskyblue3}{title}'
                message += f'{CCC.reset.all}'
                message += f'\n'
                for key, value in sorted(msg_set_dict.items()):
                    message += f'\t' * 3
                    message += f'{SUC.horizontal_bar} '
                    message += f'{CCC.foreground.set_88_256.lightsalmon1}{key}' \
                               f'{CCC.reset.all}: ' \
                               f'{CCC.foreground.set_88_256.orange1}{value: .5e}' \
                               f'{CCC.reset.all}' \
                               f'\n'

            return message

        # Make a copy of the dictionary to modify it
        msg_dict_copy = deepcopy(msg_dict)

        message = ''

        if msg_dict is None:
            return message

        # Find the length of the total epochs
        # get and remove the 'epoch' item from the dictionary
        # and reformat the string accordingly
        ep_len = int(np.ceil(np.log10(self._n_epochs)))
        epoch = msg_dict_copy.pop('epoch')
        message += f'\n'
        message += f'\t' * 1
        message += f'{CCC.foreground.set_88_256.green4}{SUC.heavy_teardrop_spoked_asterisk} '
        message += f'{CCC.foreground.set_88_256.chartreuse4}Epoch ' \
                   f'{CCC.foreground.set_88_256.green3}{epoch:{ep_len}d}' \
                   f'{CCC.foreground.set_88_256.grey27}/' \
                   f'{CCC.foreground.set_88_256.darkgreen}{self._n_epochs}' \
                   f'{CCC.reset.all}'
        message += f'\n'

        # Check if we have training values for logging, starting with 'train_'
        # get the item from the dictionary and delete it
        # Generate the message set and add it to the total message
        train_items = {key[len('train_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
                       if key.startswith('train_')}

        message += _generate_message_set({"_title": "Training", **train_items})

        # Check if we have validation values for logging, starting with 'train_'
        # get the item from the dictionary and delete it
        # Generate the message set and add it to the total message
        val_items = {key[len('val_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
                     if key.startswith('val_')}

        message += _generate_message_set({"_title": "Validation", **val_items})

        # Print the rest of the message set
        message += _generate_message_set({"_title": "Others", **msg_dict_copy})

        return message