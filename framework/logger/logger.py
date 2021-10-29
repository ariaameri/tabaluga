from __future__ import annotations
from framework.util.config import ConfigParser
from .log_hug import LogHug
from framework.util.console_colors import CONSOLE_COLORS_CONFIG as CCC
from framework.util.symbols_unicode import SYMBOL_UNICODE_CONFIG as SUC
from framework.base.base import BaseWorker, BaseEventManager
from .the_progress_bar import TheProgressBarColored
from abc import ABC
from typing import Dict, List, Union, Any
import logging
import sys
import numpy as np
import io
from datetime import datetime
from collections import OrderedDict
import threading
import signal
import re


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
        self._level: int = config.get_or_else('level', logging.INFO)

        # Get the logger
        self._logger = logging.getLogger(config.get_or_else('name', str(self._counter[0])))
        self._counter[0] += 1
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Suppress _logger output to stdout

        # TODO: Should we have logging to both the console and the file?
        # Determine whether to write to file or console and get the handlers
        self.console = config.get_or_else('console', False)
        if self.console is True:
            # Get the handler
            self.console_file: Union[LoggerConsoleFile, io.TextIOWrapper] =\
                config.get_or_else('console_handler', sys.stdout)
            self._handler = logging.StreamHandler(self.console_file)
        else:
            file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f'{file_name}.txt'
            self._handler = logging.FileHandler(config.get_or_else('file_name', file_name))

        # Set the level, format, and attach
        self._handler.setLevel(config.get_or_else('level', logging.INFO))
        self._format = config.get_or_else('format', self._create_format())
        self._handler.setFormatter(
            logging.Formatter(
                self._format
            ))
        self._logger.addHandler(self._handler)

    def set_name(self, name: str) -> None:
        """Changes the name of the current logger handler

        Parameters
        ----------
        name : str
            New name to be set

        """

        self._logger.name = name

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
        
        # Keep the stack of console (file) handlers to write to as an ordered dictionary
        self.handlers_stack: OrderedDict[Any, LoggerConsoleFile.ConsoleFile] = \
            OrderedDict({'sysout': self.ConsoleFile(sys.stdout)})

        # Keep track of the active console handler
        self.active_handler: Any = self.handlers_stack['sysout']

        # Keep track of whether or not this class is activated
        self.activated: bool = False

        # A lock to control printing to the stdout
        self.print_lock: threading.Lock = threading.Lock()

    def __enter__(self):

        self.activate()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.deactivate()

        return self

    def activate(self):
        """Activate the class to do the redirection of the stdout to this class and redirect stdout."""

        # Reroute stdout to this class to take care of all the printings
        sys.stdout = self

        # We are activated!
        self.activated = True

        return self

    def deactivate(self):
        """Deactivate the class and returns stdout to itself."""

        # Revert stdout
        sys.stdout = self.handlers_stack['sysout'].handler

        # And... deactivated!
        self.activated = False

        return self

    def find_active(self):
        """Find the active handler on top of the stack and puts in in the self.active_handler variable."""

        # Find the name of the name of the handler and if it is not paused,
        for index in range(len(self.handlers_stack.keys()) - 1, -1, -1):
            name = list(self.handlers_stack.keys())[index]
            item = self.handlers_stack[name]
            if item.ready() is True:
                self.active_handler = item
                return

    def register_handler(self, handler):
        """Add a console (file) handler to the stack.

        Parameters
        ----------
        handler
            A console handler

        """

        # Add the handler if it does not exist
        if handler not in self.handlers_stack.keys():
            self.handlers_stack[handler] = self.ConsoleFile(handler)

        # Update the new active handler
        self.find_active()

    def deregister_handler(self, handler):
        """Delete a console (file) handler from the stack.

        Parameters
        ----------
        handler
            A console handler

        """

        # Delete the handler if it exists
        if handler in self.handlers_stack.keys():
            del self.handlers_stack[handler]

        # Update the new active handler
        self.find_active()

    def pause_handler(self, handler):
        """Pause a console (file) handler on the stack.

        Parameters
        ----------
        handler
            A console handler

        """

        # Pause the handler, i.e. call pause() on it
        if handler in self.handlers_stack.keys():
            self.handlers_stack[handler].pause()

        # Update the new active handler
        self.find_active()

    def resume_handler(self, handler):
        """Resumes a possible paused console (file) handler on the stack.

        Parameters
        ----------
        handler
            A console handler

        """

        # Resume the handler, i.e. call resume() on it
        if handler in self.handlers_stack.keys():
            self.handlers_stack[handler].resume()

        # Update the new active handler
        self.find_active()

    def revert(self):
        """Reverts the current handler to the previous one."""

        # Revert only when it has more than 1 item in the stack
        # if len(self.handlers_stack) > 1:
        #     del self.handlers_stack[-1]

        if len(self.handlers_stack.keys()) > 1:
            del self.handlers_stack[list(self.handlers_stack.keys())[-1]]

        # Update the new active handler
        self.find_active()

    def __getattr__(self, item):
        """Look for the item in the item on top of the stack and then in sysout.stdout if could not find it."""

        # If not activate yet, just return the attribute of stdout
        if self.activated is False:
            return getattr(self.handlers_stack['sysout'], item)

        # Get the attribute from the active file
        attr = getattr(self.active_handler, item)

        # If the item could not be found, look in sysout finally
        return attr or getattr(self.handlers_stack['sysout'], item)

    class ConsoleFile:
        """Nested class as a thin wrapper to only hold the stdout handler and its specifications."""

        def __init__(self, handler):
            """Initializer for the class.

            Parameters
            ----------
            handler
                Handler to stdout

            """

            # Set the handler
            self.handler = handler

            # Flag to know whether or not we are paused
            self.paused = False

        def pause(self) -> None:
            """Method to pause using this handler."""

            self.paused = True

        def resume(self) -> None:
            """Method to resume using this handler."""

            self.paused = False

        def ready(self) -> bool:
            """Method to return whether or not this instance is ready to act as a console file or not.

            Returns
            -------
            A boolean indicating whether it is ready to act a console file.

            """

            ready = not self.paused

            return ready

        def __getattr__(self, item):

            attr = getattr(self.handler, item)

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
        # Update each and every logger config that we have within our config
        self.console_file = LoggerConsoleFile().activate()
        config = config.update({'_bc': {'$regex': r'\.\w+$'}}, {'$set': {'console_handler': self.console_file}})

        super().__init__(config)

    def on_os_signal(self, info: Dict = None):

        os_signal = info['signal']

        if os_signal == signal.SIGINT or os_signal == signal.SIGTERM:
            self.console_file.deactivate()


# class TQDMLogger(Logger, io.StringIO):
#
#     # TODO: Write the doc for the config argument
#
#     def __init__(self, config: ConfigParser):
#         """Initialize the tqdm logger instance.
#
#         Parameters
#         ----------
#         config : ConfigParser
#             The configuration needed for this callback instance and the data manager class.
#
#         """
#
#         # Making sure the logger is going to write to the console
#         # Make sure it does not write any prefix
#         config = config.update('console', True).update('format', '')
#
#         self._config = config
#
#         Logger.__init__(self, config)
#         io.StringIO.__init__(self)
#
#         self._tqdm = tqdm(
#             position=self._counter[0],
#             bar_format=''
#                        f'{CCC.foreground.set_88_256.grey74}'
#                        '{percentage:3.0f}% '
#                        f'{CCC.foreground.set_88_256.cornflowerblue}'
#                        '|'
#                        f'{CCC.foreground.set_88_256.steelblue4}'
#                        '{bar}'
#                        f'{CCC.foreground.set_88_256.cornflowerblue}'
#                        '| '
#                        f'{CCC.foreground.set_88_256.gold1}'
#                        '{n_fmt}'
#                        f'{CCC.foreground.set_88_256.grey}'
#                        '/'
#                        f'{CCC.foreground.set_88_256.orange2}'
#                        '{total_fmt} '
#                        f'{CCC.foreground.set_88_256.grey50}'
#                        '[{elapsed}<{remaining}, ' '{rate_fmt}{postfix}] '
#                        f'{CCC.reset.all}'
#                        '{desc}',
#             file=self  # Write to this log handler instead of stderr
#         )
#         # Bookkeeping for tqdm
#         self.buf: str = ''
#
#         # The number of total iterations
#         self._total: int = -1
#
#         self._n_epochs: int = -1
#
#     def set_number_epochs(self, epochs: int):
#         """Sets the total number of epochs.
#
#         Parameters
#         ----------
#         epochs : int
#             Number of epochs
#
#         """
#
#         # The number of total epochs
#         self._n_epochs: int = epochs
#
#     def reset(self, total: int) -> None:
#         """Set the total number of iterations and resets the tqdm.
#
#         Parameters
#         ----------
#         total : int
#             the total number of iterations expected."""
#
#         self._tqdm.reset(total=total)
#         self._total = total
#
#     def close(self) -> None:
#         """Finishes and closes the tqdm instance."""
#
#         self._tqdm.close()
#
#     def update(self, update_count: int, msg_dict: Dict = None) -> None:
#         """Update the tqdm progress bar with description set to message.
#
#         Parameters
#         ----------
#         update_count : int
#             The amount that should be added to the tqdm instance.
#         msg_dict : Dict, optional
#             Contains the dictionary message to be set as the progress bar description.
#             Will be passed to the _generate_message method, read there for more info.
#         """
#
#         self._tqdm.update(update_count)
#
#         message = self._generate_message(msg_dict)
#         self._tqdm.set_description_str(message)
#
#     def _generate_message(self, msg_dict: Dict) -> str:
#         """Generates a string based on the input to be used as tqdm bar description.
#
#         If msg is None, empty string will be returned.
#
#         Parameters
#         ---------
#         msg_dict : Dict
#             Dictionary containing the information to be used. Contains:
#                 epoch: int
#                 loss: float
#                 val_loss: float, optional
#         """
#
#         def _generate_message_set(msg_set_dict: Dict) -> str:
#             """Function to help generate the set of messages for training, validation, ... .
#
#             Parameters
#             ----------
#             msg_set_dict : dict
#                 The dictionary containing the information needed. Some are
#                     _title: containing the title of the set
#
#             Returns
#             -------
#             The generated string of the information set
#             """
#
#             message = ''
#
#             if len(msg_set_dict) > 1:
#                 message += f'\t' * 2
#                 title = str(msg_set_dict.pop('_title'))
#                 message += f'{CCC.foreground.set_88_256.deepskyblue5}{SUC.right_facing_armenian_eternity_sign} '
#                 message += f'{CCC.foreground.set_88_256.deepskyblue3}{title}'
#                 message += f'{CCC.reset.all}'
#                 message += f'\n'
#                 for key, value in sorted(msg_set_dict.items()):
#                     message += f'\t' * 3
#                     message += f'{SUC.horizontal_bar} '
#                     message += f'{CCC.foreground.set_88_256.lightsalmon1}{key}' \
#                                f'{CCC.reset.all}: ' \
#                                f'{CCC.foreground.set_88_256.orange1}{value: .5e}' \
#                                f'{CCC.reset.all}' \
#                                f'\n'
#
#             return message
#
#         # Make a copy of the dictionary to modify it
#         msg_dict_copy = {**msg_dict}
#
#         message = ''
#
#         if msg_dict is None:
#             return message
#
#         # Find the length of the total epochs
#         # get and remove the 'epoch' item from the dictionary
#         # and reformat the string accordingly
#         ep_len = int(np.ceil(np.log10(self._n_epochs)))
#         epoch = msg_dict_copy.pop('epoch')
#         message += f'\n'
#         message += f'\t' * 1
#         message += f'{CCC.foreground.set_88_256.green4}{SUC.heavy_teardrop_spoked_asterisk} '
#         message += f'{CCC.foreground.set_88_256.chartreuse4}Epoch ' \
#                    f'{CCC.foreground.set_88_256.green3}{epoch:{ep_len}d}' \
#                    f'{CCC.foreground.set_88_256.grey}/' \
#                    f'{CCC.foreground.set_88_256.darkgreen}{self._n_epochs}' \
#                    f'{CCC.reset.all}'
#         message += f'\n'
#
#         # Check if we have training values for logging, starting with 'train_'
#         # get the item from the dictionary and delete it
#         # Generate the message set and add it to the total message
#         train_items = {key[len('train_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
#                        if key.startswith('train_')}
#
#         message += _generate_message_set({"_title": "Training", **train_items})
#
#         # Check if we have validation values for logging, starting with 'train_'
#         # get the item from the dictionary and delete it
#         # Generate the message set and add it to the total message
#         val_items = {key[len('val_'):]: msg_dict_copy.pop(key) for key, value in msg_dict.items()
#                      if key.startswith('val_')}
#
#         message += _generate_message_set({"_title": "Validation", **val_items})
#
#         # Print the rest of the message set
#         message += _generate_message_set({"_title": "Others", **msg_dict_copy})
#
#         return message
#
#     def write(self, buf: str) -> None:
#         """For tqdm to write to the buffer."""
#
#         self.buf = buf.strip('\r\n\t ')
#
#     def flush(self) -> None:
#         """For tqdm.
#
#         Will write tqdm messages as infos."""
#
#         self._logger.info(self.buf)


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
        title = f'{CCC.foreground.set_88_256.green4}{SUC.heavy_teardrop_spoked_asterisk} '
        title += f'{CCC.foreground.set_88_256.chartreuse4}Epoch ' \
                 f'{CCC.foreground.set_88_256.green3}{epoch:{ep_len}d}' \
                 f'{CCC.foreground.set_88_256.grey27}/' \
                 f'{CCC.foreground.set_88_256.darkgreen}{self._n_epochs}' \
                 f'{CCC.reset.all}'

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
                f'{CCC.foreground.set_88_256.green4}{SUC.heavy_teardrop_spoked_asterisk} '
        title += f'{CCC.foreground.set_88_256.chartreuse4}Epoch ' \
                 f'{CCC.foreground.set_88_256.green3}{epoch:{ep_len}d}' \
                 f'{CCC.foreground.set_88_256.grey27}/' \
                 f'{CCC.foreground.set_88_256.darkgreen}{self._n_epochs}' \
                 f'{CCC.reset.all}'

        # If there is a stat entry, use it
        if msg_dict.get('stat') is not None:
            message = msg_dict.get('stat').str_representation_with_title(title=title)
        else:
            message = title

        # Indent the messages once
        message = re.sub(r'(^|\n)', r'\1\t', message)

        return message
