from ..util.config import ConfigParser
from ..util.util import REGEX_INDENT_NEW_LINE_ONLY
from ..util.symbols_unicode import SYMBOL_UNICODE_CONFIG as SUC
import logging
from collections import OrderedDict
from typing import Union, Any, List
import threading
from datetime import datetime
import sys
import io
import colored
from readerwriterlock import rwlock


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

            try:
                return getattr(self.handler, item)
            except:
                return None


class Logger:
    """An abstract base/parent class for all logger classes."""

    # Keep track of how many logger instances we have
    _counter: List[int] = [0]
    _rwlock = rwlock.RWLockRead()

    # Keep track of supported logger functionality/abilities
    log_abilities = ['debug', 'report', 'info', 'warning', 'error']

    def __init__(self, config: ConfigParser = None):
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

        self._config = config if config is not None else ConfigParser()

        # The level at which we log
        self._level: int = self._get_logging_level(self._config.get_or_else('level', "info"))

        # Get the logger
        # we try to make a unique name for the logger so that we do not mistakenly grab another logger with the
        # same name, then, we set the name of the logger to the thing we want
        with self._rwlock.gen_wlock():
            count = self._counter[0]
            self._logger = logging.getLogger(str(count))
            self._logger_name = self._config.get_or_else('name', str(count))
            self._logger_name = self.set_name(self._logger_name)
            self._counter[0] += 1
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Suppress _logger output to stdout

        # Determine whether to write to the console and get the handlers
        self.console = self._config.get_or_else('console', True)
        if self.console is True:
            # Get the handler
            self.console_file: Union[LoggerConsoleFile, io.TextIOWrapper] =\
                self._config.get_or_else('console_handler', sys.stdout)
            console_handler = logging.StreamHandler(self.console_file)
            self._console_handler = self._set_handler_properties(console_handler, self._create_format("console"))
            self._logger.addHandler(self._console_handler)

        # Determine whether to write to file and get the handlers
        self.file = self._config.get_or_else('file', False)
        if self.file is True:
            file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name_prefix = self._config.get_option("filename_prefix").map(lambda x: f"{x}-").get_or_else("")
            file_name = f'{file_name_prefix}{file_name}.txt'
            file_handler = logging.FileHandler(self._config.get_or_else('file_name', file_name))
            self._file_handler = self._set_handler_properties(file_handler, self._create_format("file"))
            self._logger.addHandler(self._file_handler)

        prefix_len = 1 + len(self._logger_name) + 1  # 1 for arrow and 1 for the space after the name
        self._indenter = \
            lambda text, add_space: REGEX_INDENT_NEW_LINE_ONLY.sub(rf"\1{' ' * (prefix_len + add_space)}", text)

    def _set_handler_properties(self, handler: logging.Handler, format: str) -> logging.Handler:
        """
        Sets the properties of the handler and returns it

        Parameters
        ----------
        handler : logging.Handler
            handler to set the properties on

        format : str
            format to set for this handler

        Returns
        -------
        logging.Handler
            result

        """

        # Set the level, format, and attach
        handler.setLevel(self._level)
        handler.setFormatter(
            logging.Formatter(
                format
            ))

        return handler

    def set_name(self, name: str) -> str:
        """Changes the name of the current logger handler

        Parameters
        ----------
        name : str
            New name to be set

        Returns
        -------
        str
            final name of the logger

        """

        from ..communicator import mpi
        if mpi.mpi_communicator.is_distributed() is True:
            name_suffix = f' ' \
                           f'{colored.fg("dodger_blue_3")}at rank ' \
                           f'{colored.fg("dodger_blue_2")}{mpi.mpi_communicator.get_rank()}' \
                           f'{colored.fg("grey_50")}/' \
                           f'{colored.fg("dodger_blue_2")}{mpi.mpi_communicator.get_size()}' \
                           f'{colored.fg("grey_50")}-' \
                           f'{colored.fg("dodger_blue_2")}{mpi.mpi_communicator.get_node_rank()}' \
                           f'{colored.fg("grey_50")}-' \
                           f'{colored.fg("dodger_blue_2")}{mpi.mpi_communicator.get_local_rank()}' \
                           f'{colored.fg("grey_50")}/' \
                           f'{colored.fg("dodger_blue_2")}{mpi.mpi_communicator.get_local_size()}' \
                           f'{colored.attr("reset")}'
            name += name_suffix

        self._logger.name = name

        return name

    def get_abilities(self) -> List[str]:
        """Method to return the list of abilities for the logger.

        Returns
        -------
        A list of all the levels at which this logger can perform

        """

        return self.log_abilities

    def _create_format(self, handler_type: str) -> str:
        """
        Creates a custom logger format and returns the string.

        Parameters
        ----------
        handler_type : str
            Type of handler, it can be "console" or "file"

        Returns
        -------
        The string containing the customized logger format

        """

        format = ''

        if handler_type == "console":
            format += f'{colored.fg("dodger_blue_3")}{SUC.rightwards_arrow_to_bar} ' \
                      f'{colored.fg("gold_3b")}%(name)s' \
                      f'{colored.fg("grey_50")} ' \
                      f'{colored.attr("reset")}%(message)s'

        if handler_type == "file":
            format += '%(asctime)s ' \
                      '- ' \
                      '%(name)s' \
                      ' ' \
                      '%(message)s'

        return format

    def _get_logging_level(self, level: str) -> int:
        """
        Converts a string level to a logging package value.

        Parameters
        ----------
        level : str
            log level, look at the log_abilities to see them

        Returns
        -------
        int
            logging package value

        """

        level = level.lower()

        if level == "debug":
            return logging.DEBUG
        elif level == "report":
            return logging.INFO
        elif level == "info":
            return logging.INFO
        elif level == "warning":
            return logging.WARNING
        elif level == "error":
            return logging.ERROR
        else:
            raise ValueError(f"could not understand the log level of {level}, it has to be one of {self.log_abilities}")

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
        report_message = \
            f'{colored.fg("green_4")}'\
            f'report: '\
            f'{colored.attr("reset")}' \
            f'{msg}'

        # indent with 8 extra spaces to take care of 'report: '
        report_message = self._indenter(report_message, 8)

        self._logger.info(report_message)

    def info(self, msg: str) -> None:
        """Writes the message given as an info.

        Parameters
        ----------
        msg : str
            The message to be written as info

        """

        # Adds colored 'info: ' to the beginning of the message
        info_message = \
            f'{colored.fg("green_3a")}' \
            f'info: ' \
            f'{colored.attr("reset")}' \
            f'{msg}'

        # indent with 6 extra spaces to take care of 'info: '
        info_message = self._indenter(info_message, 6)

        self._logger.info(info_message)

    def warning(self, msg: str) -> None:
        """Writes the message given as a warning.

        Parameters
        ----------
        msg : str
            The message to be written as warning

        """

        # Adds colored 'warning: ' to the beginning of the message
        warning_message = \
            f'{colored.fg("red_1")}'\
            f'warning: '\
            f'{colored.attr("reset")}' \
            f'{msg}'

        self._logger.warning(warning_message)

    def error(self, msg: str) -> None:
        """Writes the message given as an error.

        Parameters
        ----------
        msg : str
            The message to be written as error

        """

        # Adds colored 'ERROR: ' to the beginning of the message and color the message as well
        error_message = \
            f'{colored.bg("red")}{colored.fg("white")}'\
            f'ERROR: ' \
            f'{msg}' \
            f'{colored.attr("reset")}'

        # indent with 7 extra spaces to take care of 'ERROR: '
        error_message = self._indenter(error_message, 7)

        self._logger.error(error_message)

    def debug(self, msg: str) -> None:
        """Writes the message given as an debug.

        Parameters
        ----------
        msg : str
            The message to be written as debug

        """

        # Adds colored 'debug: ' to the beginning of the message
        debug_message = \
            f'{colored.fg("indian_red_1b")}'\
            f'debug: '\
            f'{colored.attr("reset")}' \
            f'{msg}'

        # indent with 7 extra spaces to take care of 'debug: '
        debug_message = self._indenter(debug_message, 7)

        self._logger.debug(debug_message)
