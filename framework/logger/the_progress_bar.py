from __future__ import annotations
import math
import multiprocessing
import pathlib
import select
import subprocess
import threading
import sys
from typing import List, Optional, Any, Callable
from abc import abstractmethod, ABC
import kombu
import numpy as np
import time
import datetime
import os
import colored
from ..util.data_muncher import DataMuncher
from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.calculation import Calculation
from ..util.option import Some
from ..communicator import mpi
from ..communicator import rabbitmq
import re
import fcntl
import termios
from enum import Enum

# message template to use
# keep in mind that this has to be exactly the same as the default values set in the classes below
gather_info_data_msg_template: DataMuncher = DataMuncher({
    'rank': 0,
    'progress_bar': {
        'prefix': {
            "state": {
                "item":
                    {
                        "current_item_index": 0,
                        "total_items_count": -np.inf,
                    }
                }
            },
        'bar': {
            "percentage": 0.01,
            "bar_chars": '',
        },
        'suffix': {
            "state": {
                    "item": {
                        "current_item_index": 0,
                        "total_items_count": -np.inf,
                    },
                },
            "statistics": {
                    "average": {
                        "average_item_per_update": -np.inf,
                        "average_time_per_update": -np.inf,
                    },
                    "time": {
                        "last_update_time": -np.inf,
                        "initial_progress_bar_time": -np.inf,
                    }
                }
        },
        'description': {
            'full': {
                'before': '',
                'after': '',
            },
            'short': {
                'before': '',
                'after': '',
            },
        },
    },
    'current_iteration_index': 0,
    'aggregation_data': None,
})

# variable to hold rabbit data
rabbit_data: Optional[ConfigParser] = None

# global variables definitions
REGEX_REMOVE_NONPRINT_CHARS = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
REGEX_INDENTATION = re.compile(r'(^|\n)')
ENV_VARS = ConfigParser({
    'tty_size': 'TTY_SIZE',
    'force_tty': 'FORCE_TTY',
})


# function to initialize rabbit data
# this is because mpi_communicator is not ready at the time of importing this module and will become ready later
def make_rabbit_data():

    global rabbit_data

    # if already defined, skip
    if rabbit_data is not None:
        return

    # set the variable
    rabbit_data = ConfigParser({
        'gather_info': {
            'exchange': {
                'name': 'tpb_data_gather',
                'type': rabbitmq.RabbitMQExchangeType.DIRECT,
                'durable': False,
                'auto_delete': True,
                'delivery_mode': 1
            },
            'queue': {
                'name': f'tpb_data_gatherer_info_{mpi.mpi_communicator.get_rank()}',
                'name_template': f'tpb_data_gatherer_info_<rank>',
                # exchange is in '..exchange'
                # routing key is in '..message'
                "max_length": 1,  # we only want the latest update
                'durable': False,
                'auto_delete': True,
            },
            'message': {
                'routing_key': f'data.update_info.{mpi.mpi_communicator.get_rank()}',
            },
            'consumer': {
                'name': 'tpb_data_gather_consumer',
            },
        },
        'printer_gatherer': {
            'exchange': {
                'name': 'tpb_printer',
                'type': rabbitmq.RabbitMQExchangeType.DIRECT,
                'durable': False,
                'auto_delete': True,
                'delivery_mode': 1
            },
            'queue': {
                'name': f'tpb_printer',
                # exchange is in '..exchange'
                # routing key is in '..message'
                'durable': False,
                'auto_delete': True,
            },
            'message': {
                'routing_key': f'data.printer.manager',
            },
            'consumer': {
                'name': 'tpb_printer_consumer',
            },
        },
    })


class TheProgressBarBase(ABC, BaseWorker):

    class Modes(Enum):
        """Enum for modes of operation"""

        NORMAL = 1
        NOTTY = 2

    class Roles(Enum):
        """Enum for type of roles"""

        SINGLE = 1
        MANAGER = 2  # Whether we are in master mode, meaning we are responsible for printing the progress bar
        WORKER = 3

    def __init__(self, stdout_handler=None, config: ConfigParser = None):
        """Initializes the instance."""

        super().__init__(config)

        # set rabbit data
        make_rabbit_data()

        # Create a lock so that one thread at a time can use the console to write
        self.print_lock = threading.Lock()

        # Create a threading.Event for knowing when to write the progress bar
        self.event_print = threading.Event()

        # A daemon thread placeholder for running the update of the progress bar and other stuff
        # Also, a daemon thread placeholder for when on pause
        initial_run_threads_info = {
            'print': {
                'main': None,  # daemon thread placeholder for running the update of the progress bar
                'resume': None,  # daemon thread placeholder for when on pause
            },
            'gather_info': {
                'main': None,
            }
        }
        self.run_thread_info = DataMuncher(initial_run_threads_info)

        # Get a CursorModifier that contains ANSI escape codes for the cursor
        self.cursor_modifier = self.CursorModifier()

        # Console controlling book keeping
        self.original_sysout = sys.__stdout__
        self._isatty_original = self._make_isatty()

        # Keep the stdout handler to write to
        self.stdout_handler = stdout_handler or self.original_sysout

        # keep the actions that we should take based on the configurations
        initial_actions = {
            'get_bar': self._get_progress_bar_with_spaces,
            'aggregator': {  # this is used to aggregate the given user data
                'full': lambda x: '',
                'short': lambda x: '',
            }
        }
        self.actions = DataMuncher(initial_actions)

        # keep info regarding the sleep times
        r, w = multiprocessing.Pipe(duplex=False)  # create a channel to talk to the sleeper
        initial_sleep_timer_info = {
            # channel to talk to the timer
            'pipe': {
                'read': r,
                'write': w,
            },
            # the amount of update we should see before we do an update
            'update_interval': self._config.get_or_else('sleep_timer.update_interval', -1),
            # the number of times we should do an update in an iteration
            'update_number_per_iteration': self._config.get_or_else('sleep_timer.update_number_per_iteration', 4),
            'stat': {
                'last_item_index': -np.inf,
            }
        }
        self.sleep_timer_info = DataMuncher(initial_sleep_timer_info)

        # Book keeping for the information regarding the progress bar
        initial_progress_bar_info = {
            'progress_bar': {  # Everything related to the string of the progress bar
                'prefix': '',
                'bar': '',  # The bar itself
                'suffix': '',
                'description': {
                    'full': {
                        'before': '',
                        'after': '',
                    },
                    'short': {
                        'before': '',
                        'after': '',
                    },
                },
                'progress_bar': '',  # The whole progress bar, consisting of the previous fields
            },
            'console': {  # Information regarding the console
                'rows': -1,
                'columns': -1
            },
            'aggregation_data': {
                'aggregation_data': None,
            }
        }
        self.progress_bar_info = DataMuncher(initial_progress_bar_info)

        # Book keeping for the information regarding the collected statistics
        initial_statistics_info = {
            'time': {
                'initial_run_time': -np.inf,  # The time when this instance is first activated
                'initial_progress_bar_time': -np.inf,  # The time when this instance is activated or reset,
                                                       # i.e. the time when this current, specific progress bar started
                'last_update_time': -np.inf  # The time when the latest update to this instance happened
            },
            'average': {
                'average_time_per_update': -np.inf,
                'average_item_per_update': -np.inf
            }
        }
        self.statistics_info = DataMuncher(initial_statistics_info)

        # Book keeping for the state of the instance
        initial_state_info = {
            'activated': False,  # Whether the instance has been activated---it can be activated only once
            'paused': False,  # Whether we are on pause mode
            'mode': self.Modes.NORMAL,
            'role': self.Roles.SINGLE,
            'parallel': {
                'rank': mpi.mpi_communicator.get_rank(),
                'size': mpi.mpi_communicator.get_size(),
                'is_distributed': mpi.mpi_communicator.is_distributed(),
                'is_main_rank': mpi.mpi_communicator.is_main_rank(),
            },
            # Whether we should write to some external stdout handler or take care of it ourselves
            'external_stdout_handler': True if stdout_handler is not None else False,
            'item': {
                'total_items_count': -np.inf,  # Count of total number of batches expected
                'current_item_index': 0,  # Current batch item/index/number
                'current_iteration_index': 0  # the current iteration index or how many times we have reset
            }
        }
        self.state_info = DataMuncher(initial_state_info)

        # book keeping for communication relate data
        initial_communication_info = {
            'gather_info': {
                'exchange': {
                    'exchange': None,
                },
                'queue': {
                    'queue': None,
                },
                'consumer': {
                    'consumer': None,  # to be used by the manager
                }
            },
            'printer_gatherer': {
                'exchange': {
                    'exchange': None,
                },
                'queue': {
                    'queue': None,
                },
                'consumer': {
                    'consumer': None,  # to be used by the manager
                }
            },
        }
        self.communication_info = DataMuncher(initial_communication_info)

        # initialize the communication related tasks
        self._init_communication()

        # A buffer for the messages to be printed
        self.buffer: List = []

        # Set of characters to be used for filling the bar
        self.bar_chars: str = '▏▎▍▌▋▊▉█'

    def __enter__(self) -> TheProgressBarBase:

        return self.activate()

    def __exit__(self, exc_type, exc_val, exc_tb):

        return self.deactivate()

    def __del__(self):
        """Called when being deleted."""

        self.deactivate()

    class CursorModifier:

        def __init__(self):

            # Dictionary of the cursor modifying escape codes
            self.cursor_dict = {
                "hide": '\033[?25l',
                "show": '\033[?25h',
                "clear_line": '\033[2K\r',
                "clear_until_end": '\033[0J',
                "up": ['\033[', 'A'],
                "down": ['\033[', 'B'],
                "right": ['\033[', 'C'],
                "left": ['\033[', 'D'],
                "begin_next_line": ['\033[', 'E'],
                "begin_previous_line": ['\033[', 'F'],
                "save_location": '\033[s',
                "restore_location": '\033[u'
            }

        def get(self, item: str, add=''):
            """Get a cursor modifying string from the this class and add 'add' if possible.

            Parameters
            ----------
            item : str
                The description of the cursor modifier to be returned
            add : Union[str, int], optional
                String to be added in the middle of the modifier, if possible

            Returns
            -------
            ANSI escape sequence corresponding to the modifier demanded

            """

            if not sys.stdout.isatty():
                return ''

            esc_sequence = self.cursor_dict.get(item, '')

            if type(esc_sequence) is list:
                esc_sequence = str(add).join(esc_sequence)

            return esc_sequence

    # action methods

    def activate(self) -> TheProgressBarBase:
        """Activates the progress bar: redirected stdout to this class and prints the progress bar

        Returns
        -------
        This instance

        """

        # If the instance is already activated, skip
        if self.state_info.get('activated') is True:
            return self

        # Update the state to know we are activated
        self.state_info = self.state_info.update({}, {'activated': True})

        # set the mode
        self._configure_mode()

        # set the role
        self._configure_role()

        # set the actions
        self._set_actions()

        # first, reset so that everything is set
        self.reset()

        # Set the initial time
        current_time = time.time()
        self.statistics_info = \
            self.statistics_info \
                .update(
                    {'_bc': '.time'},
                    {'initial_run_time': current_time, 'initial_progress_bar_time': current_time}
                )

        # set and run the running daemon thread
        self.run()

        # Redirect stdout just in case there is no stdout handler from outside
        if self.state_info.get('external_stdout_handler') is False:
            sys.stdout = self
        else:
            self._activate_external_stdout_handler()

        # If not in single mode, no need to print, thus return now
        if self._check_action() is False:
            return self

        # Hide cursor
        self._direct_write(self.cursor_modifier.get("hide"))

        # Set the printing event on to start with the printing of the progress bar
        self.event_print.set()

        return self

    def deactivate(self) -> None:
        """Deactivates the progress bar: redirected stdout to itself and closes the progress bar"""

        # Stop the run thread
        self.run_thread_info = self.run_thread_info.update({}, {'print.main': None})

        # Revert stdout back to its original place
        if self.state_info.get('external_stdout_handler') is False:
            sys.stdout = self.original_sysout
        else:
            self._deactivate_external_stdout_handler()

        # if we are not printing at all, skip
        if self._check_action() is False:
            return

        # Show cursor
        self._direct_write(self.cursor_modifier.get("show"))

        # Print the progress bar and leave it
        self._print_progress_bar(return_to_line_number=-1)

    def pause(self, pause_print_control: bool = None) -> TheProgressBarBase:
        """Pauses the progress bar: redirected stdout to itself and stops prints the progress bar.

        This method permanently paused the instance. To resume run either the `resume` or `_look_to_resume` method.

        Parameters
        ----------
        pause_print_control : bool, optional
            whether to let go the control of printing or not; if set to None, will be automatic

        Returns
        -------
        This instance

        """

        # if pause_print_control is None, set based on whether we are in distributed mode
        if pause_print_control is None:
            pause_print_control = False if self.state_info.get('parallel.is_distributed') is True else True

        # If the instance is already paused, skip
        if self.state_info.get('paused') is True:
            return self

        # Update the state to know we are paused
        self.state_info = self.state_info.update({}, {'paused': True})

        # Revert stdout back to its original place
        # do this only if necessary
        if pause_print_control is True:
            if self.state_info.get('external_stdout_handler') is False:
                sys.stdout = self.original_sysout
            else:
                self._pause_external_stdout_handler()

        # if we are not printing at all, skip
        if self._check_action() is False:
            return self

        # Show cursor
        sys.stdout.write(self.cursor_modifier.get("show") + self.cursor_modifier.get("clear_until_end"))
        sys.stdout.flush()

        # Pause printing
        self.event_print.clear()

        return self

    def resume(self) -> TheProgressBarBase:
        """Resumes the progress bar: redirected stdout to this instance and starts printing the progress bar.

        Returns
        -------
        This instance

        """

        # If the instance is not paused, skip
        if self.state_info.get('paused') is False:
            return self

        # Update the state to know we are not paused
        self.state_info = self.state_info.update({}, {'paused': False})

        # Revert stdout back to its original place
        if self.state_info.get('external_stdout_handler') is False:
            sys.stdout = self
        else:
            self._resume_external_stdout_handler()

        # if we are not printing at all, skip
        if self._check_action() is False:
            return self

        # Hide cursor
        self._direct_write(self.cursor_modifier.get("hide"))

        # No longer look if we should resume
        self.check_for_resume_thread = None

        # Start printing
        self.event_print.set()

        # notify the sleeper as well!
        self._notify_sleep()

        return self

    def update(self, count: int) -> None:
        """Updates the progress bar by adding 'count' to the number of items.

        Parameter
        ---------
        count : int
            The count at which the progress items should be increased. It has to be non-negative

        """

        if count > 0:
            # Update current item
            self.state_info = self.state_info.update({'_bc': {'$regex': 'item$'}},
                                                     {'$inc': {'current_item_index': count}})

            # Keep track of an average number of elements in each update
            self.statistics_info = \
                self.statistics_info.update(
                    {'_bc': {'$regex': 'average$'}},
                    {'average_item_per_update':
                        Calculation.exp_average(
                             item=self.statistics_info.get('average.average_item_per_update'),
                             d_item=count
                         )
                    }
                )

            # Update the time
            self._update_time_counter()

            # notify the sleeping timer
            self._notify_sleep()

    def reset(self, return_to_line_number: int = -1) -> TheProgressBarBase:
        """Resets the progress bar and returns its instance.

        Parameters
        ----------
        return_to_line_number: int, optional
            The line number to return to

        Returns
        -------
        This instance

        """

        # reset the communication tasks
        self._reset_communication()

        # Print the progress bar and leave it if we have done any progress
        if self.state_info.get('item.current_item_index') != 0:
            self._print_progress_bar(return_to_line_number=return_to_line_number)

        # do the rest of resetting
        # increment the iteration index
        self.state_info = \
            self.state_info \
                .update(
                    {'_bc': {'$regex': 'item$'}},
                    {'$inc': {'current_iteration_index': 1}}
                )

        # Set the initial time
        current_time = time.time()
        self.statistics_info = \
            self.statistics_info \
                .update(
                    {'_bc': {'$regex': 'time$'}},
                    {'initial_progress_bar_time': current_time, 'last_update_time': current_time}
                )

        # Reset the current item counter
        self.state_info = self.state_info.update(
            {'_bc': {'$regex': 'item$'}},
            {'$set': {'current_item_index': 0, 'total_items_count': -np.inf}}
        )

        # Reset the sleep info
        self.sleep_timer_info = self.sleep_timer_info.update(
            {},
            {'$set': {'stat.last_item_index': -np.inf}},
        )

        # Reset the descriptions
        self.set_description_after('')
        self.set_description_before('')
        self.set_description_short_after('')
        self.set_description_short_before('')

        return self

    @abstractmethod
    def run(self):
        """Method to run the threads needed, unblockingly."""

        raise NotImplementedError

    def run_print(self) -> None:
        """Prints the progress bar and takes care of other controls.

        Method to be run by the daemon progress bar thread.
        """

        while self.run_thread_info.get('print.main'):

            # Wait for the event to write
            self.event_print.wait()
            # Print the progress bar only if it is focused on
            # If we are not focused, pause
            if self._check_if_should_print():
                self._print_progress_bar()
            else:
                self.pause()
                self._look_to_resume()  # Constantly check if we can resume

            self._sleep()

    def _make_run_thread(self) -> None:
        """Makes and sets the run thread"""

        # now, create the thread object
        run_print_thread = threading.Thread(
            name='run_daemon_thread',
            target=self.run_print,
            args=(),
            daemon=True
        )
        self.run_thread_info = self.run_thread_info.update({}, {'print.main': run_print_thread})

    @abstractmethod
    def run_gather_info(self) -> None:
        """Publish the information to the broker for communication."""

        raise NotImplementedError

    def _make_gather_info_thread(self) -> None:
        """Makes and sets the gather info thread"""

        # now, create the thread object
        gather_info_thread = threading.Thread(
            name='gather_info_daemon_thread',
            target=self.run_gather_info,
            args=(),
            daemon=True
        )
        self.run_thread_info = self.run_thread_info.update({}, {'gather_info.main': gather_info_thread})

    def set_number_items(self, number_of_items: int) -> TheProgressBarBase:
        """Set the total number of the items.

        Parameters
        ----------
        number_of_items : int
            The total number of items

        Returns
        -------
        This instance

        """

        self.state_info = self.state_info.update({'_bc': {'$regex': 'item$'}}, {'total_items_count': number_of_items})

        return self

    def _configure_mode(self):
        """Method to find out and set the mode of operation."""

        # if we do not have a tty, operate in NOTTY mode
        if not self._check_if_atty():
            self.state_info = self.state_info.update({}, {'mode': self.Modes.NOTTY})

    @abstractmethod
    def _configure_role(self):
        """Method to find out and set the role."""

        raise NotImplementedError

    @abstractmethod
    def _set_actions(self):
        """Sets the proper actions based on the conditions."""

        raise NotImplementedError

    def _check_action(self) -> bool:
        """Method to return conditional on whether we should perform the action methods."""

        return True

    # terminal related methods

    def _make_and_get_terminal_size(self, data: DataMuncher = None) -> (int, int):
        """
        Returns the size of the terminal in form of (columns, rows).

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    "rows": ,
                    "columns": ,
                }

        Returns
        -------
        (int, int)
            terminal size in form of (columns, rows)
        """

        if data is not None:
            return data.get('columns'), data.get('rows')

        if self._check_if_atty() is False:
            return -1, -1

        out = os.get_terminal_size()
        return out.columns, out.lines

        # env = os.environ
        #
        # def ioctl_GWINSZ(fd):
        #     try:
        #         cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        #     except:
        #         return
        #     return cr
        #
        # cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        # if not cr:
        #     try:
        #         fd = os.open(os.ctermid(), os.O_RDONLY)
        #         cr = ioctl_GWINSZ(fd)
        #         os.close(fd)
        #     except:
        #         pass
        # if not cr:
        #     cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
        #
        #     ### Use get(key[, default]) instead of a try/catch
        #     # try:
        #     #    cr = (env['LINES'], env['COLUMNS'])
        #     # except:
        #     #    cr = (25, 80)
        # return int(cr[1]), int(cr[0])

    def _get_terminal_size(self) -> (int, int):
        """Returns the stored size of the terminal in form of (columns, rows)."""

        return self.progress_bar_info.get('console.columns'), self.progress_bar_info.get('console.rows')

    def _update_terminal_size(self, terminal_size: (int, int) = None) -> None:
        """Updates the stored data for the terminal size.

        Parameters
        ----------
        terminal_size: (int, int), optional
            Optional terminal size to update the internal knowledge of terminal size. If not given, will be inferred
                from the actual terminal.

        """

        if terminal_size is None:
            columns, rows = self._make_and_get_terminal_size()
        else:
            columns, rows = terminal_size

        # Retrieve and store
        self.progress_bar_info = \
            self.progress_bar_info.update(
                {'_bc': {'$regex': 'console$'}},
                {'rows': rows, 'columns': columns}
            )

    def _run_check_for_resume(self) -> None:
        """Checks to see if we are in focus to resume the printing."""

        # Check until we are in focus
        while self._check_if_focused() is False:

            time.sleep(1 / self._get_update_frequency())

        # If we are in focus, resume
        self.resume()

    def _check_if_should_print(self) -> bool:
        """
        Checks whether we should print.

        Returns
        -------
        bool

        """

        # if we are paused, we should not print!
        if self.state_info.get('paused') is True:
            return False

        mode = self.state_info.get('mode')

        if mode == self.Modes.NORMAL and self._check_if_foreground():
            return True
        elif mode == self.Modes.NOTTY:
            return True

        return False

    def _check_if_should_append_progress_bar(self) -> bool:
        """
        Checks whether we should append the progress bar at the end of each external print request.

        Returns
        -------
        bool

        """

        # if we should print
        check = self._check_if_should_print()

        # only add when in normal mode
        check &= True if self.state_info.get('mode') == self.Modes.NORMAL else False

        return check

    def _check_if_focused(self) -> bool:
        """Checks whether the terminal is focused on the progress bar so that it should be printed.

        Returns
        -------
        A boolean stating whether or not the progress bar should be printed

        """

        # Check if we are connected to a terminal
        # Check if we are a foreground process
        check = self._check_if_atty() \
            and self._check_if_foreground()

        return check

    @abstractmethod
    def _make_isatty(self) -> Callable:
        """
        Makes the new atty function.

        Returns
        -------
        Callable
            the function to be called whose output defines whether or not we have a tty.
        """

        raise NotImplementedError

    def _check_if_atty(self) -> bool:
        """
        Checks whether the terminal has a tty.

        Returns
        -------
        bool

        """

        try:
            return self._isatty_original()
        except:
            return False

    def isatty(self) -> bool:
        """
        Checks whether the terminal has a tty.

        Returns
        -------
        bool

        """

        return self._check_if_atty()

    def _check_if_foreground(self) -> bool:
        """
        Checks if we are running in foreground

        Returns
        -------
        bool

        """

        # return True
        # import subprocess
        # out = subprocess.check_output(['ps', '-o', 'stat', '-p', str(os.getpid())])
        # return True if '+' in out.decode('utf-8') else False

        try:
            return os.getpgrp() == os.tcgetpgrp(self.original_sysout.fileno())
        except:
            return False

    # external stdout handler methods

    def _activate_external_stdout_handler(self):
        """Method to activate the external stdout handler in case one is passed in the constructor."""

        # Register self to the external console file
        self.stdout_handler.register_handler(self)

    def _pause_external_stdout_handler(self):
        """Method to pause the external stdout handler in case one is passed in the constructor."""

        # Pause the external console file
        self.stdout_handler.pause_handler(self)

    def _resume_external_stdout_handler(self):
        """Method to resume the external stdout handler."""

        # Resume the external console file
        self.stdout_handler.resume_handler(self)

    def _deactivate_external_stdout_handler(self):
        """Method to deactivate the external stdout handler in case one is passed in the constructor."""

        # De-register self from the external console file
        self.stdout_handler.deregister_handler(self)

    # progress bar methods

    def get_progress_bar_string(self, terminal_size: (int, int) = None, return_to_line_number: int = 0) -> str:
        """Returns the progress bar along with its cursor modifier ANSI escape codes

        Parameters
        -------
        terminal_size: (int, int), optional
            User-defined terminal size so that the method behaves according to this size, mainly used in non-master mode
        return_to_line_number: int, optional
            The number of the line to return to the beginning of, after the progress bar

        Returns
        -------
        String containing the progress bar

        """

        # Return only the string and not the '\n' at the end if we should not return to the beginning
        result = \
            self._get_progress_bar_with_spaces(
                return_to_line_number=return_to_line_number
        )
        result = result[:-1] if return_to_line_number == -1 else result

        return result

    def _get_progress_bar_with_spaces(
            self,
            data: DataMuncher = DataMuncher(),
            return_to_line_number: int = 0,
            include_desc_before: bool = True,
            include_desc_short_before: bool = False,
            include_prefix: bool = True,
            include_bar: bool = True,
            include_suffix: bool = True,
            include_desc_after: bool = True,
            include_desc_short_after: bool = False,
    ) -> str:
        """Returns the progress bar along with its cursor modifier ANSI escape codes

        Returns
        -------
        terminal_size: (int, int), optional
            User-defined terminal size so that the method behaves according to this size, mainly used in non-master mode
        return_to_line_number: int, optional
            The number of the line to return to the beginning of, after the progress bar

        """

        # Get the progress bar
        progress_bar = \
            self._make_and_get_progress_bar(
                data=data,
                include_desc_before=include_desc_before,
                include_desc_short_before=include_desc_short_before,
                include_prefix=include_prefix,
                include_bar=include_bar,
                include_suffix=include_suffix,
                include_desc_after=include_desc_after,
                include_desc_short_after=include_desc_short_after,
            )

        # print(f"got {return_to_line_number} this time")

        # if terminal size given is negative, just return the progress_bar
        if data.get_option('terminal').filter(lambda x: x.get('columns') <= 0 or x.get('rows') <= 0).is_defined():
            return progress_bar + '\n'

        # Clear the line and write it
        # progress_bar_with_space: str = self.cursor_modifier.get('clear_line')
        progress_bar_with_space: str = self.cursor_modifier.get('clear_until_end')
        progress_bar_with_space += f'{progress_bar}'

        # Get the number of lines

        # Get the progress bar without special characters
        progress_bar_with_space_without_special_chars = self._remove_non_printing_chars(progress_bar_with_space)
        # Get the terminal size
        console_columns, _ = self._make_and_get_terminal_size(data.get_or_else('terminal', None))
        # Figure how many lines will be wrapped to the next
        number_of_lines = \
            sum(
                (len(item.expandtabs()) - 1) // console_columns
                for item
                in progress_bar_with_space_without_special_chars.split('\n')
                if item != ''
            )
        # Figure how many lines we have
        number_of_lines += progress_bar.count(f'\n')

        # Recalculate the return_to_line_number in case of negative numbers
        return_to_line_number = \
            return_to_line_number \
            if return_to_line_number >= 0 \
            else number_of_lines + 1 + 1 + return_to_line_number

        # The total number of lines we need to go up or return
        return_line_count = number_of_lines - return_to_line_number

        # Compensate for the lines to be printed and go back to the beginning of all of them
        progress_bar_with_space += \
            self.cursor_modifier.get('up', return_line_count) if return_line_count > 0 else ''
        progress_bar_with_space += \
            self.cursor_modifier.get('down', -1 * return_line_count) if return_line_count < 0 else ''
        progress_bar_with_space += f'\r'
        progress_bar_with_space += f'\n\b' if return_line_count == -1 else ''

        return progress_bar_with_space

    def _print_progress_bar(self, return_to_line_number: int = 0) -> None:
        """Clears the line and prints the progress bar

        Returns
        -------
        return_to_line_number: int, optional
            The number of the line to return to the beginning of, after the progress bar

        """

        # If we are in paused mode or we should not print, do not do anything
        if self.state_info.get('paused') is True or self._check_if_should_print() is False:
            return

        # Get the progress bar
        progress_bar = self.actions.get('get_bar')(return_to_line_number=return_to_line_number)

        # Print the progress bar
        self._direct_write(progress_bar)

    def _make_and_get_progress_bar(
            self,
            data: DataMuncher = DataMuncher(),
            include_desc_before: bool = True,
            include_desc_short_before: bool = False,
            include_prefix: bool = True,
            include_bar: bool = True,
            include_suffix: bool = True,
            include_desc_after: bool = True,
            include_desc_short_after: bool = False,
    ) -> str:
        """Returns a string containing the progress bar.

        Parameters
        ----------

        Returns
        -------
        A string containing the progress bar

        """

        # get the elements of the progress bar
        bar_prefix, bar_prefix_data = \
            self._make_and_get_bar_prefix(data.get_or_else('prefix', None)) \
            if include_prefix is True \
            else ('', DataMuncher())
        bar_suffix, bar_suffix_data = \
            self._make_and_get_bar_suffix(data.get_or_else('suffix', None)) \
            if include_suffix is True \
            else ('', DataMuncher())

        # only one of long or short description can be set. long one is preferred
        description_before, description_before_data = \
            self._make_and_get_description_before(data.get_or_else('description.full', None)) \
            if include_desc_before is True \
            else ('', DataMuncher())
        description_short_before, description_short_before_data = \
            self._make_and_get_description_short_before(data.get_or_else('description.short', None)) \
            if include_desc_short_before is True and include_desc_before is False \
            else ('', DataMuncher())

        # only one of long or short description can be set. long one is preferred
        description_after, description_after_data = \
            self._make_and_get_description_after(data.get_or_else('description.full', None)) \
            if include_desc_after is True \
            else ('', DataMuncher())
        description_short_after, description_short_after_data = \
            self._make_and_get_description_short_after(data.get_or_else('description.short', None)) \
            if include_desc_short_after is True and include_desc_after is False \
            else ('', DataMuncher())

        # have a default of this many columns for the bar
        # update it if necessary
        remaining_columns = 20
        if include_bar is True:

            if data.get_option('terminal').is_empty() or \
                    (data.get('terminal.columns') > 0 and data.get('terminal.rows') > 0):
                # Get console's width and height
                columns, rows = \
                    self._make_and_get_terminal_size(
                        data=data.get_or_else('terminal', None)
                    )
                # if columns <= 0 or rows <= 0:
                #     break
                # Calculate the written char length of the prefix, suffix, and the first line of description without the
                # special unicode or console non-printing characters
                len_bar_desc_before = \
                    len(self._remove_non_printing_chars(description_before.split('\n')[-1]).expandtabs())
                len_bar_desc_short_before = \
                    len(self._remove_non_printing_chars(description_short_before.split('\n')[-1]).expandtabs())
                len_bar_prefix = \
                    len(self._remove_non_printing_chars(bar_prefix).expandtabs())
                len_bar_suffix = \
                    len(self._remove_non_printing_chars(bar_suffix).expandtabs())
                len_bar_desc_after = \
                    len(self._remove_non_printing_chars(description_after.split('\n')[0]).expandtabs())
                len_bar_desc_short_after = \
                    len(self._remove_non_printing_chars(description_short_after.split('\n')[0]).expandtabs())

                remaining_columns = \
                    int(np.clip(
                        columns - len_bar_prefix - len_bar_suffix - 4 -
                        len_bar_desc_before - len_bar_desc_after -
                        len_bar_desc_short_before - len_bar_desc_short_after,
                        5,
                        50
                    ))  # -4 for the spaces between the fields

                # we need the information from the prefix, so if we have not found it yet, find it!
                # well, not the best design, fix it!!
                if include_prefix is False:
                    _, bar_prefix_data = self._make_and_get_bar_prefix(data.get_or_else('prefix', None))

        # get the bar itself
        bar, bar_data = \
            self._make_and_get_bar(
                length=remaining_columns,
                data=data
                    .get_option('bar')
                    # update the percentage
                    .map(lambda x: x.update({}, {"percentage": bar_prefix_data.get("percentage")}))
                    .get_or_else(None)
            ) \
            if include_bar is True \
            else ('', DataMuncher())

        progress_bar = \
            f'{description_before or description_short_before}' \
            f' {bar_prefix} {bar} {bar_suffix} ' \
            f'{description_after or description_short_after}' \
            f'{colored.attr("reset")}'

        return progress_bar

    def _get_progress_bar(self) -> str:
        """Returns the stored string containing the progress bar.

        Returns
        -------
        A string containing the progress bar

        """

        # Retrieve and return
        return self.progress_bar_info.get('progress_bar.progress_bar')

    def _update_progress_bar(self) -> None:
        """Updates the stored string that contains the whole progress bar."""

        # Retrieve and store
        self.progress_bar_info = \
            self.progress_bar_info.update(
                {'_bc': {'$regex': 'progress_bar$'}},
                {'progress_bar': self._make_and_get_progress_bar()}
            )

    # communication methods

    def _init_communication(self) -> None:
        """Initializes everything related to communication."""

        # if we are not in distributed mode, skip
        if mpi.mpi_communicator.get_size() == 1:
            return

        # initialize the 'gather info' communication
        self._init_communication_gather_info()

        # initialize the 'printer' communication
        self._init_communication_printer_gatherer()

    @abstractmethod
    def _reset_communication(self):
        """Do all the tasks required when resetting that are related to the communication"""

        raise NotImplementedError

    ## 'gather info' communication methods

    @abstractmethod
    def _init_communication_gather_info(self) -> None:
        """Initializes everything related to gathering update to the manager."""

        raise NotImplementedError

    ## 'printer gatherer' communication methods

    @abstractmethod
    def _init_communication_printer_gatherer(self) -> None:
        """Initializes everything related to printing to the manager."""

        raise NotImplementedError

    # bar prefix methods

    def _make_and_get_bar_prefix(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the string that comes before the bar.

        Parameters
        ----------
        data : DataMuncher, optional
            the data to create the bar prefix of the form:
                {
                    "state": {
                        "item":
                            {
                                "current_item_index": ,
                                "total_items_count": ,
                            }
                    }
                }
            if not provided, will create it

        Returns
        -------
        (str, DataMuncher)
            A string that comes before the bar
            Data that the string contains

        """

        # get the prefix data
        data = self._get_bar_prefix_data() if data is None else data

        # The percentage of the progress
        percent: float = self._get_percentage(data.get('state')) * 100

        bar_prefix = f'{percent:6.2f}%'

        # construct the data that has to be outputted
        output_data = DataMuncher({"percentage": percent / 100})

        # add color
        bar_prefix = f'{colored.fg("grey_74")}' \
                     f'{bar_prefix}' \
                     f'{colored.attr("reset")}'

        return bar_prefix, output_data

    def _get_bar_prefix(self) -> str:
        """Returns the stored string that comes before the bar.

        Returns
        -------
        A string that comes before the bar

        """

        # Retrieve and return
        return self.progress_bar_info.get('progress_bar.prefix')

    def _get_bar_prefix_data(self) -> DataMuncher:
        """
        Makes and return the data needed to create the bar prefix.

        Returns
        -------
        DataMuncher
            resulting data
        """

        data = \
            DataMuncher({
                "state": {
                    "item":
                        {
                            "current_item_index": self.state_info.get('item.current_item_index'),
                            "total_items_count": self.state_info.get('item.total_items_count'),
                        }
                }
            })

        return data

    def _update_bar_prefix(self) -> None:
        """Updates the stored string that comes before the bar."""

        # Retrieve and store
        self.progress_bar_info = \
            self.progress_bar_info.update(
                {'_bc': {'$regex': 'progress_bar$'}},
                {'prefix': self._make_and_get_bar_prefix()}
            )

    # bar methods

    def _make_and_get_bar(self, length: int, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns a string containing the bar itself.

        Parameters
        ----------
        length : int
            The length of the bar
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    "percentage": ,
                    "bar_chars": ,
                }

        Returns
        -------
        (str, DataMuncher)
            A string containing the bar
            Data that the string contains

        """

        # construct the data that has to be outputted
        output_data = \
            DataMuncher()

        if not length > 2:
            return '', output_data

        # get the prefix data
        data = self._get_bar_data() if data is None else data

        # The percentage of the progress
        percent: float = data.get("percentage")

        # Get the length of the bar without the borders
        bar_length = length - 2

        # get bar chars
        bar_chars = data.get("bar_chars") or self.bar_chars

        # Figure how many 'complete' bar characters (of index -1) we need
        # Figure what other character of bar characters is needed
        virtual_length = bar_length * len(bar_chars)
        whole_char_count, remainder_char_idx = divmod(int(virtual_length * percent), len(bar_chars))

        # Make the bar string
        bar: str = bar_chars[-1] * whole_char_count  # Completed parts
        bar += bar_chars[remainder_char_idx - 1] if remainder_char_idx != 0 else ''  # Half-completed parts
        bar += ' ' * (bar_length - len(bar))  # Not completed parts

        # Add the borders
        bar = f'|{bar}|'

        return bar, output_data

    def _get_bar(self) -> str:
        """Returns the stored string containing the bar itself.

        Returns
        -------
        A string containing the bar

        """

        # Retrieve and return
        return self.progress_bar_info.get('progress_bar.bar')

    def _get_bar_data(self) -> DataMuncher:
        """
        Makes and return the data needed to create the bar.

        Returns
        -------
        DataMuncher
            resulting data
        """

        # get bar prefix data to extract info
        _, data_prefix = self._make_and_get_bar_prefix()

        data = \
            DataMuncher({
                "percentage": data_prefix.get("percentage"),
                "bar_chars": self.bar_chars,
            })

        return data

    def _update_bar(self, length: int) -> None:
        """Updates the stored containing the bar.

        Parameters
        ----------
        length : int
            The length of the bar

        """

        # Retrieve and store
        self.progress_bar_info = \
            self.progress_bar_info.update(
                {'_bc': {'$regex': 'progress_bar$'}},
                {'bar': self._make_and_get_bar(length)}
            )

    # bar suffix methods

    def _make_and_get_bar_suffix(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the string that comes after the bar.

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    "state": {
                        "item": {
                            "current_item_index": ,
                            "total_items_count": ,
                        },
                    },
                    "statistics": {
                        "average": {
                            "average_item_per_update": ,
                            "average_time_per_update": ,
                        },
                        "time": {
                            "last_update_time": ,
                            "initial_progress_bar_time": ,
                        }
                    }
                }

        Returns
        -------
        (str, DataMuncher)
            A string that comes after the bar
            data built

        """

        # get the prefix data
        data = self._get_bar_suffix_data() if data is None else data

        # construct the data that has to be outputted
        output_data = \
            DataMuncher({
                "fraction": {},
                "item_per_sec": {},
                "time_since_last_update": {},
                "time_since_beginning_iteration": {},
            })

        bar_suffix, fractional_data = self._get_fractional_progress(data.get("state"))  # Fractional progress e.g. 12/20
        bar_suffix += f' '
        item_per_second: float = self._get_item_per_second(data.get("statistics"))
        item_per_second_str: str = f'{item_per_second:.2f}' if math.isfinite(item_per_second) else '?'
        bar_suffix += f'{colored.fg("grey_74")}' \
                      f'[{item_per_second_str} it/s]'
        bar_suffix += f'{colored.attr("reset")}'

        # update the output
        output_data = output_data.update({}, {
            "fraction": fractional_data,
            "item_per_sec": item_per_second,
        })

        # Time elapsed since the last update
        hours = minutes = seconds = microseconds = np.nan
        now = datetime.datetime.now()
        last_update_time = data.get('statistics.time.last_update_time')
        if math.isfinite(last_update_time):
            last_update_time = datetime.datetime.fromtimestamp(last_update_time)
            delta_time = now - last_update_time
            hours, remainder = divmod(delta_time.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            microseconds = delta_time.microseconds
            # For formatting purposes, just keep the first 4 digits
            microseconds = int(str(microseconds)[:4])
            delta_time_str_last_update = f'{hours:02d}' \
                                         f':' \
                                         f'{minutes:02d}' \
                                         f':' \
                                         f'{seconds:02d}' \
                                         f'.' \
                                         f'{microseconds:04d}'
        else:
            delta_time_str_last_update = '?'

        # update the output
        output_data = output_data.update({}, {
            '$set': {
                "time_since_last_update": {
                    "hours": hours,
                    "minutes": minutes,
                    "seconds": seconds,
                    "microseconds": microseconds,
                }
            }
        })

        # Time elapsed since the beginning of the iteration
        hours = minutes = seconds = np.nan
        init_progress_bar_time = data.get('statistics.time.initial_progress_bar_time')
        if math.isfinite(init_progress_bar_time):
            init_time = datetime.datetime.fromtimestamp(init_progress_bar_time)
            delta_time = now - init_time
            hours, remainder = divmod(delta_time.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            delta_time_str_since_iteration_beginning = f'{hours:02d}' \
                                                       f':' \
                                                       f'{minutes:02d}' \
                                                       f':' \
                                                       f'{seconds:02d}'
        else:
            delta_time_str_since_iteration_beginning = '?'

        # update the output
        output_data = output_data.update({}, {
            '$set': {
                "time_since_beginning_iteration": {
                    "hours": hours,
                    "minutes": minutes,
                    "seconds": seconds,
                }
            }
        })

        # Add the elapsed time to the bar_suffix
        bar_suffix += f' {colored.fg("grey_39")}{delta_time_str_last_update}' \
                      f'{colored.fg("grey_74")} - ' \
                      f'{delta_time_str_since_iteration_beginning}{colored.attr("reset")}'

        return bar_suffix, output_data

    def _get_bar_suffix(self) -> str:
        """Returns the stored string that comes after the bar.

        Returns
        -------
        A string that comes after the bar

        """

        # Retrieve and return
        return self.progress_bar_info.get('progress_bar.suffix')

    def _get_bar_suffix_data(self) -> DataMuncher:
        """
        Makes and return the data needed to create the bar suffix.

        Returns
        -------
        DataMuncher
            resulting data
        """

        data = \
            DataMuncher({
                "state": {
                    "item": {
                        "current_item_index": self.state_info.get('item.current_item_index'),
                        "total_items_count": self.state_info.get('item.total_items_count'),
                    },
                },
                "statistics": {
                    "average": {
                        "average_item_per_update": self.statistics_info.get("average.average_item_per_update"),
                        "average_time_per_update": self.statistics_info.get("average.average_time_per_update"),
                    },
                    "time": {
                        "last_update_time": self.statistics_info.get("time.last_update_time"),
                        "initial_progress_bar_time": self.statistics_info.get("time.initial_progress_bar_time"),
                    }
                }
            })

        return data

    def _update_bar_suffix(self) -> None:
        """Updates the stored string that comes after the bar."""

        # Retrieve and store
        self.progress_bar_info = \
            self.progress_bar_info.update(
                {'_bc': {'$regex': 'progress_bar$'}},
                {'suffix': self._make_and_get_bar_suffix()}
            )

    # description methods

    def set_description_after(self, description: str) -> None:
        """Sets the description that comes after the progress bar.

        Parameters
        ----------
        description : str
            String to update the description that comes after the progress bar

        """

        # Update the progress bar info
        self.progress_bar_info = self.progress_bar_info.update(
            {'_bc': {'$regex': 'description.full$'}},
            {'after': self._modify_description_after(description)}
        )

    def set_description_before(self, description: str) -> None:
        """Sets the description that comes before the progress bar.

        Parameters
        ----------
        description : str
            String to update the description that comes before the progress bar

        """

        # Update the progress bar info
        self.progress_bar_info = self.progress_bar_info.update(
            {'_bc': {'$regex': 'description.full$'}},
            {'before': self._modify_description_after(description)}
        )

    def set_description_short_after(self, description: str) -> None:
        """Sets the short description that comes after the progress bar.

        Parameters
        ----------
        description : str
            String to update the short description that comes after the progress bar

        """

        # make sure the short description is single line
        if len(description.split('\n')) != 1:
            raise ValueError("short description can only have a single line")

        # Update the progress bar info
        self.progress_bar_info = self.progress_bar_info.update(
            {'_bc': {'$regex': 'description.short'}},
            {'after': self._modify_description_after(description)}
        )

    def set_description_short_before(self, description: str) -> None:
        """Sets the short description that comes before the progress bar.

        Parameters
        ----------
        description : str
            String to update the short description that comes before the progress bar

        """

        # make sure the short description is single line
        if len(description.split('\n')) != 1:
            raise ValueError("short description can only have a single line")

        # Update the progress bar info
        self.progress_bar_info = self.progress_bar_info.update(
            {'_bc': {'$regex': 'description.short'}},
            {'before': self._modify_description_after(description)}
        )

    def _make_and_get_description_before(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the full description that comes before the bar.

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    'before': '',
                }

        Returns
        -------
        (str, DataMuncher)
            A string containing full description before
            data built

        """

        # get the description
        description = self._get_bar_description_before() if data is None else data.get('before')

        # construct the data that has to be outputted
        output_data = \
            DataMuncher({
                "description": description,
            })

        return description, output_data

    def _make_and_get_description_after(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the full description that comes after the bar.

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    'after': '',
                }

        Returns
        -------
        (str, DataMuncher)
            A string containing full description after
            data built

        """

        # get the description
        description = self._get_bar_description_after() if data is None else data.get('after')

        # also add the aggregation string
        subs_str = r'\1\t'
        description += \
            f'\n\n' \
            f'{REGEX_INDENTATION.sub(subs_str, self._make_and_get_aggregation_str())}' \
            f'\n'

        # construct the data that has to be outputted
        output_data = \
            DataMuncher({
                "description": description,
            })

        return description, output_data

    def _make_and_get_description_short_before(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the short description that comes before the bar.

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    'before': '',
                }

        Returns
        -------
        (str, DataMuncher)
            A string containing short description before
            data built

        """

        # get the description
        description = self._get_bar_description_short_before() if data is None else data.get('before')

        # construct the data that has to be outputted
        output_data = \
            DataMuncher({
                "description": description,
            })

        return description, output_data

    def _make_and_get_description_short_after(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the short description that comes after the bar.

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    'after': '',
                }

        Returns
        -------
        (str, DataMuncher)
            A string containing short description after
            data built

        """

        # get the description
        description = self._get_bar_description_short_after() if data is None else data.get('after')

        # also add the aggregation string
        aggregation_str = self._make_and_get_aggregation_short_str()
        # make sure its a one liner
        if '\n' in aggregation_str:
            raise ValueError('aggregation short function must result in a one-line string')
        description += f' {aggregation_str}'

        # construct the data that has to be outputted
        output_data = \
            DataMuncher({
                "description": description,
            })

        return description, output_data

    def _get_bar_description_before(self) -> str:
        """Returns the description that comes before the progress bar.

        Returns
        ----------
        The description string that comes before the bar

        """

        # Retrieve and return the progress bar description
        return self.progress_bar_info.get('progress_bar.description.full.before')

    def _get_bar_description_after(self) -> str:
        """Returns the description that comes after the progress bar.

        Returns
        ----------
        The description string that comes after the bar

        """

        # Retrieve and return the progress bar description
        return self.progress_bar_info.get('progress_bar.description.full.after')

    def _get_bar_description_short_before(self) -> str:
        """Returns the short description that comes before the progress bar.

        Returns
        ----------
        The short description string that comes before the bar

        """

        # Retrieve and return the progress bar description
        return self.progress_bar_info.get('progress_bar.description.short.before')

    def _get_bar_description_short_after(self) -> str:
        """Returns the short description that comes after the progress bar.

        Returns
        ----------
        The short description string that comes after the bar

        """

        # Retrieve and return the progress bar description
        return self.progress_bar_info.get('progress_bar.description.short.after')

    def _modify_description_before(self, description: str) -> str:
        """Modifies the description of the progress bar.

        Parameters
        ----------
        description : str
            Description string for the progress bar

        Returns
        -------
        Modified description string

        """

        return description

    def _modify_description_after(self, description: str) -> str:
        """Modifies the description of the progress bar.

        Parameters
        ----------
        description : str
            Description string for the progress bar

        Returns
        -------
        Modified description string

        """

        return description

    def _modify_description_short_before(self, description: str) -> str:
        """Modifies the description of the progress bar.

        Parameters
        ----------
        description : str
            Description string for the progress bar

        Returns
        -------
        Modified description string

        """

        return description

    def _modify_description_short_after(self, description: str) -> str:
        """Modifies the description of the progress bar.

        Parameters
        ----------
        description : str
            Description string for the progress bar

        Returns
        -------
        Modified description string

        """

        return description

    # aggregation methods

    def set_aggregation_data(self, data: Any) -> None:
        """
        Sets the aggregation data.

        This data will be passed to the custom aggregation functions provided and the result will be showed.
        This data will be sent to the manager in case of distributed run. The manager will then call a function to
        aggregate these data and use them.

        Parameters
        ----------
        data : Any
            the data to be passed to the manager for aggregation.
            note that this data has to be serialize-able.

        """

        self.progress_bar_info = self.progress_bar_info.update({}, {'aggregation_data.aggregation_data': data})

    def set_aggregator_function_full(self, func: Callable[[List], str]):
        """
        Sets the aggregator function that is used to process the aggregated data and give full output.

        Parameters
        ----------
        func : Callable[[list], str
            function that is given a list of to-be-aggregated data and should return the full aggregation.

        """

        self.actions = self.actions.update({}, {'aggregator.full': func})

    def set_aggregator_function_short(self, func: Callable[[List], str]):
        """
        Sets the aggregator function that is used to process the aggregated data and give short output.

        Parameters
        ----------
        func : Callable[[list], str
            function that is given a list of to-be-aggregated data and should return the short aggregation.

        """

        self.actions = self.actions.update({}, {'aggregator.short': func})

    def _make_and_get_aggregation_str(self) -> str:
        """
        Calls the custom full aggregation function and returns the string

        Returns
        -------
        str
            String returned by calling the custom aggregation function

        """

        aggregation_str = \
            self.actions.get('aggregator.full')([self.progress_bar_info.get('aggregation_data.aggregation_data')])

        return aggregation_str

    def _make_and_get_aggregation_short_str(self) -> str:
        """
        Calls the custom short aggregation function and returns the string

        Returns
        -------
        str
            String returned by calling the custom short aggregation function

        """

        aggregation_str = \
            self.actions.get('aggregator.short')([self.progress_bar_info.get('aggregation_data.aggregation_data')])

        return aggregation_str

    # utility methods

    def _get_percentage(self, data: DataMuncher) -> float:
        """Returns the percentage of the process.

        Parameters
        ----------
        data : DataMuncher
            the data to create the bar prefix of the form:
                {
                    "item":
                            {
                                "current_item_index": ,
                                "total_items_count": ,
                            }
                }

        Returns
        -------
        A float that is the percentage of the process

        """

        # The percentage of the progress
        if data.get('item.total_items_count') != 0:
            percent: float = data.get('item.current_item_index') / data.get('item.total_items_count')
        # if the total number of items is 0, then we are done however!
        else:
            percent: float = 1.
        percent = float(np.clip(percent, 0., 1.))

        return percent

    def _get_fractional_progress(self, data: DataMuncher) -> (str, DataMuncher):
        """Returns a string of the form x*/y* where x* and y* are the current and total number of items.

        Parameters
        ----------
        data : DataMuncher
            the data to create the bar prefix of the form:
                {
                    "item":
                            {
                                "current_item_index": ,
                                "total_items_count": ,
                            }
                }

        Returns
        -------
        (str, DataMuncher)
            A string containing the fractional progress
            data built

        """

        # Get the length of chars of total number of items for better formatting
        if data.get('item.total_items_count') > 0:
            length_items = int(np.ceil(np.log10(data.get('item.total_items_count') + 1)))
        elif data.get('item.total_items_count') == 0:
            length_items = 1
        else:
            length_items = 5

        # Create the string
        fractional_progress = f'{colored.fg("gold_3b")}' \
                              f'{data.get("item.current_item_index"): {length_items}d}' \
                              f'{colored.fg("grey_46")}' \
                              f'/'
        fractional_progress += f'{colored.fg("orange_4b")}' \
                               f'{data.get("item.total_items_count")}' \
            if data.get('item.total_items_count') >= 0 else '?'
        fractional_progress += f'{colored.attr("reset")}'

        # update the output data
        output_data = DataMuncher({
            "items": data.get("item.current_item_index"),
            "total_items": data.get("item.total_items_count") if data.get('item.total_items_count') > 0 else 0,
        })

        return fractional_progress, output_data

    def _remove_non_printing_chars(self, message: str) -> str:
        """Removes the unicode and non-printing characters from the string given and returns it.

        Parameters
        ----------
        message : str
            String to remove characters from

        Returns
        -------
        A string containing `message` with special characters removed

        """

        return REGEX_REMOVE_NONPRINT_CHARS.sub('', message)

    def _get_item_per_second(self, data: DataMuncher) -> float:
        """Returns the average number of items processed in a second.

        Parameters
        ----------
        data : DataMuncher
            the data to create the bar prefix of the form:
                {
                    "average":
                            {
                                "average_item_per_update": ,
                                "average_time_per_update": ,
                            }
                }

        Returns
        -------
        Average number of items processed in a second

        """

        average_item_per_second = \
            data.get('average.average_item_per_update') \
            / \
            data.get('average.average_time_per_update')

        return average_item_per_second

    def _update_time_counter(self) -> None:
        """Updates the internal average time counter."""

        # Figure current item time
        delta_time = time.time() - self.statistics_info.get('time.last_update_time')

        # Update the moving average
        self.statistics_info = \
            self.statistics_info.update(
                {'_bc': {'$regex': 'average$'}},
                {'average_time_per_update':
                    Calculation.exp_average(
                        item=self.statistics_info.get('average.average_time_per_update'),
                        d_item=delta_time
                    )
                }
            )

        # Update the last time
        self.statistics_info = \
            self.statistics_info.update(
                {'_bc': {'$regex': 'time$'}},
                {'last_update_time': time.time()}
            )

    def _look_to_resume(self) -> TheProgressBarBase:
        """Looks constantly to resumes the progress bar. This method should be called after pause to cause a thread
        to constantly look for a chance to resume.

        Returns
        -------
        This instance

        """

        # If the instance is not paused, skip
        if self.state_info.get('paused') is False:
            return self

        # Run the thread for checking when to resume
        self.check_for_resume_thread = threading.Thread(
            name='check_for_resume_daemon_thread',
            target=self._run_check_for_resume,
            args=(),
            daemon=True
        )
        self.check_for_resume_thread.start()

        return self

    def _sleep(self):
        """Sleep! used for sleeping between each update of the bar."""

        timeout = 1 / self._get_update_frequency()

        rs, ws, xs = select.select([self.sleep_timer_info.get('pipe.read')], [], [], timeout)

        for r in rs:
            r.recv()

    def _notify_sleep(self):
        """Figure out if it is a good time to wake the sleeping time up!"""

        if self.sleep_timer_info.get('update_interval') > 0:
            interval = self.sleep_timer_info.get('update_interval')
        elif self.sleep_timer_info.get('update_number_per_iteration') > 0:
            interval = \
                self.state_info.get('item.total_items_count') \
                // self.sleep_timer_info.get('update_number_per_iteration')
        else:
            return

        # if we should print
        # also print at 0% and 100%
        if \
                self.state_info.get('item.current_item_index') \
                >=\
                self.sleep_timer_info.get('stat.last_item_index') + interval:

            # first update the stats
            self.sleep_timer_info = \
                self.sleep_timer_info.update({}, {
                    'stat.last_item_index': self.state_info.get('item.current_item_index'),
                })

            # let the time go off
            # we will print the bar ourselves because we do not want to have a race condition: while we are telling
            # the other thread to type, this thread can go and reset the bar that would lead to undefined results
            # TODO: FIX THIS!
            # self.sleep_timer_info.get('pipe.write').send('timesup!')
            self._print_progress_bar()

    def _get_update_frequency(self) -> float:
        """Returns the number of times in a second that the progress bar should be updated.

                Returns
                -------
                The frequency at which the progress bar should be updated

                """

        # Calculated the average number of items processed per second
        average_freq: float = 1 / self.statistics_info.get('average').get('average_time_per_update')

        # Rule: update at least 2 times and at most 60 times in a second unless needed to be faster
        # Also be twice as fast as the update time difference
        # Also, if we are on pause, update with frequency 1
        if self.state_info.get('paused') is True:
            freq = 1.0
        elif self.state_info.get('mode') == self.Modes.NOTTY and self.state_info.get('role') == self.Roles.WORKER:
            freq = 1.0
        elif self.state_info.get('mode') == self.Modes.NOTTY:
            freq = 1 / 60  # if we do not have a tty, print every minute
        else:
            freq = float(np.clip(2 * average_freq, 2, 60))

        return freq

    def _exp_average(self, item: float, d_item: float, beta: float = .9) -> float:
        """
        Calculates the new exponential moving average for the inputs.

        Returns d_item if item is -np.inf

        Parameters
        ----------
        item : float
            Current value of the average
        d_item : float
            The value of the recent element
        beta : float
            Exponential moving average beta

        Returns
        -------
        Updated value of the average

        """

        average = beta * item + (1 - beta) * d_item if item != -np.inf else d_item

        return average

    # writing methods

    def _direct_write(self, msg: str) -> None:
        """Write the msg directly on the output with no buffers.

        Parameters
        ----------
        msg : str
            Message to be written

        """

        # Only write if we are not paused
        if self.state_info.get('paused') is False:
            self.stdout_handler.write(msg)
            self.stdout_handler.flush()

    def write(self, msg: str) -> None:
        """Prints a message to the output.

        Parameters
        ----------
        msg : str
            The message to be written

        """

        self.buffer.append(msg)

        # Calling 'print' will call this function twice.
        # First with the message and second with the new line character
        # If the msg ends with \n\b, then do not print the progress bar itself
        if msg == '\n' or msg.endswith('\n') or msg.endswith('\n\b'):

            with self.print_lock:

                # Add the progress bar at the end
                if not msg.endswith('\n\b') and self._check_if_should_append_progress_bar() is True:
                    self.buffer.append(self.actions.get('get_bar')())

                # Create the message from the buffer and print it with extra new line character
                msg = ''.join(self.buffer)

                self.original_sysout.write(self.cursor_modifier.get('clear_until_end'))
                self.original_sysout.write(f'{msg}')

                self.buffer = []

    def flush(self) -> None:
        """Flushes the screen."""

        self.original_sysout.write(''.join(self.buffer))
        self.original_sysout.flush()
        self.buffer = []


class TheProgressBar(TheProgressBarBase):
    """Progress bar that takes care of additional prints to the screen while updating.

    This implementation is based on the alive_progress package.
    """

    def __init__(self, stdout_handler=None, config: ConfigParser = None):
        """Initializes the instance."""

        super().__init__(stdout_handler=stdout_handler, config=config)

        # book keeping for aggregation data in case of distributed run
        self.gather_info_aggregation_data: Any = None

    # action methods

    def _check_action(self) -> bool:
        """Method to return conditional on whether we should perform the action methods."""

        if self.state_info.get('role') != self.Roles.SINGLE:
            return False

        return True

    def _configure_role(self):
        """Method to find out and set the role."""

        # find and update the role
        if self.state_info.get('parallel.is_distributed') is True:
            # update the role
            self.state_info = self.state_info.update({}, {'$set': {'role': self.Roles.WORKER}})
        else:
            # leave it be single mode
            self.state_info = self.state_info.update({}, {'$set': {'role': self.Roles.SINGLE}})

    def _set_actions(self):
        """Sets the proper actions based on the conditions."""

        if self.state_info.get('mode') == self.Modes.NORMAL:
            def get_bar_curry(return_to_line_number: int = 0):
                out = self._get_progress_bar_with_spaces(
                    data=DataMuncher(),
                    return_to_line_number=return_to_line_number,
                    include_desc_before=True,
                    include_desc_short_before=False,
                    include_prefix=True,
                    include_bar=True,
                    include_suffix=True,
                    include_desc_after=True,
                    include_desc_short_after=False,
                )
                return out

            self.actions = self.actions.update({}, {
                'get_bar': get_bar_curry,
            })

        elif self.state_info.get('mode') == self.Modes.NOTTY:
            def get_bar_curry(return_to_line_number: int = 0):
                out = self._get_progress_bar_with_spaces(
                    # the following is to avoid reading the terminal size and update the text
                    data=DataMuncher({"terminal": {"columns": -1, "rows": -1}}),
                    return_to_line_number=return_to_line_number,
                    include_desc_before=False,
                    include_desc_short_before=True,
                    include_prefix=True,
                    include_bar=False,
                    include_suffix=True,
                    include_desc_after=False,
                    include_desc_short_after=True,
                )
                return out

            self.actions = self.actions.update({}, {
                'get_bar': get_bar_curry,
            })

    def run(self):
        """Runs the daemon threads."""

        # if we are single and non-distributed, just print
        if self.state_info.get('role') == self.Roles.SINGLE:
            self._make_run_thread()
            self.run_thread_info.get('print.main').start()

        # if we are in a distributed mode, just publish
        elif self.state_info.get('role') == self.Roles.WORKER:
            self._make_gather_info_thread()
            self.run_thread_info.get('gather_info.main').start()

    def reset(self, return_to_line_number: int = -1) -> TheProgressBar:
        """Resets the progress bar and returns its instance.

        Parameters
        ----------
        return_to_line_number: int, optional
            The line number to return to

        Returns
        -------
        This instance

        """

        # do the resetting
        super().reset(return_to_line_number=return_to_line_number)

        return self

    def run_gather_info(self) -> None:
        """Publish the information to the broker for communication."""

        while self.run_thread_info.get('gather_info.main'):

            # do the communication tasks
            self._send_gather_info_broker_info()

            # sleep :D
            self._sleep()

    # terminal related methods

    def _make_isatty(self) -> Callable:
        """Makes the new atty function."""

        if ENV_VARS.get('force_tty') in os.environ.keys():
            force_tty = os.environ[ENV_VARS.get('force_tty')]
            if force_tty == '1':
                return lambda: True
            elif force_tty == '0':
                return lambda: False
            else:
                raise ValueError(
                    f"cannot understand the '{ENV_VARS.get('force_tty')}' environmental variable. accepted "
                    f"values are '0' and '1'."
                )

        return sys.stdout.isatty

    def _check_if_foreground(self) -> bool:
        """
        Checks if we are running in foreground

        Returns
        -------
        bool

        """

        # most of the time the actual foreground checking does not work, for example in Docker environment with no
        # pseudo-terminal. so, for now, we just remove it
        return True

    def _check_if_should_print(self) -> bool:
        """
        Checks whether we should print.

        Returns
        -------
        bool

        """

        if self.state_info.get('role') != self.Roles.SINGLE:
            return False

        return super()._check_if_should_print()

    # progress bar methods

    def _make_and_get_progress_bar_data(self) -> DataMuncher:

        # get the progress bar data in the correct form!
        data: DataMuncher = gather_info_data_msg_template.get('progress_bar')

        # update it!
        data = data.update({}, {'$update_only': {
            'prefix': self._get_bar_prefix_data(),
            'bar': self._get_bar_data(),
            'suffix': self._get_bar_suffix_data(),
            'description': {
                'full': {
                    'before': self._get_bar_description_before(),
                    'after': self._get_bar_description_after(),
                },
                'short': {
                    'before': self._get_bar_description_short_before(),
                    'after': self._get_bar_description_short_after(),
                },
            }
        }})

        return data

    # communication methods

    def _reset_communication(self):
        """Do all the tasks required when resetting that are related to the communication"""

        # if not in distributed mode, skip
        if not mpi.mpi_communicator.is_distributed():
            return

        # send the final message
        self._send_gather_info_broker_info()

    def _send_gather_info_broker_info(self) -> None:
        """Make and send the information to the broker."""

        # gather data
        data: dict = self._make_and_get_gather_info_communication_message_json()

        # send the data
        rabbitmq.rabbitmq_communicator.publish(
            body=data,
            routing_key=rabbit_data.get('gather_info.message.routing_key'),
            exchange=self.communication_info.get('gather_info.exchange.exchange'),
            # also declare our queue
            extra_declare=[self.communication_info.get('gather_info.queue.queue')]
        )

    ## 'gather info' communication methods

    def _init_communication_gather_info(self) -> None:
        """Initializes everything related to gathering update to the manager."""

        # define the exchange
        exchange = rabbitmq.rabbitmq_communicator.make_and_get_exchange(
            name=rabbit_data.get('gather_info.exchange.name'),
            type=rabbit_data.get('gather_info.exchange.type'),
            return_on_exist=True,
            durable=rabbit_data.get('gather_info.exchange.durable'),
            auto_delete=rabbit_data.get('gather_info.exchange.auto_delete'),
            delivery_mode=rabbit_data.get('gather_info.exchange.delivery_mode'),
        )

        # define the queue
        queue = rabbitmq.rabbitmq_communicator.make_and_get_queue(
            name=rabbit_data.get('gather_info.queue.name'),
            exchange_name=rabbit_data.get('gather_info.exchange.name'),
            routing_key=rabbit_data.get('gather_info.message.routing_key'),
            return_on_exist=True,
            max_length=rabbit_data.get('gather_info.queue.max_length'),
            durable=rabbit_data.get('gather_info.queue.durable'),
            auto_delete=rabbit_data.get('gather_info.queue.auto_delete'),
        )

        # update the info
        self.communication_info = self.communication_info.update(
            {'_bc': {'$regex': r'gather_info$'}},
            {
                'exchange.exchange': exchange,
                'queue.queue': queue,
            }
        )

    def _make_and_get_gather_info_communication_message(self) -> DataMuncher:
        """
        Makes and returns the message needed for communication in distributed mode.

        Returns
        -------
        DataMuncher
            the message in DataMuncher format

        """

        data = gather_info_data_msg_template.update(
            {},
            {'$update_only': {
                'rank': mpi.mpi_communicator.get_rank(),
                'progress_bar': self._make_and_get_progress_bar_data(),
                'current_iteration_index': self.state_info.get('item.current_iteration_index'),
                'aggregation_data': self.progress_bar_info.get('aggregation_data'),
            }}
        )

        return data

    def _make_and_get_gather_info_communication_message_json(self) -> dict:
        """
        Makes and returns json message needed for communication in distributed mode.

        Returns
        -------
        str
            the json representation of the message

        """

        data: dict = self._make_and_get_gather_info_communication_message().dict_representation()

        return data

    ## 'printer gatherer' communication methods

    def _init_communication_printer_gatherer(self) -> None:
        """Initializes everything related to printing to the manager."""

        # define the exchange
        exchange = rabbitmq.rabbitmq_communicator.make_and_get_exchange(
            name=rabbit_data.get('printer_gatherer.exchange.name'),
            type=rabbit_data.get('printer_gatherer.exchange.type'),
            return_on_exist=True,
            durable=rabbit_data.get('printer_gatherer.exchange.durable'),
            auto_delete=rabbit_data.get('printer_gatherer.exchange.auto_delete'),
            delivery_mode=rabbit_data.get('printer_gatherer.exchange.delivery_mode'),
        )

        # define the queue
        queue = rabbitmq.rabbitmq_communicator.make_and_get_queue(
            name=rabbit_data.get('printer_gatherer.queue.name'),
            exchange_name=rabbit_data.get('printer_gatherer.exchange.name'),
            routing_key=rabbit_data.get('printer_gatherer.message.routing_key'),
            return_on_exist=True,
            durable=rabbit_data.get('printer_gatherer.queue.durable'),
            auto_delete=rabbit_data.get('printer_gatherer.queue.auto_delete'),
        )

        # update the info
        self.communication_info = self.communication_info.update(
            {'_bc': {'$regex': r'printer_gatherer$'}},
            {
                'exchange.exchange': exchange,
                'queue.queue': queue,
            }
        )

    def _send_printer_gatherer_broker_info(self, data: Any) -> None:
        """
        Send the information to the broker as part of the 'printer' communication.

        Parameters
        ----------
        data : Any
            data to be sent to the manager for printing

        """

        # send the data
        try:
            rabbitmq.rabbitmq_communicator.publish(
                body=data,
                routing_key=rabbit_data.get('printer_gatherer.message.routing_key'),
                exchange=self.communication_info.get('printer_gatherer.exchange.exchange'),
                # also declare our queue
                extra_declare=[self.communication_info.get('printer_gatherer.queue.queue')]
            )
        except BaseException as e:
            sys.stderr.write(
                f"encountered an error while publishing to rabbitmq to exchange of "
                f"'{self.communication_info.get('printer_gatherer.exchange.exchange')}'"
                f"with error of '{e}'\n"
                f"here is the data that failed to be sent:\n"
                f"\t{data}"
            )
            sys.stderr.flush()

    def write(self, msg: str) -> None:
        """
        Prints a message to the output.

        Parameters
        ----------
        msg : str
            The message to be written

        """

        # if we are in distributed mode, send the msg to the manager
        if self.state_info.get('parallel.is_distributed') is True:
            self._send_printer_gatherer_broker_info(msg)
            return

        # go with the usual printing
        super().write(msg)


class TheProgressBarParallelManager(TheProgressBarBase):

    def __init__(self, stdout_handler=None, config: ConfigParser = None):
        """Initializes the instance."""

        super().__init__(stdout_handler=stdout_handler, config=config)

        # keep worker info
        initial_worker_gather_info = {
            'worker': {
                str(index): gather_info_data_msg_template
                for index in range(mpi.mpi_communicator.get_size())
            },
        }
        self.worker_gather_info = DataMuncher(initial_worker_gather_info)

        # keeping mpirun process info
        self.mpirun_process_info = self._init_mpi_process_info()

    # action methods

    def activate(self) -> TheProgressBarParallelManager:
        """Activates the progress bar: redirected stdout to this class and prints the progress bar

        Returns
        -------
        This instance

        """

        if mpi.mpi_communicator.is_main_rank() is False:
            return self

        return super().activate()

    def run(self):
        """Runs the daemon threads."""

        # if we are in a distributed mode, just publish
        if self.state_info.get('role') == self.Roles.MANAGER and mpi.mpi_communicator.is_main_rank():

            # run the gather info thread
            self._make_gather_info_thread()
            self.run_thread_info.get('gather_info.main').start()

            # run the printer-gatherer thread
            self._make_printer_gatherer_thread()
            self.run_thread_info.get('printer_gatherer.main').start()

            # run the print thread
            self._make_run_thread()
            self.run_thread_info.get('print.main').start()

    def reset(self, return_to_line_number: int = -1) -> TheProgressBarParallelManager:
        """Resets the progress bar and returns its instance.

        Parameters
        ----------
        return_to_line_number: int, optional
            The line number to return to

        Returns
        -------
        This instance

        """

        # do the resetting
        super().reset(return_to_line_number=return_to_line_number)

        # reset the worker info
        initial_worker_gather_info = {
            str(index): gather_info_data_msg_template
            for index in range(mpi.mpi_communicator.get_size())
        }
        self.worker_gather_info = self.worker_gather_info.update({}, {'worker': DataMuncher(initial_worker_gather_info)})

        return self

    def set_number_items(self, number_of_items: int) -> TheProgressBarParallelManager:
        """Set the total number of the items.

        Parameters
        ----------
        number_of_items : int
            The total number of items

        Returns
        -------
        This instance

        """

        # do nothing
        # the total number of items will be updated from the workers

        return self

    def _set_number_items(self, number_of_items: int) -> TheProgressBarParallelManager:
        """Set the total number of the items.

        Parameters
        ----------
        number_of_items : int
            The total number of items

        Returns
        -------
        This instance

        """

        super().set_number_items(number_of_items)

        return self

    def run_gather_info(self) -> None:
        """Consumes the information from the broker for communication."""

        consumer: rabbitmq.RabbitMQConsumer = self.communication_info.get('gather_info.consumer.consumer')
        consumer.run()

    def _act_on_gather_info(self, body, message: kombu.Message):
        """Callback handler for when a new message arrives for gather info operation."""

        # we only accept jsons!
        if message.content_type != "application/json":
            return

        # turn the body, which is json to a data muncher
        data: DataMuncher = DataMuncher(body)

        # only accept the info if it is in the same iteration, i.e. has the same current_iteration_index
        current_iteration_index = self.state_info.get('item.current_iteration_index')
        if \
                data.get_value_option('current_iteration_index')\
                .filter(lambda x: x == current_iteration_index)\
                .is_empty():
            return

        # do the updates based on the received info
        self._act_on_gather_info_update(data)

        # save the result
        self.worker_gather_info = self.worker_gather_info.update(
            {},
            {f'worker.{data.get("rank")}': data}
        )

        # if data.get('current_iteration_index') == 3:
        #     data.print()

        # self._direct_write(self._get_progress_bar_with_spaces())

    def _act_on_gather_info_update(self, data: DataMuncher):
        """Update this instance counter based on the received data in gather info process."""

        # update the total count
        if self.worker_gather_info.get(
                f'worker.{data.get("rank")}.progress_bar.prefix.state.item.total_items_count') == -np.inf \
                and data.get('progress_bar.prefix.state.item.total_items_count') != -np.inf:
            new_total = data.get('progress_bar.prefix.state.item.total_items_count')
            updated_total = self.state_info.get_value_option('item.total_items_count') \
                .filter(lambda x: x != -np.inf) \
                .or_else(Some(0)) \
                .map(lambda x: x + new_total).get()
            self._set_number_items(updated_total)

        # update myself!
        # first, extract current item index from somewhere!
        # then find the difference of the current version we have of it and the new version we received to update myself
        updated_item_index = \
            data.get('progress_bar.prefix.state.item.current_item_index')
        current_item_index = \
            self.worker_gather_info.get(f'worker.{data.get("rank")}.progress_bar.prefix.state.item.current_item_index')
        delta_item_count = \
            updated_item_index - current_item_index
        self.update(count=delta_item_count)

    def run_printer_gatherer(self) -> None:
        """Consumes the broker information.."""

        consumer: rabbitmq.RabbitMQConsumer = self.communication_info.get('printer_gatherer.consumer.consumer')
        consumer.run()

    def _make_printer_gatherer_thread(self) -> None:
        """Makes and sets the printer_gatherer thread"""

        # now, create the thread object
        printer_gatherer_thread = threading.Thread(
            name='printer_gatherer_daemon_thread',
            target=self.run_printer_gatherer,
            args=(),
            daemon=True
        )
        self.run_thread_info = self.run_thread_info.update({}, {'printer_gatherer.main': printer_gatherer_thread})

    def _act_on_printer_gatherer(self, body, message: kombu.Message):
        """Callback handler for when a new message arrives for printer gatherer operation."""

        # we only accept strings!
        # if message.content_type != "application/json":
        #     return

        # just print it!
        self.write(body)

    def _check_action(self) -> bool:
        """Method to return conditional on whether we should perform the action methods."""

        if self.state_info.get('role') != self.Roles.MANAGER:
            return False

        return True

    def _configure_role(self):
        """Method to find out and set the role."""

        # find and update the role
        if self.state_info.get('parallel.is_distributed') is True:
            # update the role
            self.state_info = self.state_info.update({}, {'$set': {'role': self.Roles.MANAGER}})
        else:
            self.state_info = self.state_info.update({}, {'$set': {'role': self.Roles.SINGLE}})

    def _set_actions(self):
        """Sets the proper actions based on the conditions."""

        if self.state_info.get('mode') == self.Modes.NORMAL:
            def get_bar_curry(return_to_line_number: int = 0):
                out = self._get_progress_bar_with_spaces(
                    data=DataMuncher(),
                    return_to_line_number=return_to_line_number,
                    include_desc_before=True,
                    include_desc_short_before=False,
                    include_prefix=True,
                    include_bar=True,
                    include_suffix=True,
                    include_desc_after=True,
                    include_desc_short_after=False,
                )
                return out

            self.actions = self.actions.update({}, {
                'get_bar': get_bar_curry,
            })

        # TODO: FIX THIS!
        elif self.state_info.get('mode') == self.Modes.NOTTY:
            def get_bar_curry(return_to_line_number: int = 0):
                out = self._get_progress_bar_with_spaces(
                    # the following is to avoid reading the terminal size and update the text
                    data=DataMuncher({"terminal": {"columns": -1, "rows": -1}}),
                    return_to_line_number=return_to_line_number,
                    include_desc_before=False,
                    include_desc_short_before=True,
                    include_prefix=True,
                    include_bar=False,
                    include_suffix=True,
                    include_desc_after=False,
                    include_desc_short_after=True,
                )
                return out

            self.actions = self.actions.update({}, {
                'get_bar': get_bar_curry,
            })

        # if self.state_info.get('mode') == self.Modes.NORMAL:
        #     def get_bar_curry(return_to_line_number: int = 0):
        #         out = self._get_progress_bar_with_spaces(
        #             data=XXX,
        #             return_to_line_number=return_to_line_number,
        #             include_desc_before=True,
        #             include_desc_short_before=False,
        #             include_prefix=True,
        #             include_bar=True,
        #             include_suffix=True,
        #             include_desc_after=True,
        #             include_desc_short_after=False,
        #         )
        #         return out
        #
        #     self.actions = self.actions.update({}, {
        #         'get_bar': get_bar_curry,
        #     })
        #
        # elif self.state_info.get('mode') == self.Modes.NOTTY:
        #     def get_bar_curry(return_to_line_number: int = 0):
        #         out = self._get_progress_bar_with_spaces(
        #             # the following is to avoid reading the terminal size and update the text
        #             data=XXXDataMuncher({"terminal": {"columns": -1, "rows": -1}}),
        #             return_to_line_number=return_to_line_number,
        #             include_desc_before=False,
        #             include_desc_short_before=True,
        #             include_prefix=True,
        #             include_bar=False,
        #             include_suffix=True,
        #             include_desc_after=False,
        #             include_desc_short_after=True,
        #         )
        #         return out
        #
        #     self.actions = self.actions.update({}, {
        #         'get_bar': get_bar_curry,
        #     })

        return
        raise NotImplementedError

    # description methods

    def set_description_after(self, description: str) -> None:
        """Sets the description that comes after the progress bar.

        Parameters
        ----------
        description : str
            String to update the description that comes after the progress bar

        """

        # do nothing
        return

    def set_description_short_after(self, description: str) -> None:
        """Sets the short description that comes after the progress bar.

        Parameters
        ----------
        description : str
            String to update the short description that comes after the progress bar

        """

        # do nothing
        return

    def _make_and_get_description_after(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the full description that comes after the bar.

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    'after': '',
                }

        Returns
        -------
        (str, DataMuncher)
            A string containing full description after
            data built

        """

        # get the passed description and return it
        if data is not None:
            return data.get('after'), DataMuncher({"description": data.get('after')})

        # get the terminal size and manipulate it
        # we assume that the bars are indented by a fixed amount. this amount is the max we can indent them
        terminal_cols, terminal_rows = \
            self._make_and_get_terminal_size() \
            if self.state_info.get('mode') == self.Modes.NORMAL \
            else (-1, -1)
        terminal_cols -= len(('\t' * 3).expandtabs())

        # to minimize the lookup, make a local variable
        attr_reset = colored.attr("reset")

        # construct the progress bar for each worker
        worker_progress_bars: List[str] = []
        for rank in range(mpi.mpi_communicator.get_size()):
            progress_bar = \
                self._make_and_get_progress_bar(
                    self.worker_gather_info.get(f'worker.{rank}')
                    .update(
                        {},
                        {
                            '$set':
                            {
                                # set terminal size
                                'progress_bar.terminal': {'columns': terminal_cols, 'rows': terminal_rows}},
                                # remove any descriptions
                                # 'progress_bar.description': gather_info_data_msg_template.get('progress_bar.description'),
                            }
                    )
                    .get('progress_bar'),
                    include_desc_before=False,
                    include_desc_short_before=False,
                    include_prefix=True,
                    include_bar=True,
                    include_suffix=True,
                    include_desc_after=False,
                    include_desc_short_after=True,
                )
            worker_progress_bars.append(
                f'{colored.fg("blue")}{rank} {colored.fg("dodger_blue_3")}\u21C0{attr_reset} {progress_bar}'
            )

        # construct the whole thing
        prefix = '\t' * 2

        # the progress bars
        border_color = colored.fg("chartreuse_3a")
        final_bars = \
            f'{prefix}{border_color}\u251C{attr_reset} ' + \
            f'\n{prefix}{border_color}\u251C{attr_reset} '.join(worker_progress_bars[:-1]) + \
            f'\n{prefix}{border_color}\u2514{attr_reset} {worker_progress_bars[-1]}'

        # get the aggregation
        # first, gather all aggregation data
        aggregation_data = \
            [
                self.worker_gather_info.get_or_else(f'worker.{rank}.aggregation_data.aggregation_data', None)
                for rank
                in range(self.state_info.get('parallel.size'))
            ]
        # now, get the string
        aggregation_str = self.actions.get('aggregator.full')(aggregation_data)
        # finally, indent the string
        aggregation_str = REGEX_INDENTATION.sub(r'\1' + prefix, aggregation_str)

        # construct the final output
        description = \
            f'\n' \
            f'{prefix}{border_color}\u25CD{attr_reset}\n' \
            f'{final_bars}\n\n' \
            f'{aggregation_str}'

        # construct the data that has to be outputted
        output_data = \
            DataMuncher({
                "description": description,
            })

        return description, output_data

    # communication methods

    def _init_mpi_process_info(self) -> DataMuncher:
        """Finds the mpi process info and return them as a DataMuncher."""

        # get the stdout file descriptor path
        try:
            stdout_fd = mpi.mpi_communicator.mpi_tty_fd
        except:
            stdout_fd = None

        # try to find the terminal size
        try:
            out = \
                subprocess.check_output(
                    ['stty',
                     '-F', stdout_fd,
                     'size'])\
                .decode('utf-8').strip().split()
            out = [int(item) for item in out]
            lines, columns = out
        except:
            lines = columns = -1

        # if we get a zero, look for the env variable for tty size
        if lines == columns == 0:
            if ENV_VARS.get('tty_size') in os.environ.keys():
                tty_size = os.environ[ENV_VARS.get('tty_size')]
                try:
                    lines, columns = [int(item) for item in tty_size.split()]
                except:
                    self._log.warning(f"failed getting terminal size from {ENV_VARS.get('tty_size')} environmental var")

        data = DataMuncher({
            'mpirun': {
                'pid': os.getppid(),
                'fd': {
                    'stdout': stdout_fd,
                },
            },
            'terminal': {
                'size': {  # this is the initial size of the terminal
                    'lines': lines,
                    'columns': columns,
                }
            }
        })

        return data

    def _reset_communication(self):
        """Do all the tasks required when resetting that are related to the communication"""

        # do nothing for now!
        return

    ## 'gather info' communication methods

    def _init_communication_gather_info(self) -> None:
        """Initializes everything related to gathering update to the manager."""

        # define the exchange
        exchange = rabbitmq.rabbitmq_communicator.make_and_get_exchange(
            name=rabbit_data.get('gather_info.exchange.name'),
            type=rabbit_data.get('gather_info.exchange.type'),
            return_on_exist=True,
            durable=rabbit_data.get('gather_info.exchange.durable'),
            auto_delete=rabbit_data.get('gather_info.exchange.auto_delete'),
            delivery_mode=rabbit_data.get('gather_info.exchange.delivery_mode'),
        )

        # update the info
        self.communication_info = self.communication_info.update(
            {'_bc': {'$regex': r'gather_info$'}},
            {
                'exchange.exchange': exchange,
            }
        )

        # make all the queues
        queue_names = []
        for index in range(mpi.mpi_communicator.get_size()):
            # define the queue
            queue_name = rabbit_data.get('gather_info.queue.name_template').replace('<rank>', str(index))
            queue = rabbitmq.rabbitmq_communicator.make_and_get_queue(
                name=queue_name,
                exchange_name=rabbit_data.get('gather_info.exchange.name'),
                routing_key=rabbit_data.get('gather_info.message.routing_key'),
                return_on_exist=True,
                max_length=rabbit_data.get('gather_info.queue.max_length'),
                durable=rabbit_data.get('gather_info.queue.durable'),
                auto_delete=rabbit_data.get('gather_info.queue.auto_delete'),
            )

            # update the info
            queue_names.append(queue_name)
            self.communication_info = self.communication_info.update(
                {'_bc': {'$regex': r'gather_info$'}},
                {
                    f'queue.{index}.queue': queue,
                }
            )

        # make the consumer and store it
        consumer = \
            rabbitmq.rabbitmq_communicator.consume(
                name=rabbit_data.get('gather_info.consumer.name'),
                queue_names=queue_names,
            ).add_callback([self._act_on_gather_info])
        self.communication_info = self.communication_info.update(
            {'_bc': {'$regex': r'gather_info$'}},
            {
                f'consumer.consumer': consumer,
            }
        )

    ## 'printer gatherer' communication methods

    def _init_communication_printer_gatherer(self) -> None:
        """Initializes everything related to printing to the manager."""

        # define the exchange
        exchange = rabbitmq.rabbitmq_communicator.make_and_get_exchange(
            name=rabbit_data.get('printer_gatherer.exchange.name'),
            type=rabbit_data.get('printer_gatherer.exchange.type'),
            return_on_exist=True,
            durable=rabbit_data.get('printer_gatherer.exchange.durable'),
            auto_delete=rabbit_data.get('printer_gatherer.exchange.auto_delete'),
            delivery_mode=rabbit_data.get('printer_gatherer.exchange.delivery_mode'),
        )

        # define the queue
        queue = rabbitmq.rabbitmq_communicator.make_and_get_queue(
            name=rabbit_data.get('printer_gatherer.queue.name'),
            exchange_name=rabbit_data.get('printer_gatherer.exchange.name'),
            routing_key=rabbit_data.get('printer_gatherer.message.routing_key'),
            return_on_exist=True,
            durable=rabbit_data.get('printer_gatherer.queue.durable'),
            auto_delete=rabbit_data.get('printer_gatherer.queue.auto_delete'),
        )

        # make the consumer and store it
        consumer = \
            rabbitmq.rabbitmq_communicator.consume(
                name=rabbit_data.get('printer_gatherer.consumer.name'),
                queue_names=[rabbit_data.get('printer_gatherer.queue.name')],
            ).add_callback([self._act_on_printer_gatherer])

        # update the info
        self.communication_info = self.communication_info.update(
            {'_bc': {'$regex': r'printer_gatherer$'}},
            {
                'exchange.exchange': exchange,
                'queue.queue': queue,
                'consumer.consumer': consumer,
            }
        )

    # progress bar methods

    def _make_and_get_progress_bar_data(self) -> DataMuncher:

        pass

    # terminal related methods

    def _make_and_get_terminal_size(self, data: DataMuncher = None) -> (int, int):
        """
        Returns the size of the terminal in form of (columns, rows).

        Parameters
        ----------
        data : DataMuncher
            the data needed to create the bar in form of:
                {
                    "rows": ,
                    "columns": ,
                }

        Returns
        -------
        (int, int)
            terminal size in form of (columns, rows)
        """

        if data is not None:
            return data.get('columns'), data.get('rows')

        if self._check_if_atty() is False:
            return -1, -1

        # THE FOLLOWING SUBPROCESS TASK IS RATHER HEAVY, SO FOR NOW, WE IGNORE IT AND GO WITH THE INITIAL TERMINAL SIZE
        lines = self.mpirun_process_info.get('terminal.size.lines')
        columns = self.mpirun_process_info.get('terminal.size.columns')

        return columns, lines

        # we are in distributed mode which means the actual terminal belongs to mpirun, our parent process
        # therefore, we read the parent process's terminal size
        # try:
        #     out = \
        #         subprocess.check_output(
        #             ['stty',
        #              '-F', self.mpirun_process_info.get('mpirun.fd.stdout'),
        #              'size'])\
        #             .decode('utf-8').strip().split()
        #     out = [int(item) for item in out]
        #     lines, columns = out
        # except:
        #     lines = columns = -1
        #
        # return columns, lines

    def _make_isatty(self) -> Callable:
        """Makes the new atty function."""

        if ENV_VARS.get('force_tty') in os.environ.keys():
            force_tty = os.environ[ENV_VARS.get('force_tty')]
            if force_tty == '1':
                return lambda: True
            elif force_tty == '0':
                return lambda: False
            else:
                raise ValueError(
                    f"cannot understand the '{ENV_VARS.get('force_tty')}' environmental variable. accepted"
                    f"values are '0' and '1'."
                )

        if mpi.mpi_communicator.is_distributed() is False:
            return sys.stdout.isatty

        # if we are in distributed mode running with mpirun, we have to check for the stdout file descriptor of our
        # parent process, which is the mpirun process itself
        parent_stdout_fd = mpi.mpi_communicator.mpi_tty_fd
        # if mpirun is connected to a tty, then the file descriptor 1 of it has to have a group owner of 'tty' on Linux
        if parent_stdout_fd.group() == 'tty':
            return lambda: True
        else:
            return lambda: False

    def _check_if_foreground(self) -> bool:
        """
        Checks if we are running in foreground

        Returns
        -------
        bool

        """

        # THIS IS CUMBERSOME! IT TAKES A LONG TIME TO RUN THE FOLLOWING SUBPROCESS COMMAND
        # SO, FOR NOW, WE DO NOT TEST WHETHER IF WE ARE IN THE FOREGROUND OR NOT
        return True

        # the following runs a subprocess command to determine if we are running in the foreground
        # out = subprocess.check_output(['ps', '-o', 'stat', '-p', str(os.getpid())])
        # return True if '+' in out.decode('utf-8') else False

    def _check_if_should_print(self) -> bool:
        """
        Checks whether we should print.

        Returns
        -------
        bool

        """

        if self.state_info.get('role') != self.Roles.MANAGER:
            return False

        return super()._check_if_should_print()

    # utility methods

    def _notify_sleep(self):
        """Figure out if it is a good time to wake the sleeping time up!"""

        # check if we have received data from all workers, then proceed
        for rank in range(self.state_info.get('parallel.size')):
            if self.worker_gather_info.get(
                    f'worker.{rank}.progress_bar.prefix.state.item.total_items_count') == -np.inf:
                return

        return super()._notify_sleep()
