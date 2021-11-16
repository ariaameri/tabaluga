from __future__ import annotations
import multiprocessing
import select
import threading
import sys
from typing import List
from abc import abstractmethod, ABC
import numpy as np
import time
import datetime
import os
import colored
from ..util.data_muncher import DataMuncher
from ..util.calculation import Calculation
import re
import fcntl
import termios
import struct
from enum import Enum


class TheProgressBarBase(ABC):

    class Modes(Enum):
        """Enum for modes of operation"""

        NORMAL = 1
        NOTTY = 2

    class Roles(Enum):
        """Enum for type of roles"""

        SINGLE = 1
        MANAGER = 2  # Whether we are in master mode, meaning we are responsible for printing the progress bar
        WORKER = 3

    def __init__(self, stdout_handler=None):
        """Initializes the instance."""

        # Create a lock so that one thread at a time can use the console to write
        self.print_lock = threading.Lock()

        # Create a threading.Event for knowing when to write the progress bar
        self.event_print = threading.Event()

        # A daemon thread placeholder for running the update of the progress bar
        # Also, a daemon thread placeholder for when on pause
        self.run_thread: threading.Thread = threading.Thread(
            name='run_daemon_thread',
            target=self.run,
            args=(),
            daemon=True
        )  # Setting it to None will cause the daemon process not to run
        self.check_for_resume_thread: threading.Thread = None

        # Get a CursorModifier that contains ANSI escape codes for the cursor
        self.cursor_modifier = self.CursorModifier()

        # Console controlling book keeping
        self.original_sysout = sys.__stdout__
        self._isatty_original = sys.stdout.isatty
        self._fileno_original = sys.stdout.fileno

        # Keep the stdout handler to write to
        self.stdout_handler = stdout_handler or self.original_sysout

        # keep the actions that we should take based on the configurations
        initial_actions = {
            'get_bar': self._get_progress_bar_with_spaces,
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
            'update_interval': -1,
            # the number of times we should do an update in an iteration
            'update_number_per_iteration': 4,
            'stat': {
                'last_item_index': 0,
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
            }
        }
        self.progress_bar_info = DataMuncher(initial_progress_bar_info)

        # Book keeping for the information regarding the collected statistics
        initial_statistics_info = {
            'time': {
                'initial_run_time': -1,  # The time when this instance is first activated
                'initial_progress_bar_time': -1,  # The time when this instance is activated or reset,
                                                  # i.e. the time when this current, specific progress bar started
                'last_update_time': -1  # The time when the latest update to this instance happened
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
            # Whether we should write to some external stdout handler or take care of it ourselves
            'external_stdout_handler': True if stdout_handler is not None else False,
            'item': {
                'total_items_count': -1,  # Count of total number of batches expected
                'current_item_index': 0  # Current batch item/index/number
            }
        }
        self.state_info = DataMuncher(initial_state_info)

        # A buffer for the messages to be printed
        self.buffer: List = []

        # Set of characters to be used for filling the bar
        self.bar_chars: str = '▏▎▍▌▋▊▉█'

    def __enter__(self) -> TheProgressBarBase:

        return self.activate()

    def __exit__(self, exc_type, exc_val, exc_tb):

        return self.deactivate()

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

        # Set the initial time
        current_time = time.time()
        self.statistics_info = \
            self.statistics_info \
                .update(
                    {'_bc': '.time'},
                    {'initial_run_time': current_time, 'initial_progress_bar_time': current_time}
                )

        return self

    @abstractmethod
    def deactivate(self) -> None:
        """Deactivates the progress bar: redirected stdout to itself and closes the progress bar"""

        raise NotImplementedError

    def pause(self) -> TheProgressBarBase:
        """Pauses the progress bar: redirected stdout to itself and stops prints the progress bar.

        This method permanently pauses the instance. To resume run either the `resume` or `_look_to_resume` method.

        Returns
        -------
        This instance

        """

        # If the instance is already paused, skip
        if self.state_info.get('paused') is True:
            return self

        # Update the state to know we are paused
        self.state_info = self.state_info.update({}, {'paused': True})

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

        return self

    def run(self) -> None:
        """Prints the progress bar and takes care of other controls.

        Method to be run by the daemon progress bar thread.
        """

        while self.run_thread:

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

    def reset(self) -> TheProgressBarBase:
        """Resets the progress bar and returns its instance.

        Returns
        -------
        This instance

        """

        # Set the initial time
        current_time = time.time()
        self.statistics_info = \
            self.statistics_info\
                .update(
                    {'_bc': {'$regex': 'time$'}},
                    {'initial_progress_bar_time': current_time, 'last_update_time': current_time}
                )

        # Reset the current item counter
        self.state_info = self.state_info.update(
            {'_bc': {'$regex': 'item$'}},
            {'$set': {'current_item_index': 0, 'total_items_count': -1}}
        )

        # Reset the sleep info
        self.sleep_timer_info = self.sleep_timer_info.update(
            {},
            {'$set': {'stat.last_item_index': 0}},
        )

        # Reset the descriptions
        self.set_description_after('')
        self.set_description_before('')
        self.set_description_short_after('')
        self.set_description_short_before('')

        return self

    @abstractmethod
    def update(self, count: int) -> None:
        """Updates the progress bar by adding 'count' to the number of items.

        Parameter
        ---------
        count : int
            The count at which the progress items should be increased. It has to be non-negative

        """

        raise NotImplementedError

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

        env = os.environ

        def ioctl_GWINSZ(fd):
            try:
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            except:
                return
            return cr

        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except:
                pass
        if not cr:
            cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

            ### Use get(key[, default]) instead of a try/catch
            # try:
            #    cr = (env['LINES'], env['COLUMNS'])
            # except:
            #    cr = (25, 80)
        return int(cr[1]), int(cr[0])

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
        if data.get_option('terminal').filter(lambda x: x.get('columns') < 0 or x.get('rows') < 0).is_defined():
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

        # If we are in paused mode or non-master mode, do not do anything
        if self.state_info.get('paused') is True:
            return

        # Get the progress bar with spaces
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
            f'{description_after or description_short_after}'

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
        bar_chars = data.get("bar_chars")

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
        data = self._get_bar_data_suffix() if data is None else data

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
        bar_suffix += f'[{item_per_second:.2f} it/s]'

        # update the output
        output_data = output_data.update({}, {
            "fraction": fractional_data,
            "item_per_sec": item_per_second,
        })

        # Time elapsed since the last update
        now = datetime.datetime.now()
        last_update_time = datetime.datetime.fromtimestamp(data.get('statistics.time.last_update_time'))
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

        # update the output
        output_data = output_data.update({}, {
            "time_since_last_update": {
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
                "microseconds": microseconds,
            }
        })

        # Time elapsed since the beginning of the iteration
        init_time = datetime.datetime.fromtimestamp(data.get('statistics.time.initial_progress_bar_time'))
        delta_time = now - init_time
        hours, remainder = divmod(delta_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        delta_time_str_since_iteration_beginning = f'{hours:02d}' \
                                         f':' \
                                         f'{minutes:02d}' \
                                         f':' \
                                         f'{seconds:02d}'

        # update the output
        output_data = output_data.update({}, {
            "time_since_beginning_iteration": {
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
            }
        })

        # Add the elapsed time to the bar_suffix
        bar_suffix += f' {delta_time_str_last_update} - {delta_time_str_since_iteration_beginning}'

        return bar_suffix, output_data

    def _get_bar_suffix(self) -> str:
        """Returns the stored string that comes after the bar.

        Returns
        -------
        A string that comes after the bar

        """

        # Retrieve and return
        return self.progress_bar_info.get('progress_bar.suffix')

    def _get_bar_data_suffix(self) -> DataMuncher:
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
        percent: float = data.get('item.current_item_index') / data.get('item.total_items_count')
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
        length_items = int(np.ceil(np.log10(data.get('item.total_items_count')))) if data.get(
            'item.total_items_count') > 0 else 5

        # Create the string
        fractional_progress: str = f'{data.get("item.current_item_index"): {length_items}d}'
        fractional_progress += f'/'
        fractional_progress += f'{data.get("item.total_items_count")}' if data.get(
            'item.total_items_count') > 0 else '?'

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

        return re.sub(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]', '', message)

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

    @abstractmethod
    def _notify_sleep(self):
        """Figure out if it is a good time to wake the sleeping time up!"""

        raise NotImplementedError

    @abstractmethod
    def _get_update_frequency(self) -> float:
        """Returns the number of times in a second that the progress bar should be updated.

        Returns
        -------
        The frequency at which the progress bar should be updated

        """

        raise NotImplementedError

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
                if not msg.endswith('\n\b') and self._check_if_should_print() is True:
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

    def __init__(self, stdout_handler=None):
        """Initializes the instance."""

        super().__init__(stdout_handler=stdout_handler)

    # action methods

    def activate(self) -> TheProgressBar:
        """Activates the progress bar: redirected stdout to this class and prints the progress bar

        Parameters
        ----------
        master_mode : bool, optional
            Whether this instance should behave in master mode.
                When master mode is on, this instance takes care of printing the progress bar and takes care of stdout.
                When master mode is off, this instance only updates its own state and can be accessed with the method
                    `get_progress_bar_string` to retrieve the string of the progress bar.

        Returns
        -------
        This instance

        """

        super().activate()

        # If not in single mode, no need to print, thus return now
        if self.state_info.get('role') != self.Roles.SINGLE:
            return self

        # Redirect stdout just in case there is no stdout handler from outside
        if self.state_info.get('external_stdout_handler') is False:
            sys.stdout = self
        else:
            self._activate_external_stdout_handler()

        # Hide cursor
        self._direct_write(self.cursor_modifier.get("hide"))

        # Get the running daemon thread
        self.run_thread.start()

        # Set the printing event on to start with the printing of the progress bar
        self.event_print.set()

        # finally, reset so that everything is set
        self.reset()

        return self

    def deactivate(self) -> None:
        """Deactivates the progress bar: redirected stdout to itself and closes the progress bar"""

        # Stop the run thread
        self.run_thread = None

        # Show cursor
        self._direct_write(self.cursor_modifier.get("show"))

        # Revert stdout back to its original place
        if self.state_info.get('external_stdout_handler') is False:
            sys.stdout = self.original_sysout
        else:
            self._deactivate_external_stdout_handler()

        # Print the progress bar and leave it
        self._print_progress_bar(return_to_line_number=-1)

    def pause(self) -> TheProgressBarBase:
        """Pauses the progress bar: redirected stdout to itself and stops prints the progress bar.

        This method permanently paused the instance. To resume run either the `resume` or `_look_to_resume` method.

        Returns
        -------
        This instance

        """

        super().pause()

        # if we are not printing at all, skip
        if self.state_info.get('role') != self.Roles.SINGLE:
            return self

        # Revert stdout back to its original place
        if self.state_info.get('external_stdout_handler') is False:
            sys.stdout = self.original_sysout
        else:
            self._pause_external_stdout_handler()

        # Show cursor
        sys.stdout.write(self.cursor_modifier.get("show"))
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

        super().resume()

        # if we are not printing at all, skip
        if self.state_info.get('role') != self.Roles.SINGLE:
            return self

        # Revert stdout back to its original place
        if self.state_info.get('external_stdout_handler') is False:
            sys.stdout = self
        else:
            self._resume_external_stdout_handler()

        # Hide cursor
        self._direct_write(self.cursor_modifier.get("hide"))

        # No longer look if we should resume
        self.check_for_resume_thread = None

        # Start printing
        self.event_print.set()

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

        # Print the progress bar and leave it if we have done any progress
        if self.state_info.get('item.current_item_index') != 0:
            self._print_progress_bar(return_to_line_number=return_to_line_number)

        # do the rest of resetting
        super().reset()

        return self

    def _configure_role(self):
        """Method to find out and set the role."""

        # find and update the role
        if self.state_info.get('parallel.size') > 1:
            if self.state_info.get('parallel.rank') == 0:
                role = self.Roles.MANAGER
            else:
                role = self.Roles.WORKER
            # update the role
            self.state_info = self.state_info.update({}, {'$set': {'role': role}})
        else:
            # leave it be single mode
            pass

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

    # utility methods

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
            # self.sleep_timer_info.get('pipe.write').send('timestup!')
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
        elif self.state_info.get('mode') == self.Modes.NOTTY:
            freq = 1 / 60  # if we do not have a tty, print every minute
        else:
            freq = float(np.clip(2 * average_freq, 2, 60))

        return freq


class TheProgressBarColored(TheProgressBar):

    def __init__(self, handler=None):

        super().__init__(handler)

    # TODO: Code duplication, fix it

    def _make_and_get_bar_prefix(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the string that comes before the bar.

        Parameters
        ----------
        data : DataMuncher, optional
            refer to super class

        Returns
        -------
        (str, DataMuncher)
            refer to super class

        """

        # Get the original one
        bar_prefix, bar_prefix_data = super()._make_and_get_bar_prefix()

        bar_prefix = f'{colored.fg("grey_74")}' \
                     f'{bar_prefix}' \
                     f'{colored.attr("reset")}'

        return bar_prefix, bar_prefix_data

    def _make_and_get_bar_suffix(self, data: DataMuncher = None) -> (str, DataMuncher):
        """Returns the string that comes after the bar.

        Parameters
        ----------
        data : DataMuncher, optional
            refer to super class

        Returns
        -------
        (str, DataMuncher)
            refer to super class

        """

        # get the prefix data
        data = self._get_bar_data_suffix() if data is None else data

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
        bar_suffix += f'{colored.fg("grey_74")}' \
                      f'[{item_per_second:.2f} it/s]'
        bar_suffix += f'{colored.attr("reset")}'

        # update the output
        output_data = output_data.update({}, {
            "fraction": fractional_data,
            "item_per_sec": item_per_second,
        })

        # Time elapsed since the last update
        now = datetime.datetime.now()
        last_update_time = datetime.datetime.fromtimestamp(data.get('statistics.time.last_update_time'))
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

        # Time elapsed since the beginning
        init_time = datetime.datetime.fromtimestamp(data.get('statistics.time.initial_progress_bar_time'))
        delta_time = now - init_time
        hours, remainder = divmod(delta_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        delta_time_str_since_beginning = f'{hours:02d}' \
                                         f':' \
                                         f'{minutes:02d}' \
                                         f':' \
                                         f'{seconds:02d}'

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
                      f'{delta_time_str_since_beginning}{colored.attr("reset")}'

        return bar_suffix, output_data

    def _get_fractional_progress(self, data: DataMuncher) -> (str, DataMuncher):
        """Returns a string of the form x*/y* where x* and y* are the current and total number of items.

        Parameters
        ----------
        data : DataMuncher
            refer to super class

        Returns
        -------
        (str, DataMuncher)
            A string containing the fractional progress
            data built

        """

        # Get the length of chars of total number of items for better formatting
        length_items = int(np.ceil(np.log10(data.get('item.total_items_count')))) if data.get(
            'item.total_items_count') > 0 else 5

        # Create the string
        fractional_progress: str = f'{colored.fg("gold_3b")}' \
                                   f'{data.get("item.current_item_index"): {length_items}d}'
        fractional_progress += f'{colored.fg("grey_46")}' \
                               f'/'
        fractional_progress += f'{colored.fg("orange_4b")}' + \
                               f'{data.get("item.total_items_count")}' if data.get('item.total_items_count') > 0 else '?'
        fractional_progress += f'{colored.attr("reset")}'

        # update the output data
        output_data = DataMuncher({
            "items": data.get("item.current_item_index"),
            "total_items": data.get("item.total_items_count") if data.get('item.total_items_count') > 0 else 0,
        })

        return fractional_progress, output_data

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
        terminal_size : (int, int), optional
            The given terminal size so that the method behaves according to this size, mainly used in non-master mode

        Returns
        -------
        A string containing the progress bar

        """

        # Get the original progress bar
        progress_bar = super()._make_and_get_progress_bar(
            data=data,
            include_desc_before=include_desc_before,
            include_desc_short_before=include_desc_short_before,
            include_prefix=include_prefix,
            include_bar=include_bar,
            include_suffix=include_suffix,
            include_desc_after=include_desc_after,
            include_desc_short_after=include_desc_short_after,
        )

        # Always reset the color back to normal
        progress_bar += f'{colored.attr("reset")}'

        return progress_bar
