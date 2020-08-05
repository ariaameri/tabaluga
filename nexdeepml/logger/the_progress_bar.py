from __future__ import annotations
import threading
import sys
from typing import List
import numpy as np
import time
import os


class TheProgressBar:
    """Progress bar that takes care of additional prints to the screen while updating.

    This implementation is based on the alive_progress package.
    """

    def __init__(self):
        """Initializes the instance."""

        # Create a lock so that one thread at a time can use the console to write
        self.print_lock = threading.Lock()

        # Create a threading.Event for knowing when to write the progress bar
        self.event = threading.Event()

        # A daemon thread placeholder for running the update of the progress bar
        self.run_thread: threading.Thread = None  # Setting it to None will cause the daemon process not to run

        # Get a CursorModifier that contains ANSI escape codes for the cursor
        self.cursor_modifier = self.CursorModifier()

        # Console controlling book keeping
        self.original_sysout = sys.stdout
        self.isatty = sys.stdout.isatty
        self.fileno = sys.stdout.fileno

        # The description to write at the end of the progress bar
        self.description: str = ''

        # The total number of items and the current item
        self.number_of_items: int = -1
        self.current_item: int = 0

        # A buffer for the messages to be printed
        self.buffer: List = []

        # Set of characters to be used for filling the bar
        self.bar_chars: str = '▏▎▍▌▋▊▉█'

        # Book keeping for calculating the elapsed time
        self.init_time: float
        self.last_update_time: float
        self.average_time_per_update: float = -np.inf
        self.average_item_per_update: float = -np.inf
        self.call_count: int = 0

    def __enter__(self) -> TheProgressBar:

        return self.activate()

    def __exit__(self, exc_type, exc_val, exc_tb):

        return self.deactivate()

    def activate(self) -> TheProgressBar:
        """Activates the progress bar: redirected stdout to this class and prints the progress bar

        Returns
        -------
        This instance

        """

        # Set the initial time
        self.init_time = self.last_update_time = time.time()

        # Redirect stdout
        sys.stdout = self

        # Hide cursor
        self._direct_write(self.cursor_modifier.get("hide"))

        # Get the running daemon thread
        self.run_thread = threading.Thread(
            name='run_daemon_thread',
            target=self.run,
            args=(),
            daemon=True
        )
        self.run_thread.start()

        return self

    def deactivate(self) -> None:
        """Deactivates the progress bar: redirected stdout to itself and closes the progress bar"""

        # Stop the run thread
        self.run_thread = None

        # Print the progress bar and leave it
        self._print_progressbar(return_to_beginning=False)

        # Show cursor
        self._direct_write(self.cursor_modifier.get("show"))

        sys.stdout = self.original_sysout

    def run(self) -> None:
        """Prints the progress bar and takes care of other controls.

        Method to be run by the daemon progress bar thread.
        """

        while self.run_thread:

            # Print the progress bar only if it is focused on
            if self._check_if_focused():
                self._print_progressbar()

            time.sleep(1 / self._get_update_frequency())

    def _check_if_focused(self) -> bool:
        """Checks whether the terminal is focused on the progress bar so that it should be printed.

        Returns
        -------
        A boolean stating whether or not the progress bar should be printed

        """

        # Check if we are connected to a terminal
        # Check if we are a foreground process
        check = self.isatty() \
            and (os.getpgrp() == os.tcgetpgrp(self.original_sysout.fileno()))

        return check

    def set_number_items(self, number_of_items: int) -> TheProgressBar:
        """Set the total number of the items.

        Parameters
        ----------
        number_of_items : int
            The total number of items

        Returns
        -------
        This instance

        """

        self.number_of_items = number_of_items

        return self

    def reset(self) -> TheProgressBar:
        """Resets the progress bar and returns its instance.

        Returns
        -------
        This instance

        """

        # Print the progress bar and leave it
        self._print_progressbar(return_to_beginning=False)

        # Set the initial time
        self.init_time = self.last_update_time = time.time()

        # Reset the current item counter
        self.current_item = 0

        return self

    def update(self, count: int) -> None:
        """Updates the progress bar by adding 'count' to the number of items.

        Parameter
        ---------
        count : int
            The count at which the progress items should be increased. It has to be non-negative

        """

        if count >= 0:
            # Update current item
            self.current_item += count

            # Keep track of an average number of elements in each update
            self.average_item_per_update = self._exp_average(self.average_item_per_update, count)

            # Update the time
            self._update_time_counter()

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

            esc_sequence = self.cursor_dict.get(item, '')

            if type(esc_sequence) is list:
                esc_sequence = str(add).join(esc_sequence)

            return esc_sequence

    def _exp_average(self, item: float, d_item: float, beta: float = .9) -> float:
        """Calculates the new exponential moving average for the inputs.

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

    def set_description(self, description: str) -> None:
        """Sets the description of the progress bar.

        Parameters
        ----------
        description : str
            String to update the description for the progress bar

        """

        self.description = description

    def _get_progressbar_with_spaces(self, return_to_beginning: bool = True) -> str:
        """Retunrs the progress bar along with its cursor modifier ANSI escape codes

        Returns
        -------
        return_to_beginning: bool, optional
            Whether to return the cursor to the beginning of the progress bar

        """

        # Get the progress bar
        progress_bar = self._get_progressbar()

        # Clear the line and write it
        progress_bar_with_space: str = self.cursor_modifier.get('clear_line')
        progress_bar_with_space += f'{progress_bar}'

        if return_to_beginning:
            number_of_lines = progress_bar.count(f'\n')
            progress_bar_with_space += self.cursor_modifier.get('up', number_of_lines) if number_of_lines != 0 else ''
            progress_bar_with_space += f'\r'
        else:
            progress_bar_with_space += f'\n'

        return progress_bar_with_space

    def _print_progressbar(self, return_to_beginning: bool = True) -> None:
        """Clears the line and prints the progress bar

        Returns
        -------
        return_to_beginning: bool, optional
            Whether to return the cursor to the beginning of the progress bar

        """

        # Get the progress bar with spaces
        progress_bar = self._get_progressbar_with_spaces(return_to_beginning=return_to_beginning)

        # Print the progress bar
        self._direct_write(progress_bar)

    def _get_progressbar(self) -> str:
        """Returns a string containing the progress bar.

        Returns
        -------
        A string containing the progress bar

        """

        # Get console's width and height
        columns, rows = self._get_terminal_size()

        bar_prefix = self._get_bar_prefix()
        bar_suffix = self._get_bar_suffix()

        remaining_columns = \
            int(np.clip(
                columns - len(bar_prefix) - len(bar_suffix) - 3 - len(self.description.split('\n')[0]),
                30,
                50
            ))

        bar = self._get_bar(remaining_columns)

        progress_bar = f'{bar_prefix} {bar} {bar_suffix} {self.description}'

        # Trim the progress bar to the number of columns of the console
        progress_bar = progress_bar[:columns]

        return progress_bar

    def _get_terminal_size(self) -> (int, int):

        env = os.environ

        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termios, struct, os
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

    def _get_bar(self, length: int) -> str:
        """Returns a string containing the bar itself.

        Parameters
        ----------
        length : int
            The length of the bar

        Returns
        -------
        A string containing the bar

        """

        # The percentage of the progress
        percent: float = self._get_percentage()

        # Get the length of the bar without the borders
        bar_length = length - 2

        # Figure how many 'complete' bar characters (of index -1) we need
        # Figure what other character of bar characters is needed
        virtual_length = bar_length * len(self.bar_chars)
        whole_char_count, remainder_char_idx = divmod(int(virtual_length * percent), len(self.bar_chars))

        # Make the bar string
        bar: str = self.bar_chars[-1] * whole_char_count  # Completed parts
        bar += self.bar_chars[remainder_char_idx - 1] if remainder_char_idx != 0 else ''  # Half-completed parts
        bar += ' ' * (bar_length - len(bar))  # Not completed parts

        # Add the borders
        bar = f'|{bar}|'

        return bar

    def _get_bar_prefix(self) -> str:
        """Returns the string that comes before the bar.

        Returns
        -------
        A string that comes before the bar

        """

        # The percentage of the progress
        percent: float = self._get_percentage() * 100

        bar_prefix = f'{percent:6.2f}%'

        return bar_prefix

    def _get_percentage(self) -> float:
        """Returns the percentage of the process.

        Returns
        -------
        A float that is the percentage of the process

        """

        # The percentage of the progress
        percent: float = self.current_item / self.number_of_items
        percent = float(np.clip(percent, 0., 1.))

        return percent

    def _get_bar_suffix(self) -> str:
        """Returns the string that comes after the bar.

        Returns
        -------
        A string that comes after the bar

        """

        bar_suffix: str = self._get_fractional_progress()  # Fractional progress e.g. 12/20
        bar_suffix += f' '
        bar_suffix += f'[{self._get_item_per_second():.2f} it/s]'

        return bar_suffix

    def _get_fractional_progress(self) -> str:
        """Returns a string of the form x*/y* where x* and y* are the current and total number of items.

        Returns
        -------
        A string containing the fractional progress

        """

        # Get the length of chars of total number of items for better formatting
        length_items = int(np.ceil(np.log10(self.number_of_items))) if self.number_of_items > 0 else 5

        # Create the string
        fractional_progress: str = f'{self.current_item: {length_items}d}'
        fractional_progress += f'/'
        fractional_progress += f'{self.number_of_items}' if self.number_of_items > 0 else '?'

        return fractional_progress

    def _get_item_per_second(self) -> float:
        """Returns the average number of items processed in a second.

        Returns
        -------
        Average number of items processed in a second

        """

        average_item_per_second = self.average_item_per_update / self.average_time_per_update

        return average_item_per_second

    def _update_time_counter(self) -> None:
        """Updates the internal average time counter."""

        # Figure current item time
        delta_time = time.time() - self.last_update_time

        # Update the moving average
        self.average_time_per_update = self._exp_average(self.average_time_per_update, delta_time)

        # Update the last time
        self.last_update_time = time.time()

    def _get_update_frequency(self) -> float:
        """Returns the number of times in a second that the progress bar should be updated.

        Returns
        -------
        The frequency at which the progress bar should be updated

        """

        # Calculated the average number of items processed per second
        average_freq: float = 1 / self.average_time_per_update

        # Rule: update at least 2 times and at most 60 times in a second unless needed to be faster
        # Also be twice as fast as the update time difference
        freq = float(np.clip(2 * average_freq, 2, 60))

        return freq

    def _direct_write(self, msg: str) -> None:
        """Write the msg directly on the output with no buffers.

        Parameters
        ----------
        msg : str
            Message to be written

        """

        with self.print_lock:
            self.original_sysout.write(msg)
            self.flush()

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
        if msg == '\n':

            with self.print_lock:

                # Add the progress bar at the end
                self.buffer.append(self._get_progressbar_with_spaces())

                # Create the message from the buffer and print it with extra new line character
                msg = ''.join(self.buffer)

                self.original_sysout.write(self.cursor_modifier.get('clear_until_end'))
                self.original_sysout.write(f'{msg}')

                self.buffer = []

    def flush(self) -> None:
        """Flushes the screen."""

        self.original_sysout.flush()
