import threading
import sys
from typing import List


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

        # The description to write at the end of the progress bar
        self.description: str = ''

        # The total number of iterations and the current iteration
        self.number_of_iterations: int = -1
        self.current_iteration: int = 0

        # A buffer for the messages to be printed
        self.buffer: List = []

    def __enter__(self):

        return self.activate()

    def __exit__(self, exc_type, exc_val, exc_tb):

        return self.deactivate()

    def activate(self):
        """Activates the progress bar: redirected stdout to this class and prints the progress bar

        Returns
        -------
        This instance

        """

        sys.stdout = self
        return self

    def deactivate(self):
        """Deactivates the progress bar: redirected stdout to itself and closes the progress bar"""

        sys.stdout = sys.__stdout__

    def set_number_iterations(self, number_of_iterations: int):
        """Set the total number of the iterations.

        Parameters
        ----------
        number_of_iterations : int
            The total number of iterations

        Returns
        -------
        This instance

        """

        self.number_of_iterations = number_of_iterations

        return self

    def reset(self):
        """Resets the progress bar and returns its instance.

        Returns
        -------
        This instance

        """

        pass

    def update(self, count: int) -> None:
        """Updates the progress bar by adding 'count' to the number of iterations.

        Parameter
        ---------
        count : int
            The count at which the progress iterations should be increased

        """

        self.current_iteration += count

    def set_description(self, description: str) -> None:
        """Sets the description of the progress bar.

        Parameters
        ----------
        description : str
            String to update the description for the progress bar

        """

        self.description = description

    def _print_progressbar(self):
        """Prints the progress bar"""

        pass

    def _get_progressbar(self) -> str:
        """Returns a string containing the progress bar.

        Returns
        -------
        A string containing the progress bar

        """

        pass

    def _get_bar(self) -> str:
        """Returns a string containing the bar itself.

        Returns
        -------
        A string containing the bar

        """

        pass

    def _update_frequency(self) -> float:
        """Returns the number of times in a second that the progress bar should be updated.

        Returns
        -------
        The frequency at which the progress bar should be updated

        """

        pass

    def write(self, msg: str) -> None:
        """Prints a message to the output.

        Parameters
        ----------
        msg : str
            The message to be written

        """

        # Calling 'print' will call this function twice.
        # First with the message and second with the new line character
        if msg != f'\n':
            self.buffer.append(msg)
        else:
            with self.print_lock:
                # Create the message from the buffer and print it with extra new line character
                msg = ''.join(msg for msg in self.buffer)

                sys.__stdout__.write(self.cursor_modifier.get('clear_until_end'))
                sys.__stdout__.write(f'{msg}\n')

                self.buffer = []

    def flush(self):
        """Flushes the screen."""

        sys.__stdout__.flush()
