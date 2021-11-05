import os
import sys


def check_terminal_focused() -> bool:
    """
    Returns whether or not the terminal is focused.

    Returns
    -------
    bool
        True if we are focused and False otherwise

    """

    # check if we have a tty
    result = check_terminal_atty()
    # check if we are on the foreground
    result &= os.getpgrp() == os.tcgetpgrp(sys.stdout.fileno())

    return result


def check_terminal_atty() -> bool:
    """
    Returns whether or not the terminal is has a tty.

    Returns
    -------
    bool
        True if we have tty and False otherwise

    """

    # check if we have a tty
    result = sys.stdout.isatty()

    return result
