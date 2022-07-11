import os
import pathlib
import subprocess
import sys
import re
from enum import Enum

REGEX_INDENT: re.Pattern = re.compile(r'(^|\n)')
REGEX_INDENT_NEW_LINE_ONLY: re.Pattern = re.compile(r'(\n)')


class EventMode(Enum):

    train: str = "train"
    validation: str = "validation"
    test: str = "test"


class EventTime(Enum):
    EpochBegin: str = 'epoch_begin'
    EpochEnd: str = 'epoch_end'
    Begin: str = 'begin'
    End: str = 'end'
    TrainBegin: str = 'train_begin'
    TrainEnd: str = 'train_end'
    TrainEpochBegin: str = 'train_epoch_begin'
    TrainEpochEnd: str = 'train_epoch_end'
    BatchBegin: str = 'batch_begin'
    BatchEnd: str = 'batch_end'
    TrainBatchBegin: str = 'train_batch_begin'
    TrainBatchEnd: str = 'train_batch_end'
    ValBegin: str = 'val_begin'
    ValEnd: str = 'val_end'
    ValEpochBegin: str = 'val_epoch_begin'
    ValEpochEnd: str = 'val_epoch_end'
    ValBatchBegin: str = 'val_batch_begin'
    ValBatchEnd: str = 'val_batch_end'
    TestBegin: str = 'test_begin'
    TestEnd: str = 'test_end'
    TestBatchBegin: str = 'test_batch_begin'
    TestBatchEnd: str = 'test_batch_end'
    TestEpochBegin: str = 'test_epoch_begin'
    TestEpochEnd: str = 'test_epoch_end'


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
    result &= check_terminal_foreground()

    return result


def check_terminal_foreground() -> bool:
    """
    Returns whether or not the terminal is in foreground.

    Returns
    -------
    bool
        True if we are foreground and False otherwise

    """

    try:
        # check
        result = os.getpgrp() == os.tcgetpgrp(sys.stdout.fileno())
    except:
        return False

    return result


def check_terminal_atty() -> bool:
    """
    Returns whether or not the terminal is has a tty.

    Returns
    -------
    bool
        True if we have tty and False otherwise

    """

    try:
        # check if we have a tty
        result = sys.stdout.isatty()
    except:
        return False

    return result


def get_tty_fd() -> pathlib.Path:
    """
    Returns the tty file descriptor.

    Returns
    -------
    pathlib.Path
        file descriptor path
    """

    try:
        return pathlib.Path(os.ttyname(sys.stdout.fileno()))
    except:
        out = subprocess.check_output(
                ['readlink', '-f', f'/proc/{os.getpid()}/fd/1']
                ).decode('utf-8').strip()
        return pathlib.Path(out)
