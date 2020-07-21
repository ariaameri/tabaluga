from typing import Dict
import logging
import sys
from tqdm import tqdm
import numpy as np
import io


class Logger:
    """An abstract base/parent class for all logger classes."""

    # Keep track of how many logger instances we have
    _counter = [0]

    # TODO: Figure out the way configurations have to be passed to the class

    def __init__(self, config):
        """Initializes the logger class

        Parameters
        ----------
        config
            The configuration for the logger class
            It should consist of the following attributes
                name : str
                    The name of the logger
                level : {logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
                    The level at which logging should happen.
                format : str
                    The format at which the logger should log
                console : boolean
                    Whether or not to log to console. If false, will write to file whose path is provided by filename
                filename : str
                    The path of the file to write the log file to. Can be omitted if console is False

        """

        # Save the config
        self._config = config

        # The level at which we log
        self._level = config.level

        # Get the logger
        self._logger = logging.getLogger(config.name)  # TODO: For later: check if the name already exists
        self._counter[0] += 1
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Suppress _logger output to stdout

        # TODO: Should we have logging to both the console and the file?
        # Determine whether to write to file or console and get the handlers
        if config.console is True:
            self._handler = logging.StreamHandler(sys.stdout)
        else:
            self._handler = logging.FileHandler(config.filename)

        # Set the level, format, and attach
        self._handler.setLevel(config.level)
        self._handler.setFormatter(logging.Formatter(config.format))
        self._logger.addHandler(self._handler)

    def info(self, msg: str) -> None:
        """Writes the message given as an info.

        Parameters
        ----------
        msg : str
            The message to be written as info

        """

        self._logger.info(msg)

    def error(self, msg: str) -> None:
        """Writes the message given as an error.

        Parameters
        ----------
        msg : str
            The message to be written as error

        """

        self._logger.error(msg)

    def debug(self, msg: str) -> None:
        """Writes the message given as an debug.

        Parameters
        ----------
        msg : str
            The message to be written as debug

        """

        self._logger.debug(msg)

    def set_level(self, level: int) -> None:
        """Sets the level of logging.

        Parameters
        ----------
        level : {logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
            The level at which logging should happen.

        """

        self._level = level
        self._handler.level = level


class TQDMLogger(Logger, io.StringIO):

    # TODO: Write the doc for the config argument

    def __init__(self, config):

        # Making sure the logger is going to write to the console
        config.console = True

        Logger.__init__(self, config)
        io.StringIO.__init__(self)

        self._tqdm = tqdm(
            position=self._counter[0],
            bar_format='{percentage:3.0f}%|{bar}{r_bar} {desc}',
            file=self  # Write to this log handler instead of stderr
        )
        # Bookkeeping for tqdm
        self.buf: str = ''

        # The number of total iterations
        self._total: int = -1

        # The number of total epochs
        self._n_epochs: int = config.n_epochs

    def set_total(self, total: int) -> None:
        """Set the total number of iterations and resets the tqdm.

        Also, the number is updated in self._config.total

        Parameters
        ----------
        total : int
            the total number of iterations expected."""

        self._tqdm.reset(total=total)
        self._total = total
        self._config.total = total

    def close(self) -> None:
        """Finishes and closes the tqdm instance."""

        self._tqdm.close()

    def update(self, update_count: int, msg=None) -> None:
        """Update the tqdm progress bar with description set to message.

        Parameters
        ----------
        update_count : int
            The amount that should be added to the tqdm instance.
        msg : Any, optional
            Contains the message to be set as the progress bar description.
            Will be passed to the _generate_message method.
        """

        self._tqdm.update(update_count)

        message = self._generate_message(msg)
        self._tqdm.set_description(message)

    def _generate_message(self, msg: Dict) -> str:
        """Generates a string based on the input to be used as tqdm bar description.

        If msg is None, empty string will be returned.

        Parameters
        ---------
        msg : Dict
            Dictionary containing the information to be used. Contains:
                epoch: int
                loss: float
                val_loss: float, optional
        """

        message = ''

        if msg is None:
            return message

        # Find the length of the total epochs
        # and reformat the string accordingly
        ep_len = int(np.ceil(np.log10(self._n_epochs)))
        message += f'Epoch {msg["epoch"]:{ep_len}d}/{self._n_epochs}: '

        message += f'loss: {msg["loss"]: .5e}'

        if 'val_loss' in msg.keys():
            message += f', val_loss: {msg["val_loss"]: .5e}'

        # Print the rest in msg
        for key in [key for key in msg.keys() if key != 'epoch' and key != 'loss' and key != 'val_loss']:
            message += f', {key}: {msg[key]}'

        message += ' '

        return message

    def write(self, buf: str) -> None:
        """For tqdm to write to the buffer."""

        self.buf = buf.strip('\r\n\t ')

    def flush(self) -> None:
        """For tqdm.

        Will write tqdm messages as infos."""

        self._logger.info(self.buf)
