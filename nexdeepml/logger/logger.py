from ..util.config import ConfigParser
from ..base.base import BaseWorker, BaseEventManager
from typing import Dict, List
import logging
import sys
from tqdm import tqdm
import numpy as np
import io
from datetime import datetime


class Logger(BaseWorker):
    """An abstract base/parent class for all logger classes."""

    # Keep track of how many logger instances we have
    _counter: List[int] = [0]

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

        # Save the config
        self._config = config

        # The level at which we log
        self._level: int = config.level if config.level is not None else logging.INFO

        # Get the logger
        self._logger = logging.getLogger(config.name if config.name is not None else self._counter[0])
        self._counter[0] += 1
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Suppress _logger output to stdout

        # TODO: Should we have logging to both the console and the file?
        # Determine whether to write to file or console and get the handlers
        self.console = config.console
        if self.console is True:
            self._handler = logging.StreamHandler(sys.stdout)
        else:
            file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f'{file_name}.txt'
            self._handler = logging.FileHandler(config.file_name if config.file_name is not None else file_name)

        # Set the level, format, and attach
        self._handler.setLevel(config.level if config.level is not None else logging.INFO)
        self._handler.setFormatter(
            logging.Formatter(
                config.format if config.format is not None else '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
            ))
        self._logger.addHandler(self._handler)

        super().__init__()

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

    def __init__(self, config: ConfigParser):

        # Making sure the logger is going to write to the console
        assert config.console is True, "console config is not True for TQDM!"

        self._config = config

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
        self._n_epochs: int = self._config.n_epochs

    def reset(self, total: int) -> None:
        """Set the total number of iterations and resets the tqdm.

        Also, the number is updated in self._config.total

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
        self._tqdm.set_description(message)

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

        message = ''

        if msg_dict is None:
            return message

        # Find the length of the total epochs
        # and reformat the string accordingly
        ep_len = int(np.ceil(np.log10(self._n_epochs)))
        message += f'Epoch {msg_dict["epoch"]:{ep_len}d}/{self._n_epochs}: '

        message += f'loss: {msg_dict["loss"]: .5e}'

        if 'val_loss' in msg_dict.keys():
            message += f', val_loss: {msg_dict["val_loss"]: .5e}'

        # Print the rest in msg
        for key in [key for key in msg_dict.keys() if key != 'epoch' and key != 'loss' and key != 'val_loss']:
            message += f', {key}: {msg_dict[key]}'

        message += ' '

        return message

    def write(self, buf: str) -> None:
        """For tqdm to write to the buffer."""

        self.buf = buf.strip('\r\n\t ')

    def flush(self) -> None:
        """For tqdm.

        Will write tqdm messages as infos."""

        self._logger.info(self.buf)
