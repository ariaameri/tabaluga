import logging


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
        self._logger = logging.getLogger(config.name) # TODO: For later: check if the name already exists
        self._counter[0] += 1
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Suppress _logger output to stdout

        # TODO: Should we have logging to both the console and the file?
        # Determine whether to write to file or console and get the handlers
        if config.console is True:
            self._handler = logging.StreamHandler()
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
