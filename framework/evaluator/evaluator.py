from ..base import base
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..model.model import ModelManager, Model
from ..logger.logger import Logger
from ..logger.log_hug import LogHug
from typing import Union, List, Dict, Type
from abc import ABC, abstractmethod
import signal
import sys
import os
import traceback


class Evaluator(base.BaseEventManager, ABC):
    """A class to help with training a neural network."""

    def __init__(self, config: ConfigParser = None):
        """Initializer for the this instance of the class"""

        # initialize mpi
        from ..communicator import mpi
        mpi.init(config.get_or_empty("mpi"))

        # initialize rabbitmq if exist
        if mpi.mpi_communicator.is_distributed() is True:
            from ..communicator import rabbitmq
            rabbitmq.init(config.get_or_else("rabbitmq", ConfigParser()))

        super().__init__(config)

        # initialize the console handler
        self._console_handler.activate()

        # Total number of epochs, total batch count, batch size, current epoch, and current batch number
        self.epochs: int = self._config.get('epochs')
        self.number_of_iterations: int = -1  # Has to be set by the data loader
        self.batch_size: int = -1  # Has to be set by the data loader
        self.epoch: int = 0
        self.batch: int = 0

        # Set placeholders for the train and validation data
        self.data: DataMuncher = DataMuncher()

        # Set placeholder for model
        self.model: ModelManager = self.load_model()

        # Create history list for keeping the history of the net
        self.history = []
        # Make dummy variables
        self.train_info_dict = {}
        self.val_info_dict = {}
        self.train_statistics = LogHug()

        # Set the universal logger
        self._universal_logger = self._create_universal_logger()
        self.set_universal_logger(self._universal_logger)

        # Register OS signals to be caught
        self._register_signal_catch()

        # Register exception hook to be caught
        self._register_exception_hook()

    def load_model(self) -> Union[ModelManager, Model]:
        """Creates an instance of the model, loads weights and returns it."""

        pass

    def _create_universal_logger(self) -> Logger:
        """Creates a universal logger instance and returns it."""

        logger = Logger(self._config.get('universal_logger'))

        return logger

    def evaluation(self) -> List[Dict]:
        """Performs the training and validation.

        Returns
        -------
        A List[Dict] containing the history of the train/validation process

        """

        self.epoch = 0

        # Everything is beginning
        self.on_begin()

        while self.epoch < self.epochs:

            # Epoch has started
            self.on_epoch_begin()

            # Do one epoch
            epoch_history = self.one_epoch()

            # Epoch has ended
            self.on_epoch_end()

            # This is the end of the epoch, so, epoch number is incremented
            # Also, history is recorder
            self.epoch += 1
            self.history.append(epoch_history)

        # Everything is finished
        self.on_end()

        return self.history

    def one_epoch(self) -> Dict:
        """Performs the training and validation for one epoch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        # Empty out the train statistics
        self.train_statistics = LogHug()

        # Training
        if self.epoch == 0:
            self.on_train_begin()

        self.on_train_epoch_begin()
        train_info_dict = self.train_one_epoch()
        self.on_train_epoch_end()

        if self.epoch == (self.epochs - 1):
            self.on_train_end()

        # Validation
        if self.epoch == 0:
            self.on_val_begin()

        self.on_val_epoch_begin()
        val_info_dict = self.val_one_epoch()
        self.on_val_epoch_end()

        if self.epoch == (self.epochs - 1):
            self.on_val_end()

        epoch_dict = {
            'epoch': self.epoch,
            **train_info_dict,
            **val_info_dict
        }

        return epoch_dict

    def predict_one_epoch(self) -> Dict:
        """Trains the neural network for one epoch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        # Make Train entry
        self.train_statistics = self.train_statistics.update({}, {'$set': {'Train': {}}})

        for self.batch in range(self.number_of_iterations):

            self.on_batch_begin()
            self.on_train_batch_begin()

            self.train_info_dict = self.train_one_batch()
            self.train_statistics = \
                self.train_statistics.update(
                    {'_bc': {'$regex': 'Train'}},
                    {'$set': self.train_info_dict},
                    {'recursive': True}
                )

            self.on_train_batch_end()
            self.on_batch_end()

        return self.train_info_dict

    @abstractmethod
    def predict_one_batch(self) -> Dict:
        """Trains the neural network for one batch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        raise NotImplementedError

    def signal_catcher(self, os_signal, frame):
        """Catches an OS signal and calls it on its workers."""

        # Take care of signals
        if os_signal == signal.SIGINT:
            info = {'signal': os_signal}
            self.on_os_signal(info)
            self._universal_log('Interrupt signal received, exiting...', 'error')
            sys.exit(1)
        elif os_signal == signal.SIGTERM:
            info = {'signal': os_signal}
            self.on_os_signal(info)
            self._universal_log('Termination signal received, exiting...', 'error')
            sys.exit(0)
        elif os_signal == signal.SIGTSTP:
            info = {'signal': os_signal}
            self.on_os_signal(info)
            self._universal_log('Terminal stop signal received.', 'warning')
            signal.signal(os_signal, signal.SIG_DFL)
            os.kill(os.getpid(), os_signal)
        elif os_signal == signal.SIGCONT:
            info = {'signal': os_signal}
            self.on_os_signal(info)
            self._universal_log('Continue signal received.', 'info')

    def _register_signal_catch(self):
        """Registers what signals should be caught by this instance."""

        signal.signal(signal.SIGINT, self.signal_catcher)
        signal.signal(signal.SIGTERM, self.signal_catcher)
        signal.signal(signal.SIGTSTP, self.signal_catcher)
        signal.signal(signal.SIGCONT, self.signal_catcher)

    def exception_hook(self, type, value, tracebacks: traceback):
        """Method to be called when exception happens"""

        # assuming the error does not happen in universal log!!!
        self._universal_log("Received an exception", level='error')
        self._universal_log(f"Exception type: {type}", level='error')
        self._universal_log(f"Exception value: {value}", level='error')
        exception = "".join(traceback.format_tb(tracebacks))
        self._universal_log(f"Exception traceback: \n{exception}", level='error')

        # now, terminate ourselves!
        self.terminate()

    def _register_exception_hook(self):
        """Registers the global exception hook."""

        sys.excepthook = self.exception_hook

    def __del__(self):
        """Method to be called when the object is being deleted."""

        # deactivate the console handler
        self._console_handler.deactivate()
