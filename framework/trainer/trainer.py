from ..base import base
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.data_muncher import UPDATE_MODIFIERS as UM, UPDATE_OPERATIONS as UO, UPDATE_CONDITIONALS as UC
from ..util.data_muncher import FILTER_OPERATIONS as FO, FILTER_MODIFIERS as FM
from ..callback.callback import CallbackManager, Callback
from ..model.model import ModelManager, Model
from ..logger.logger import Logger
from typing import Union, List, Dict, Type
from abc import ABC, abstractmethod
import signal
import sys
import os
import traceback

BATCH_STRING = 'batch'
EPOCH_STRING = 'batch'


class Trainer(base.BaseEventManager, ABC):
    """A class to help with training a neural network."""

    def __init__(self, config: ConfigParser = None):
        """Initializer for this instance of the class"""

        # set communicators' configs
        self._set_communicator_configs(config)

        # set async config
        from ..asyncer import config as async_config
        async_config.asyncer_config = (config or ConfigParser()).get_or_empty('asyncer')

        # initialize mpi
        # from ..communicator import mpi
        # if config.contains_key("mpi"):
        #     # reinitialize mpi
        #     mpi.mpi_communicator = mpi.init_with_config(config.get_or_empty("mpi"))

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
        self.model: ModelManager = self.create_model()

        # Create history list for keeping the history of the net
        self.history = []  # do not use this! the content of this variable can change in the future
        # Make dummy variables
        self.train_batch_info: DataMuncher = DataMuncher()  # keep the current batch info
        self.val_batch_info: DataMuncher = DataMuncher()  # keep the current batch info
        self.test_batch_info: DataMuncher = DataMuncher()  # keep the current batch info
        self.train_epoch_info: List[DataMuncher] = []  # keep the current epoch info
        self.val_epoch_info: List[DataMuncher] = []  # keep the current epoch info
        self.test_epoch_info: List[DataMuncher] = []  # keep the current epoch info
        self.train_life_info: List[List[DataMuncher]] = []  # keep info across epochs for the whole thing
        self.val_life_info: List[List[DataMuncher]] = []  # keep info across epochs for the whole thing
        self.test_life_info: List[List[DataMuncher]] = []  # keep info across epochs for the whole thing
        self.train_current_statistics = DataMuncher()  # current statistics/info

        # Set the universal logger
        self._universal_logger = self._create_universal_logger()
        self.set_universal_logger(self._universal_logger)

        # Register OS signals to be caught
        self._register_signal_catch()

        # Register exception hook to be caught
        self._register_exception_hook()

    def _set_communicator_configs(self, config: ConfigParser):
        """Sets the configurations for the communicators."""

        from ..communicator import config as comm_config

        comm_config.mpi_config = config.get_or_empty("mpi")
        comm_config.rabbit_config = config.get_or_empty("rabbitmq")
        comm_config.mongo_config = config.get_or_empty('mongodb')
        comm_config.influx_config = config.get_or_empty('influxdb')
        comm_config.zmq_config = config.get_or_empty('zmq_internal')

    def create_model(self) -> Union[ModelManager, Model]:
        """Creates an instance of the model and returns it."""

        pass

    def _create_universal_logger(self) -> Logger:
        """Creates a universal logger instance and returns it."""

        logger = Logger(self._config.get('universal_logger'))

        return logger

    def train_and_test(self) -> List[DataMuncher]:
        """Performs the training, validation, and testing.

        Returns
        -------
        A List[DataMuncher] containing the history of the train/validation/testing process

        """

        self.epoch = 0

        # Everything is beginning
        self.on_begin()

        # do the training
        self.train()

        # do the testing
        self.test()

        # Everything is finished
        self.on_end()

        return self.history

    def train(self) -> List[DataMuncher]:
        """Performs the training and validation.

        Returns
        -------
        A List[DataMuncher] containing the history of the train/validation process

        """

        while self.epoch < self.epochs:

            # Epoch has started
            self.on_epoch_begin()

            # Do one epoch
            epoch_history = self.one_epoch()

            # Epoch has ended
            self.on_epoch_end()

            # This is the end of the epoch, so, epoch number is incremented
            self.epoch += 1

            # Bookkeeping
            self.history.append(epoch_history)

        return self.history

    def one_epoch(self) -> DataMuncher:
        """Performs the training and validation for one epoch.

        Returns
        -------
        A DataMuncher containing the history of the process

        """

        # Empty out the train statistics
        self.train_current_statistics = DataMuncher()

        # Training
        if self.epoch == 0:
            self.on_train_begin()

        self.on_train_epoch_begin()
        self.train_epoch_info: List[DataMuncher] = self.train_one_epoch()
        # bookkeeping
        self.train_life_info.append(self.train_epoch_info)
        self.on_train_epoch_end()

        if self.epoch == (self.epochs - 1):
            self.on_train_end()

        # Validation
        if self.epoch == 0:
            self.on_val_begin()

        self.on_val_epoch_begin()
        self.val_epoch_info: List[DataMuncher] = self.val_one_epoch()
        # bookkeeping
        self.val_life_info.append(self.val_epoch_info)
        self.on_val_epoch_end()

        if self.epoch == (self.epochs - 1):
            self.on_val_end()

        epoch_info = DataMuncher({
            'epoch': self.epoch,
            'train': self.train_epoch_info,
            'validation': self.val_epoch_info,
        })

        return epoch_info

    def train_one_epoch(self) -> List[DataMuncher]:
        """Trains the neural network for one epoch.

        Returns
        -------
        A list of DataMuncher containing the history of the process

        """

        # Make Train entry
        self.train_current_statistics = self.train_current_statistics.update({}, {UO.SET: {'Train': {}}})

        # empty out the epoch info
        self.train_epoch_info = []

        for self.batch in range(self.number_of_iterations):

            self.on_batch_begin()
            self.on_train_batch_begin()

            # train on batch
            self.train_batch_info: DataMuncher = self.train_one_batch()

            # keep the result
            # decided to update the train epoch info incrementally in case it was needed
            self.train_epoch_info.append(
                # add additional info
                self.train_batch_info.update(
                    {},
                    {
                        UO.SET_ONLY: {
                            EPOCH_STRING: self.epoch,
                            BATCH_STRING: self.batch,
                        }
                    }
                )
            )
            self.train_current_statistics = \
                self.train_current_statistics.update(
                    {'Train': {FO.EXISTS: 1}},
                    {UO.SET: {'Train': self.train_batch_info}},
                )

            self.on_train_batch_end()
            self.on_batch_end()

        return self.train_epoch_info

    def val_one_epoch(self) -> List[DataMuncher]:
        """Performs validation for the neural network for one epoch.

        Returns
        -------
        A list of DataMuncher containing the history of the process

        """

        # Make Validation entry
        self.train_current_statistics = self.train_current_statistics.update({}, {UO.SET: {'Validation': {}}})

        # empty out the epoch info
        self.val_epoch_info = []

        for self.batch in range(self.number_of_iterations):

            self.on_val_batch_begin()

            # validate on batch
            self.val_batch_info: DataMuncher = self.val_one_batch()

            # keep the result
            # decided to update the validation epoch info incrementally in case it was needed
            self.val_epoch_info.append(
                # add additional info
                self.val_batch_info.update(
                    {},
                    {
                        UO.SET_ONLY: {
                            EPOCH_STRING: self.epoch,
                            BATCH_STRING: self.batch,
                        }
                    }
                )
            )
            self.train_current_statistics = \
                self.train_current_statistics.update(
                    {'Validation': {FO.EXISTS: 1}},
                    {UO.SET: {'Validation': self.val_batch_info}},
                )

            self.on_val_batch_end()

        return self.val_epoch_info

    def test(self) -> List[DataMuncher]:
        """ Performs the testing.

        Returns
        -------
        A list of DataMuncher containing the history of the process

        """

        # test is beginning
        self.on_test_begin()

        # the only test epoch is beginning
        self.on_test_epoch_begin()

        # do one epoch of test
        self.test_epoch_info: List[DataMuncher] = self.test_one_epoch()

        # bookkeeping
        self.test_life_info.append(self.test_epoch_info)
        epoch_info = DataMuncher({
            'epoch': self.epoch,
            'test': self.test_epoch_info,
        })
        self.history.append(epoch_info)

        # the only test epoch is beginning
        self.on_test_epoch_end()

        # test is done
        self.on_test_end()

        return self.history

    def test_one_epoch(self) -> List[DataMuncher]:
        """Tests the neural network for one epoch.

        Returns
        -------
        A list of DataMuncher containing the history of the process

        """

        # Make Test entry
        self.train_current_statistics = self.train_current_statistics.update({}, {UO.SET: {'Test': {}}})

        # empty out the epoch info
        self.test_epoch_info = []

        for self.batch in range(self.number_of_iterations):

            self.on_test_batch_begin()

            # test on batch
            self.test_batch_info: DataMuncher = self.test_one_batch()

            # keep the result
            # decided to update the train epoch info incrementally in case it was needed
            self.test_epoch_info.append(
                # add additional info
                self.test_batch_info.update(
                    {},
                    {
                        UO.SET_ONLY: {
                            BATCH_STRING: self.batch,
                        }
                    }
                )
            )
            self.train_current_statistics = \
                self.train_current_statistics.update(
                    {'Test': {FO.EXISTS: 1}},
                    {UO.SET: {'Test': self.test_batch_info}},
                )

            self.on_test_batch_end()

        return self.test_epoch_info

    @abstractmethod
    def train_one_batch(self) -> DataMuncher:
        """Trains the neural network for one batch.

        Returns
        -------
        A DataMuncher containing the history of the process

        """

        raise NotImplementedError

    @abstractmethod
    def val_one_batch(self) -> DataMuncher:
        """Performs validation for the neural network for one batch.

        Returns
        -------
        A DataMuncher containing the history of the process

        """

        raise NotImplementedError

    @abstractmethod
    def test_one_batch(self) -> DataMuncher:
        """Performs test for the neural network for one batch.

        Returns
        -------
        A DataMuncher containing the history of the process

        """

        raise NotImplementedError

    def signal_catcher(self, os_signal, frame):
        """Catches an OS signal and calls it on its workers."""

        # Take care of signals
        if os_signal == signal.SIGINT:
            info = DataMuncher({'signal': os_signal})
            self.on_os_signal(info)
            self._universal_log('Interrupt signal received, exiting...', 'error')
            sys.exit(1)
        elif os_signal == signal.SIGTERM:
            info = DataMuncher({'signal': os_signal})
            self.on_os_signal(info)
            self._universal_log('Termination signal received, exiting...', 'error')
            sys.exit(0)
        elif os_signal == signal.SIGTSTP:
            info = DataMuncher({'signal': os_signal})
            self.on_os_signal(info)
            self._universal_log('Terminal stop signal received.', 'warning')
            signal.signal(os_signal, signal.SIG_DFL)
            os.kill(os.getpid(), os_signal)
        elif os_signal == signal.SIGCONT:
            info = DataMuncher({'signal': os_signal})
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
