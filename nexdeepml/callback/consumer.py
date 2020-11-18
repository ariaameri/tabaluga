from .callback import CallbackManager, ManagerCallback, TrainStatExpAverage
from ..base.base import BaseWorker
from ..dataloader.consumer import SampleDataManager
from ..logger.consumer import SampleLoggerManager
from ..util.config import ConfigParser
from typing import Dict, List
from collections import OrderedDict
from ..process.consumer import SampleProcessManager
from tqdm import tqdm


class SampleDataManagerCallback(ManagerCallback):
    """Simple class to create and initialize data loader callback and DataManager instance."""

    def __init__(self, config: ConfigParser, trainer=None):
        """Initialize the callback instance and create the DataManager instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this callback instance and the data manager class.
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config, trainer)

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Data manager callback'
    #
    #     return string

    def create_workers(self):
        """Create the DataManager instance"""

        self.workers['data_manager'] = SampleDataManager(self._config)

    def on_train_epoch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, update the batch size of the train data loader."""

        # Create the 'train' entry in the trainer's data
        self.trainer.data = self.trainer.data.update({}, {'$set_on_insert': {'train': {}}})

        self.trainer.batch_size, self.trainer.number_of_iterations = \
            self.workers['data_manager'].on_train_epoch_begin()

    def on_val_epoch_begin(self, info: Dict = None):
        """On beginning of val epoch, update the batch size of the val data loader."""

        # Create the 'val' entry in the trainer's data
        self.trainer.data = self.trainer.data.update({}, {'$set_on_insert': {'val': {}}})

        self.trainer.batch_size, self.trainer.number_of_iterations = \
            self.workers['data_manager'].on_val_epoch_begin()

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, load the batch data and put it in the trainer."""

        # Update the training data
        self.trainer.data = \
            self.trainer.data.update(
                {},
                {'$update_only':
                    {'train':
                          self.workers['data_manager'].on_batch_begin(
                              {
                                  'batch': self.trainer.batch
                              }
                          )
                    }
                }
            )

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, load the batch data and put it in the trainer."""

        # Update the validation data
        self.trainer.data = \
            self.trainer.data.update(
                {},
                {'$update_only':
                    {'val':
                        self.workers['data_manager'].on_val_batch_begin(
                            {
                                'batch': self.trainer.batch
                            }
                        )
                    }
                }
            )


class SampleProcessCallback(ManagerCallback):
    """Simple class to create and initialize data process callback and ProcessManager instance."""

    def __init__(self, config: ConfigParser, trainer=None):
        """Initialize the callback instance and create the DataManager instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this callback instance and the data manager class.
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config, trainer)

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Data process callback'
    #
    #     return string

    def create_workers(self):
        """Create the ProcessManager instance"""

        self.workers['data_process'] = SampleProcessManager(self._config)

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, process the loaded train data."""

        # Update the training data
        self.trainer.data = \
            self.trainer.data.update(
                {},
                {'$update_only':
                    {'train':
                        self.workers['data_process'].on_batch_begin(
                            {
                                'data': self.trainer.data.get('train')
                            }
                        )
                    }
                }
            )

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, process the loaded val data."""

        # Update the validation data
        self.trainer.data = \
            self.trainer.data.update(
                {},
                {'$update_only':
                    {'val':
                        self.workers['data_process'].on_val_batch_begin(
                            {
                                'data': self.trainer.data.get('val')
                            }
                        )
                    }
                }
            )


class SampleLoggerCallback(ManagerCallback):

    def __init__(self, config: ConfigParser, trainer=None):
        """Initialize the callback instance and create the Logger instances.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this callback instance and the data manager class.
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config, trainer)

    def create_workers(self):
        """Create the Logger instance"""

        self.workers['logger'] = SampleLoggerManager(self._config)
        # self.workers['logger'] = SampleTheProgressBarLoggerManager(self._config)

    def on_train_epoch_begin(self, info: Dict = None):

        self.workers['logger'].on_train_epoch_begin({
            'number_of_iterations': self.trainer.number_of_iterations,
            'epochs': self.trainer.epochs,
            'epoch': self.trainer.epoch
        })

    def on_batch_end(self, info: Dict = None):

        info = {
            'epoch': self.trainer.epoch,
            'stat': self.trainer.train_statistics,
        }

        self.workers['logger'].on_batch_end({
            'batch_size': 1,
            **info
        })

    def on_val_epoch_begin(self, info: Dict = None):

        self.workers['logger'].on_val_epoch_begin({
            'number_of_iterations': self.trainer.number_of_iterations,
            'epoch': self.trainer.epoch,
            'stat': self.trainer.train_statistics,
        })

    def on_val_batch_end(self, info: Dict = None):

        info = {
            'epoch': self.trainer.epoch,
            'stat': self.trainer.train_statistics
        }

        self.workers['logger'].on_val_batch_end({
            'batch_size': 1,
            **info
        })


class SampleCallbackManager(CallbackManager):
    """Simple CallbackManager that manages all instances of Callback's."""

    def __init__(self, config: ConfigParser, trainer):
        """Initializes the callback manager class.

        Parameters
        ----------
        config : ConfigParser
            The configuration for this instance and the rest of the instances it will initialize
        trainer : Trainer
            Reference to the trainer instance

        """

        super().__init__(config, trainer)

        # Create the workers
        self.create_workers()

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Callback manager'
    #
    #     return string

    def create_workers(self):
        """Creates and initializes workers."""

        self.workers['data_loader'] = \
            SampleDataManagerCallback(
                self._config.get_or_else('data_loader', None),
                self.trainer
            )
        self.workers['process'] = \
            SampleProcessCallback(
                self._config.get_or_else('process', None),
                self.trainer
            )
        self.workers['logger'] = \
            SampleLoggerCallback(
                self._config.get_or_else('logger', None),
                self.trainer
            )
        self.workers['train_stat_exp_average'] = \
            TrainStatExpAverage(
                self._config.get_or_else('train_stat_exp_average', None),
                self.trainer
            )
