from ..trainer.trainer import Trainer
from ..util.config import ConfigParser
from typing import List, Dict
from ..logger.log_hug import LogHug
from abc import ABC, abstractmethod


class Predictor(Trainer, ABC):
    """A class to help with prediction of neural network."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # Total number of epochs, total batch count, batch size, current epoch, and current batch number
        self.epochs: int = 1
        self.number_of_iterations: int = -1  # Has to be set by the data loader
        self.batch_size: int = -1  # Has to be set by the data loader
        self.epoch: int = 0
        self.batch: int = 0

        self.train_info_dict = None
        self.val_info_dict = {}

        # Register OS signals to be caught
        self._register_signal_catch()

        # Register exception hook to be caught
        self._register_exception_hook()

    def train(self) -> List[Dict]:
        pass

    def prediction(self) -> List[Dict]:
        """Performs the prediction.

        Returns
        -------
        A List[Dict] containing the history of the validation process

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
            # This is the end of the epoch, so, epoch number is incremented
            # Also, history is recorder
            self.epoch += 1
            self.history.append(epoch_history)

        # Everything is finished
        self.on_end()

        return self.history

    def one_epoch(self) -> Dict:
        """Performs the validation for one epoch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        # Empty out the train statistics
        self.train_statistics = LogHug()

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
            **val_info_dict
        }

        return epoch_dict

    def val_one_epoch(self) -> Dict:
        """Performs validation for the neural network for one epoch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        # Make Validation entry
        self.train_statistics = self.train_statistics.update({}, {'$set': {'Validation': {}}})

        for self.batch in range(self.number_of_iterations):

            self.on_val_batch_begin()

            self.val_info_dict = self.val_one_batch()

            self.train_statistics = \
                self.train_statistics.update(
                    {'_bc': {'$regex': 'Validation'}},
                    {'$set': self.val_info_dict},
                    {'recursive': True}
                )

            self.on_val_batch_end()
            self.on_predict_batch_end()

        return self.val_info_dict

    @abstractmethod
    def val_one_batch(self) -> Dict:
        """Performs validation for the neural network for one batch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        raise NotImplementedError
