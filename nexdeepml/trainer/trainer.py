from ..base import base
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
import numpy as np
from ..callback.callback import CallbackManager, Callback
from ..model.model import ModelManager, Model
from typing import Union, List, Dict, Type
from abc import ABC, abstractmethod


class Trainer(base.BaseEventManager, ABC):
    """A class to help with training a neural network."""

    def __init__(self, config: ConfigParser):
        """Initializer for the this instance of the class"""

        super().__init__(config)

        # Total number of epochs, total batch count, batch size, current epoch, and current batch number
        self.epochs: int = self._config.epochs
        self.number_of_iterations: int = -1  # Has to be set by the data loader
        self.batch_size: int = -1  # Has to be set by the data loader
        self.epoch: int = 0
        self.batch: int = 0

        # Set placeholders for the train and validation data
        self.train_data: DataMuncher = DataMuncher()
        self.val_data: DataMuncher = DataMuncher()

        # Set placeholder for callbacks
        self.callback: Type[CallbackManager] = self.create_callback()

        # Set placeholder for model
        self.model: Type[ModelManager] = self.create_model()

        # Create history list for keeping the history of the net
        self.history = []
        # Make dummy variables
        self.train_info_dict = {}
        self.val_info_dict = {}

    def create_callback(self) -> Union[Type[CallbackManager], Type[Callback]]:
        """Creates an instance of the callback and returns it."""

        pass

    def create_model(self) -> Union[Type[ModelManager], Type[Model]]:
        """Creates an instance of the model and returns it."""

        pass

    def train(self) -> List[Dict]:
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

    def train_one_epoch(self) -> Dict:
        """Trains the neural network for one epoch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        for self.batch in range(self.number_of_iterations):

            self.on_batch_begin()
            self.on_train_batch_begin()

            self.train_info_dict = self.train_one_batch()

            self.on_train_batch_end()
            self.on_batch_end()

        return self.train_info_dict

    def val_one_epoch(self) -> Dict:
        """Performs validation for the neural network for one epoch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        for self.batch in range(self.number_of_iterations):

            self.on_val_batch_begin()

            self.val_info_dict = self.val_one_batch()

            self.on_val_batch_end()

        return self.val_info_dict

    @abstractmethod
    def train_one_batch(self) -> Dict:
        """Trains the neural network for one batch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        raise NotImplementedError

    @abstractmethod
    def val_one_batch(self) -> Dict:
        """Performs validation for the neural network for one batch.

        Returns
        -------
        A dictionary containing the history of the process

        """

        raise NotImplementedError
