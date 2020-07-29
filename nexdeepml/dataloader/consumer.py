from .dataloader import DataLoaderManager, DataManager
from .image import ImageLoader
from ..util.config import ConfigParser
import numpy as np
from typing import Dict, OrderedDict


class SampleDataLoaderManager(DataLoaderManager):
    """A simple DataLoaderManager that creates DataLoader's for its metadata."""

    def __init__(self, config, metadata):
        super().__init__(config, metadata)

    def create_workers(self):
        self.workers['image_loader'] = ImageLoader(self._config, self.metadata)

    def __getitem__(self, item):

        return list(self.workers.items())[0][1][item]


class SampleDataManager(DataManager):
    """A simple DataManager that creates train, val, and test DataLoaderManager and manages them."""
    
    def __init__(self, config: ConfigParser):
        """Initialize the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance
        """

        self.workers: OrderedDict

        super().__init__(config)

    def create_workers(self):
        """Creates DataLoaderManagers (workers) for train, val, and test data."""

        self.workers['train'] = SampleDataLoaderManager(self._config, self.train_metadata)
        self.workers['val'] = SampleDataLoaderManager(self._config, self.val_metadata)
        self.workers['test'] = SampleDataLoaderManager(self._config, self.test_metadata)

    def on_epoch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, update the batch size of the train data loader."""

        self.workers['train'].set_batch_size(info['batch_size'])

    def on_val_epoch_begin(self, info: Dict = None):
        """On beginning of val epoch, update the batch size of the val data loader."""

        self.workers['val'].set_batch_size(info['batch_size'])

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, load the batch data and put it in the trainer."""

        trainer = info['trainer']
        batch = info['batch']  # The batch number
        trainer.train_data = self.workers['train'][batch]

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, load the batch data and put it in the trainer."""

        trainer = info['trainer']
        batch = info['batch']
        trainer.val_data = self.workers['val'][batch]
