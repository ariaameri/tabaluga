from .trainer import Trainer
from ..util.config import ConfigParser
from ..callback.consumer import SampleCallbackManager
from ..model.consumer import SamplePyTorchModelManager
from typing import Dict
import signal
import torch


class SampleTrainer(Trainer):

    def __init__(self, config: ConfigParser):
        super().__init__(config)

        self.create_workers()

    def create_workers(self):

        self.workers['callback'] = SampleCallbackManager(self._config.get('callback'), self)

    def create_model(self):

        pass

    def train_one_batch(self) -> Dict:

        return {}

    def val_one_batch(self) -> Dict:

        return {}


class SamplePyTorchTrainer(Trainer):
    """Sample pyTorch Trainer that incorporates pyTorch neural model."""

    def __init__(self, config: ConfigParser):

        super().__init__(config)

        self.create_workers()

        self.model: SamplePyTorchModelManager
        self.create_model()

        self.optimizer = torch.optim.Adam(self.model.parameters())

        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.CrossEntropyLoss()

    def create_workers(self):

        self.workers['callback'] = SampleCallbackManager(self._config.get('callback'), self)

    def create_model(self):

        self.model = SamplePyTorchModelManager(self._config.get('model'))

    def train_one_batch(self) -> Dict:

        # Get the neural net output
        deep_out = self.model(self.data.get('train.data'))

        self.optimizer.zero_grad()

        loss = self.criterion(deep_out[0], self.data.get('train.labels'))

        self.optimizer.step()

        return {'train_loss': loss}

    def val_one_batch(self) -> Dict:

        # Get the neural net output
        with torch.no_grad():
            deep_out = self.model(self.data.get('train.data'))

            loss = self.criterion(deep_out[0], self.data.get('train.labels'))

        return {'val_loss': loss}

