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
        self.workers['callback'] = SampleCallbackManager(self._config.callback, self)

    def create_model(self):

        pass

    def train_one_batch(self) -> Dict:

        return {}

    def val_one_batch(self) -> Dict:

        return {}

    def signal_catcher(self, os_signal, frame):
        """Catches an OS signal and calls it on its workers."""

        # Take care of SIGINT
        if os_signal == signal.SIGINT:
            info = {'signal': os_signal}
            self.on_os_signal(info)


class SamplePyTorchTrainer(Trainer):
    """Sample pyTorch Trainer that incorporates pyTorch neural model."""

    def __init__(self, config: ConfigParser):

        super().__init__(config)

        self.create_workers()

        self.model: SamplePyTorchModelManager
        self.create_model()

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.criterion = torch.nn.MSELoss()

    def create_workers(self):

        self.workers['callback'] = SampleCallbackManager(self._config.callback, self)

    def create_model(self):

        self.model = SamplePyTorchModelManager(self._config.model)

    def train_one_batch(self) -> Dict:

        return {}

    def val_one_batch(self) -> Dict:

        return {}

