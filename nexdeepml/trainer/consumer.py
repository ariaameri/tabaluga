from .trainer import Trainer
from ..util.config import ConfigParser
from ..callback.consumer import SampleCallbackManager
from typing import Dict
import signal


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
