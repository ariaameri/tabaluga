from deepmlframework.trainer.trainer import Trainer
from deepmlframework.util.config import ConfigParser
from deepmlframeworksample.callback.consumer import SampleCallbackManager
from typing import Dict


class SampleTrainer(Trainer):

    def __init__(self, config: ConfigParser):
        super().__init__(config)

        self.create_workers()

    def create_workers(self):

        self.workers['callback'] = SampleCallbackManager(self._config.get('callback'), self)

    def create_model(self):

        pass

    def train_one_batch(self) -> Dict:
        import numpy as np
        return {'loss': np.random.rand(), 'doss': np.random.rand(), 'qoss': {'loss': np.random.rand()}}

    def val_one_batch(self) -> Dict:
        import numpy as np
        return {'loss': np.random.rand(), 'doss': np.random.rand(), 'qoss': {'loss': np.random.rand()}}

