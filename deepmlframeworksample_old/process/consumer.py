from nexdeepml.process.process import ProcessManager, Process
from .preprocess.consumer import SampleImagePreprocessManager
from nexdeepml.util.config import ConfigParser
from nexdeepml.process.pyTorch import ToTorchGPU
from typing import Dict


class SampleProcessManager(ProcessManager):
    """Simple ProcessManager class that manages processes and pre- and post-process managers."""

    def __init__(self, config: ConfigParser):
        """Initializes the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)

        self.create_workers()

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image process manager'
    #
    #     return string

    def create_workers(self):
        """Creates the pre- and post-processing managers as workers."""

        self.workers['preprocess'] = SampleImagePreprocessManager(self._config.get('preprocess'))
        self.workers['model_process'] = ToTorchGPU(ConfigParser())

    def on_train_begin(self, info: Dict = None):
        """On beginning of train epoch, process the model."""

        model = info['model']

        # Put the model to GPU
        model = self.workers['model_process'].process(model)

        return model

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) batch, process the loaded train data."""

        data = info['data']
        processed_data = self.workers['preprocess'].on_batch_begin({'data': data})

        return processed_data

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, process the loaded val data."""

        data = info['data']
        processed_data = self.workers['preprocess'].on_val_begin({'data': data})

        return processed_data
