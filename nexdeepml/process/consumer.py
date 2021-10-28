from .process import ProcessManager, Process
from .image import SampleImagePreprocessManager
from ..util.config import ConfigParser
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
