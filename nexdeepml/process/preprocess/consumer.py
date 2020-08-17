from .preprocess import PreprocessManager
from .image import ImageNormalizer, ImageResizer
from .pyTorch import ToTorchTensor, ToTorchGPU
from ...util.config import ConfigParser
from typing import Dict


class SampleImagePreprocessManager(PreprocessManager):
    """A simple class to manage Preprocess instances."""

    def __init__(self, config: ConfigParser):
        """Initializer.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance and its workers.

        """

        super().__init__(config)

        self.create_workers()

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image preprocess manager'
    #
    #     return string

    def create_workers(self):
        """Creates Preprocess instances."""

        self.workers['image_resizer'] = ImageResizer(self._config.resize)

        self.workers['image_normalizer'] = ImageNormalizer()

        self.workers['to_torch_tensor'] = ToTorchTensor(ConfigParser())

        self.workers['to_torch_gpu'] = ToTorchGPU(ConfigParser())

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, process the loaded train image data."""

        # data = info['data']['data']
        data = info['data']

        # processed_data = self.workers['image_resizer'].resize(data)
        processed_data = data.map(self.workers['image_resizer'].process)
        # processed_data = self.workers['image_normalizer'].normalize(processed_data)
        processed_data = processed_data.map(self.workers['image_normalizer'].process)
        processed_data = processed_data.map(self.workers['to_torch_tensor'].process)
        processed_data = processed_data.map(self.workers['to_torch_gpu'].process)

        return processed_data

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, process the loaded val image data."""

        data = info['data']

        # processed_data = self.workers['image_resizer'].resize(data)
        processed_data = data.map(self.workers['image_resizer'].process)
        # processed_data = self.workers['image_normalizer'].normalize(processed_data)
        processed_data = processed_data.map(self.workers['image_normalizer'].process)
        processed_data = processed_data.map(self.workers['to_torch_tensor'].process)
        processed_data = processed_data.map(self.workers['to_torch_gpu'].process)

        return processed_data
