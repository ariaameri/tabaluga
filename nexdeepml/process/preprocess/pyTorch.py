from . import preprocess
from ...util.config import ConfigParser
import numpy as np
import torch


class ToTorchTensor(preprocess.Preprocess):
    """Converts data to torch tensors.

    It takes input numpy inputs and converts them to torch tensors.
    """

    def __init__(self, config: ConfigParser):
        """Initializes the class instance.

        Parameters
        ----------
        config : ConfigParser
            Contains the config needed including:
                destination_image_size : (int, int)
                    The image size to resize images to.
                interpolation, optional
                    The type of interpolation to use

        """

        # Set the dtype that should be used for conversion
        self.dtype = config.dtype or torch.float

        super().__init__(config)

    def process(self, data: np.ndarray) -> torch.Tensor:
        """"Converts the given data to torch tensors.

        It takes input data of np.ndarray type and converts them to torch.Tensor with dtype of self.dtype.

        Parameters
        ----------
        data : np.ndarray
            A numpy array containing the data

        Returns
        -------
        torch tensor of the data

        """

        # Do the conversion
        try:
            return torch.tensor(data, dtype=self.dtype)
        except Exception:
            raise Exception('Cannot convert data to torch tensor.')
