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
                dtype
                    The dtype of the final data

        """

        # Set the dtype that should be used for conversion
        self.dtype = config.get_or_else('dtype', torch.float)

        super().__init__(config)

    def process(self, data: np.ndarray, dtype=None) -> torch.Tensor:
        """"Converts the given data to torch tensors.

        It takes input data of np.ndarray type and converts them to torch.Tensor with dtype of self.dtype.

        Parameters
        ----------
        data : np.ndarray
            A numpy array containing the data
        dtype
            The dtype of the final tensor. This will override the given dtype in the constructor.
                If not given, the default will be self.dtype

        Returns
        -------
        torch tensor of the data

        """

        # Do the conversion
        dtype = dtype or self.dtype
        try:
            return torch.tensor(data, dtype=dtype)
        except Exception:
            raise Exception('Cannot convert data to torch tensor.')
