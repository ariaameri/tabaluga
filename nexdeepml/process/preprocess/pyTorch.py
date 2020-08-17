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
        self.dtype = config.dtype or torch.float

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


class ToTorchGPU(preprocess.Preprocess):
    """Transfers the data to gpu for pyTorch usage."""

    def __init__(self, config: ConfigParser):
        """Initializes the class instance.

        Parameters
        ----------
        config : ConfigParser
            Contains the config needed including:
                gpu_number : int
                    The number of the gpu to transfer the data to

        """

        super().__init__(config)

        # Bookkeeping for the number of the gpu to use
        self.gpu_number: int = config.gpu_number or 0

        # Check to see if cuda is available and then create the device
        self.is_cuda_available: bool = torch.cuda.is_available()
        self.device = torch.device('gpu', self.gpu_number) if self.is_cuda_available else torch.device('cpu')

    def process(self, data):
        """Transfers the given data to self.device, which should be a gpu

        It can take inputs of the type torch.tensor, torch.model, ... or anything that
        has the to(torch.device(.)) method.

        Parameters
        ----------
        data
            Data to be transferred to GPU

        Returns
        -------
        Data that is on GPU

        """

        # Do the transfer
        out = data.to(self.device) if self.is_cuda_available else data

        return out

