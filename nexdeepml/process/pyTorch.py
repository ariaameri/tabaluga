from .process import Process
from ..util.config import ConfigParser
import torch


class ToTorchGPU(Process):
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
        self.gpu_number: int = config.get_or_else('gpu_number', 0)

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
