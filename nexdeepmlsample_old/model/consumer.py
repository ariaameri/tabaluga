from nexdeepml.model import model
from nexdeepml.util.config import ConfigParser
from abc import ABC
import torch
import numpy as np
from .joint_stem_detection.joint_stem_pytorch import CropWeed


class SamplePyTorchModelManager(model.ModelPyTorchManager, torch.nn.Module):
    """Sample model manager for pyTorch."""

    def __init__(self, config: ConfigParser):
        """Initializes the model by setting the configuration and the layers needed.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance

        """

        super().__init__(config)

        # Defining the models here for the pyTorch engine to capture its parameters
        # self.model1 = SamplePyTorchModel(self._config.model, 5, 2, 5)

        self.model1 = CropWeed()

        # self.create_workers()

    def create_workers(self) -> None:
        """Creates model instances as workers."""

        # self.workers['model1'] = self.model1

        # self.workers['joint_stem'] = CropWeed()

        pass

    def forward(self, x):
        """Feedforward of the neural network.

        Parameters
        ----------
        x
            Input to the neural net

        """

        # out = self.workers['model1'](x)

        # out = self.workers['joint_stem'](x)
        out = self.model1(x)

        return out


class SamplePyTorchModel(model.ModelPyTorch):
    """Sample pyTorch model to implement a simple neural network."""

    def __init__(self, config: ConfigParser, d_in: int, h: int, d_out: int):
        """Initializes the model by setting the configuration and the layers needed.
        
        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance
        d_in : int
            Input dimension
        h : int
            Hidden dimension
        d_out : int
            Output dimension

        """
        
        super(SamplePyTorchModel, self).__init__(config)

        self.input_linear = torch.nn.Linear(d_in, h)
        self.middle_linear = torch.nn.Linear(h, h)
        self.output_linear = torch.nn.Linear(h, d_out)

    def forward(self, x):
        """Forward pass of the model"""

        h_relu = torch.nn.functional.relu(self.input_linear(x))

        h_relu = torch.nn.functional.relu(self.middle_linear(h_relu))

        y_pred = self.output_linear(h_relu)

        return y_pred
