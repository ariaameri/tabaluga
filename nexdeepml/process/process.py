from ..base import base
from .preprocess import preprocess
from .postprocess import postprocess


class ProcessManager(base.BaseEventManager):
    """This abstract class manages the Pre- and Post-Process instances or Managers."""

    def __init__(self):
        """Initializes the class."""

        super().__init__()