from ...util.config import ConfigParser
from ...base import base


class PreprocessManager(base.BaseManager):
    """This abstract class manages the Pre-Process instances."""

    def __init__(self):
        """Initializes the pre-process manager."""

        super().__init__()


class Preprocess(base.BaseWorker):
    """A class that is the base/parent class of any pre-process to be defined."""

    def __init__(self):
        """Initializes the pre-process instance."""

        super().__init__()


