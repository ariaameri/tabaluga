from ...util.config import ConfigParser
from ...base import base


class PostprocessManager(base.BaseManager):
    """This abstract class manages the Post-Process instances."""

    def __init__(self):
        """Initializes the post-process manager."""

        super().__init__()


class Postprocess(base.BaseWorker):
    """A class that is the base/parent class of any post-process to be defined."""

    def __init__(self):
        """Initializes the post-process instance."""

        super().__init__()


