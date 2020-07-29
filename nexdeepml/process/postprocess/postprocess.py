from ...util.config import ConfigParser
from ...base import base


class PostprocessManager(base.BaseEventManager):
    """This abstract class manages the Post-Process instances."""

    def __init__(self, config: ConfigParser):
        """Initializes the post-process manager.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance and its workers.

        """

        super().__init__(config)


class Postprocess(base.BaseEventWorker):
    """A class that is the base/parent class of any post-process to be defined."""

    def __init__(self, config: ConfigParser):
        """Initializes the post-process instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)


