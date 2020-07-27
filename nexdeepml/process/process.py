from ..base import base


class ProcessManager(base.BaseManager):
    """This abstract class manages the Pre- and Post-Process instances or Managers."""

    def __init__(self):
        """Initializes the class."""

        super().__init__()