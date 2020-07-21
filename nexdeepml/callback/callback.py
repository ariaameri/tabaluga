from typing import Dict
from ..logger.logger import TQDMLogger


class Callback:
    """An abstract class that is the base/parent class of any callbacks to be defined."""

    def __init__(self):
        """Initializes the callback class"""

        pass

    # TODO: Figure out the type of output for the callback functions
    # TODO: Does callback have to perform the change or pass it back to the trainer to do the change?
    # TODO: Do we get to save a config?

    # Training (testing) methods

    def on_begin(self, info: Dict):
        """Method to be called at the beginning of training (testing).

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_end(self, info: Dict):
        """Method to be called at the end of training (testing).

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_epoch_begin(self, info: Dict):
        """Method to be called at the beginning of each training (testing) epoch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_epoch_end(self, info: Dict):
        """Method to be called at the end of each training (testing) epoch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_batch_begin(self, info: Dict):
        """Method to be called at the beginning of each training (testing) batch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_batch_end(self, info: Dict):
        """Method to be called at the end of each training (testing) batch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    # Validation methods

    def on_val_begin(self, info: Dict):
        """Method to be called at the beginning of validation.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_val_end(self, info: Dict):
        """Method to be called at the end of validation.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_val_batch_begin(self, info: Dict):
        """Method to be called at the beginning of each validation batch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented

    def on_val_batch_end(self, info: Dict):
        """Method to be called at the end of each validation batch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        raise NotImplemented
