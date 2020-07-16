from typing import Dict


class Callback:
    """An abstract class that is the base/parent class of any callbacks to be defined."""

    def __init__(self):
        """Initializes the callback class"""

        pass

    # TODO: Figure out the type of output for the callback functions
    # TODO: Does callback have to perform the change or pass it back to the trainer to do the change?

    def on_begin(self, info: Dict) -> Dict:
        """Method to be called at the beginning.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        pass

    def on_end(self, info: Dict) -> Dict:
        """Method to be called at the end.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        pass

    def on_epoch_begin(self, info: Dict) -> Dict:
        """Method to be called at the beginning of each epoch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        pass

    def on_epoch_end(self, info: Dict) -> Dict:
        """Method to be called at the end of each epoch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        pass

    def on_batch_begin(self, info: Dict) -> Dict:
        """Method to be called at the beginning of each batch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        pass

    def on_batch_end(self, info: Dict) -> Dict:
        """Method to be called at the end of each batch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        Returns
        -------
        Config instance of the produced information

        """

        pass
