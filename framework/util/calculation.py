class Calculation:
    """Class to hold common calculation functions."""

    @staticmethod
    def exp_average(d_item: float, item: float = None, beta: float = .9, count: int = -1) -> float:
        """Calculates the new exponential moving average for the inputs.

        Returns d_item if item is -np.inf

        Parameters
        ----------
        d_item : float
            The value of the recent element
        item : float, optional
            Current value of the average. If not given, `d_item` will be returned
        beta : float
            Exponential moving average beta
        count : int, optional
            Count number of the new item, used for bias correction

        Returns
        -------
        Updated value of the average

        """

        import numpy as np

        # Do the exponential averaging
        average = \
            beta * item + (1 - beta) * d_item \
            if item != -np.inf and item != np.inf and item is not None \
            else d_item

        if 1 <= count <= 30:
            average /= (1 - beta ** count)

        return average
