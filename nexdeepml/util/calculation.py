class Calculation:
    """Class to hold common calculation functions."""

    @staticmethod
    def exp_average(item: float, d_item: float, beta: float = .9) -> float:
        """Calculates the new exponential moving average for the inputs.

        Returns d_item if item is -np.inf

        Parameters
        ----------
        item : float
            Current value of the average
        d_item : float
            The value of the recent element
        beta : float
            Exponential moving average beta

        Returns
        -------
        Updated value of the average

        """

        import numpy as np

        # Do the exponential averaging
        average = beta * item + (1 - beta) * d_item if item != -np.inf else d_item

        return average
