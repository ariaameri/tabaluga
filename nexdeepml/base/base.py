from typing import List, Dict


class BaseWorker:
    """This abstract class servers as the parent of all worker classes."""

    def __init__(self):
        """Initializes the worker."""

        pass

    # Training event methods

    def on_begin(self, info: Dict):
        """Method to be called at the event of beginning of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_end(self, info: Dict):
        """Method to be called at the event of end of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_epoch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each training epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_epoch_end(self, info: Dict):
        """Method to be called at the event of end of each training epoch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        pass

    def on_batch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_batch_end(self, info: Dict):
        """Method to be called at the event of end of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    # Validation event methods

    def on_val_begin(self, info: Dict):
        """Method to be called at the event of beginning of validation.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        pass

    def on_val_end(self, info: Dict):
        """Method to be called at event of the end of validation.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_val_batch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each validation batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_val_batch_end(self, info: Dict):
        """Method to be called at the event of end of each validation batch.

            Parameters
            ----------
            info : dict
                The information needed

            """

        pass

    # Test event methods

    def on_test_begin(self, info: Dict):
        """Method to be called at the event of beginning of testing.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        pass

    def on_test_end(self, info: Dict):
        """Method to be called at event of the end of testing.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_test_batch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_test_batch_end(self, info: Dict):
        """Method to be called at the event of end of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass


class BaseManager(BaseWorker):
    """This abstract class servers as the parent of all manager classes.

    All managers are assumed to have self.workers: List[BaseWorker].
    At every event, the event will be called on the self.workers elements.
    """

    def __init__(self):
        """Initializes the manager."""

        super().__init__()

        self.workers: List[BaseWorker] = []

    def get_worker(self, index: int):
        """Returns the worker given its index.

        Parameters
        ----------
        index : int
            The index of the worker in class' worker list

        Returns
        -------
        worker : BaseWorker
            Reference to the worker inquired
        """

        return self.workers[index]

    def on_begin(self, info: Dict):
        """Method to be called at the event of beginning of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_begin(info)

    def on_end(self, info: Dict):
        """Method to be called at the event of end of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_end(info)

    def on_epoch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each training epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_epoch_begin(info)

    def on_epoch_end(self, info: Dict):
        """Method to be called at the event of end of each training epoch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        for worker in self.workers:
            worker.on_epoch_end(info)

    def on_batch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_batch_begin(info)

    def on_batch_end(self, info: Dict):
        """Method to be called at the event of end of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_batch_end(info)

    # Validation event methods

    def on_val_begin(self, info: Dict):
        """Method to be called at the event of beginning of validation.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        for worker in self.workers:
            worker.on_val_begin(info)

    def on_val_end(self, info: Dict):
        """Method to be called at event of the end of validation.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_end(info)

    def on_val_batch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each validation batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_batch_begin(info)

    def on_val_batch_end(self, info: Dict):
        """Method to be called at the event of end of each validation batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_batch_end(info)

    # Test event methods

    def on_test_begin(self, info: Dict):
        """Method to be called at the event of beginning of testing.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        for worker in self.workers:
            worker.on_test_begin(info)

    def on_test_end(self, info: Dict):
        """Method to be called at event of the end of testing.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_test_end(info)

    def on_test_batch_begin(self, info: Dict):
        """Method to be called at the event of beginning of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_test_batch_begin(info)

    def on_test_batch_end(self, info: Dict):
        """Method to be called at the event of end of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_test_batch_end(info)