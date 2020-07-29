from typing import List, Dict, Union
from collections import OrderedDict
from abc import ABC, abstractmethod


class BaseWorker:
    """Class to serve as the base of all workers."""

    def __init__(self):
        """Initializer of the instance."""

        pass


class BaseEventWorker(BaseWorker):
    """This abstract class servers as the parent of all worker classes."""

    def __init__(self):
        """Initializes the worker."""

        super().__init__()

    # General events

    def on_epoch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each training epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_epoch_end(self, info: Dict = None):
        """Method to be called at the event of end of each training epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_end(self, info: Dict = None):
        """Method to be called at the event of end of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    # Training event methods

    def on_train_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_train_end(self, info: Dict = None):
        """Method to be called at the event of end of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_train_epoch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each epoch for training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_train_epoch_end(self, info: Dict = None):
        """Method to be called at the event of end of each epoch for training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_train_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_train_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    # Validation event methods

    def on_val_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of validation.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        pass

    def on_val_end(self, info: Dict = None):
        """Method to be called at event of the end of validation.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_val_epoch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each validation epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_val_epoch_end(self, info: Dict = None):
        """Method to be called at the event of end of each validation epoch.

            Parameters
            ----------
            info : dict
                The information needed

            """

        pass

    def on_val_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each validation batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_val_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each validation batch.

            Parameters
            ----------
            info : dict
                The information needed

            """

        pass

    # Test event methods

    def on_test_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of testing.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        pass

    def on_test_end(self, info: Dict = None):
        """Method to be called at event of the end of testing.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_test_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass

    def on_test_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        pass


class BaseEventManager(BaseEventWorker, ABC):
    """This abstract class servers as the parent of all manager classes.

    All managers are assumed to have self.workers: List[BaseWorker].
    At every event, the event will be called on the self.workers elements.
    """

    def __init__(self):
        """Initializes the manager.

        Goes over all the methods that start with 'on_' and calls the same event over all its workers.

        """

        super().__init__()

        self.workers: OrderedDict = OrderedDict()

        # Call the events on each worker for each event starting with 'on_'
        self._method_names: List[str] = [method for method in dir(self) if method.startswith('on_')]
        for method_name in self._method_names:
            setattr(self, method_name, self._event_helper(method_name))
        pass

    def _event_helper(self, method_name: str):
        """Helper function that returns a generic event function.

        Parameters
        ----------
        method_name : str
            The name of the method for which event is being generated

        Returns
        -------
        An event function
        """

        def _event(info: Dict = None):

            """An event function.

            Parameters
            ----------
            info : dict
                The information needed

            """

            for _, worker in this.workers.items():
                getattr(worker, method_name)(info)

        this = self
        return _event

    def get_worker(self, index: Union[str, int]) -> BaseEventWorker:
        """Returns the worker given its index.

        Parameters
        ----------
        index : Union[str, int]
            The index or name of the worker in class' worker ordered dictionary

        Returns
        -------
        worker : BaseEventWorker
            Reference to the worker inquired
        """

        # Take care of numerical index
        if type(index) == int:
            index = list(self.workers)[index]

        return self.workers[index]

    @abstractmethod
    def create_workers(self):
        """Creates and initializes workers."""

        raise NotImplementedError
