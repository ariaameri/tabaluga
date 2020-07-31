from ..util.config import ConfigParser
from typing import List, Dict, Union
from abc import ABC, abstractmethod
import numpy as np
import re


class BaseWorker:
    """Class to serve as the base of all workers."""

    def __init__(self, config: ConfigParser = None):
        """Initializer of the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        self._config = config

    def print_config(self, depth: int = -1):
        """Prints the configuration of the instance.

        Parameters
        ----------
        depth : int, optional
            The depth until which we want to print the workers. If not given, will go until the end

        """

        print(self._config.str_representation(depth=depth))


class BaseEventWorker(BaseWorker):
    """This abstract class servers as the parent of all worker classes."""

    def __init__(self, config: ConfigParser = None):
        """Initializes the worker.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)

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

    def __init__(self, config: ConfigParser = None):
        """Initializes the manager.

        Goes over all the methods that start with 'on_' and calls the same event over all its workers.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance and its workers.

        """

        super().__init__(config)

        self.workers: Workers = Workers()

        # Call the events on each worker for each event starting with 'on_'
        # self._method_names: List[str] = [method for method in dir(self) if method.startswith('on_')]
        # for method_name in self._method_names:
        #     setattr(self, method_name, self._event_helper(method_name))

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

            for worker in this.workers:
                getattr(worker, method_name)(info)

        this = self
        return _event

    def get_worker(self, index: Union[str, int]) -> BaseWorker:
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

        return self.workers[index]

    def print_workers(self, depth: int = -1):
        """Prints the representation of the workers.

        Parameters
        ----------
        depth : int, optional
            The depth until which we want to print the workers. If not given, will go until the end

        """

        print(self.workers.str_representation(depth=depth))

    @abstractmethod
    def create_workers(self):
        """Creates and initializes workers."""

        raise NotImplementedError
    
    def on_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_begin(info)

    def on_end(self, info: Dict = None):
        """Method to be called at the event of end of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_end(info)

    def on_epoch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each training epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_epoch_begin(info)

    def on_epoch_end(self, info: Dict = None):
        """Method to be called at the event of end of each training epoch.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        for worker in self.workers:
            worker.on_epoch_end(info)

    # Training event methods

    def on_train_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_train_begin(info)

    def on_train_end(self, info: Dict = None):
        """Method to be called at the event of end of training.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_train_end(info)

    def on_train_epoch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each training epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_train_epoch_begin(info)

    def on_train_epoch_end(self, info: Dict = None):
        """Method to be called at the event of end of each training epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_train_epoch_end(info)

    def on_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_batch_begin(info)

    def on_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_batch_end(info)

    def on_train_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_train_batch_begin(info)

    def on_train_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each training batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_train_batch_end(info)

    # Validation event methods

    def on_val_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of validation.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        for worker in self.workers:
            worker.on_val_begin(info)

    def on_val_end(self, info: Dict = None):
        """Method to be called at event of the end of validation.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_end(info)

    def on_val_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each validation batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_batch_begin(info)

    def on_val_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each validation batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_batch_end(info)

    def on_val_epoch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each validation epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_epoch_begin(info)

    def on_val_epoch_end(self, info: Dict = None):
        """Method to be called at the event of end of each validation epoch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_val_epoch_end(info)

    # Test event methods

    def on_test_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of testing.

        Parameters
        ----------
        info : dict
            The information that has to be passed to the callback

        """

        for worker in self.workers:
            worker.on_test_begin(info)

    def on_test_end(self, info: Dict = None):
        """Method to be called at event of the end of testing.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_test_end(info)

    def on_test_batch_begin(self, info: Dict = None):
        """Method to be called at the event of beginning of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_test_batch_begin(info)

    def on_test_batch_end(self, info: Dict = None):
        """Method to be called at the event of end of each testing batch.

        Parameters
        ----------
        info : dict
            The information needed

        """

        for worker in self.workers:
            worker.on_test_batch_end(info)


class Workers:
    """A class to contain all workers in order for the manager classes."""

    def __init__(self):
        """Initializer of the instance."""

        # Keep the workers oder list as a list of strings
        self._workers_name_order: List[str] = []

        # Book keeping for iteration
        self._current_iteration_count: int = 0

    def register_worker(self, name: str, worker: BaseWorker, rank: int = -1):
        """Registers a new worker (or manager).

        Parameters
        ----------
        name : str
            The name of the worker (or manager)
        worker : BaseWorker
            The reference to the worker (or manager)
        rank : int, optional
            Rank of the worker (or manager) in the list. If not given, will insert at the end

        """

        # Check if the name of the workers are unique, conforming, and a subclass of BaseWorker
        if name in self._workers_name_order:
            raise Exception('Worker name exists! Please choose another name.')
        if re.match(r'^[a-zA-Z](\w|\d)+$', name) is None:
            raise Exception('Worker name should only consist of alphanumeric values starting with a letter.')
        # if inspect.isclass(worker) is False or issubclass(type(worker), BaseWorker) is False:
        #     raise Exception('Worker has to be an instance of the BaseWorker class.')

        # Add the worker as attribute and add it in the order list
        if rank == -1:
            self._workers_name_order.append(name)
        else:
            self._workers_name_order.insert(rank, name)

        self.__dict__[name] = worker

    def replace_worker(self, name: str, worker: BaseWorker) -> None:
        """Replaces an existing worker.

        Parameters
        ----------
        name : str
            The name of the worker to be replaced
        worker : BaseWorker
            The worker reference to be replaced

        """

        if name in self.__dict__:
            self.__dict__[name] = worker
        else:
            raise Exception(f'Could not find the worker with the name {name} to replace it!')

    def __len__(self):
        """Get the total number of workers."""

        return len(self._workers_name_order)

    def __iter__(self):
        """Method for making the class iterable."""

        return self

    def __next__(self):
        """Iterate over the workers"""

        # If we have not run out of workers
        if self._current_iteration_count < len(self._workers_name_order):

            # Get the worker
            worker = getattr(self, self._workers_name_order[self._current_iteration_count])

            self._current_iteration_count += 1

            return worker

        else:

            self._current_iteration_count = 0
            raise StopIteration

    def __getitem__(self, item) -> Union[BaseWorker, None]:
        """Get a worker.

        item can be string, return worker by name, or int, return worker by rank.
        """

        item_type = type(item)

        # If the query is with string, return based on name
        if item_type == str:
            # Check if the worker name exists
            if item in self.__dict__:
                worker = getattr(self, item)
                return worker

        # If the query is with int, return based on rank
        if item_type == int:
            # Check if the number is within range
            if item < len(self._workers_name_order):
                worker = getattr(self, self._workers_name_order[item])
                return worker

        # if everything else fail
        return None

    def get_rank(self, worker_name: str) -> int:
        """Get the rank of the worker by their name.

        Parameters
        ----------
        worker_name : str
            The name of the worker

        Returns
        -------
        The rank of the worker

        """

        # If the worker name exists, return it; otherwise, return -1
        if worker_name in self.__dict__:
            return self._workers_name_order.index(worker_name)
        else:
            return -1

    def __setattr__(self, key, value):
        """Set a new worker."""

        # Use a terrible way of setting class variables.
        # All class variables have to start with _
        # No worker name can start with _
        if key[0] == '_':
            self.__dict__[key] = value
        else:
            self.register_worker(key, value)

    def __setitem__(self, key, value):
        """Set an item like a dictionary."""

        self.__setattr__(key, value)

    def __str__(self) -> str:
        """Get a string representation of the instance."""

        out_string = self.str_representation(depth=1)

        return out_string

    def print(self, depth: int = -1):
        """Prints the workers and goes in depth.

        Parameters
        ----------
        depth : int, optional
            The depth until which we want to print the workers. If not given, will go until the end.

        """

        print(self.str_representation(depth))

    def str_representation(self, depth: int = -1) -> str:
        """Returns the string representation of the current instance and the its workers in depth.

        Parameters
        ----------
        depth : int, optional
            The depth until which we want to print the workers. If not given, will go until the end

        Returns
        -------
        String representation of the current instance and its workers in depth

        """

        if depth == 0:
            return ''

        out_string = ''
        # Find the number of digits to use for representing the workers
        length_worker_digit = \
            int(np.ceil(np.log10(len(self._workers_name_order)))) if len(self._workers_name_order) != 0 else 1

        for index, worker_name in enumerate(self._workers_name_order):
            worker = self.__dict__[worker_name]
            # Construct current worker's string
            out_string += \
                f'\033[38;5;28m{index:{length_worker_digit}d}\033[0m ' \
                f'\033[37m->\033[0m ' \
                f'\033[34m{worker_name}\033[0m: \033[38;5;245m{self.__dict__[worker_name]}\033[0m\n'
            # Check if the worker has worker and we have to go deep
            if issubclass(type(worker), BaseWorker) and 'workers' in worker.__dict__:
                # Get worker's string representation
                worker_string = worker.workers.str_representation(depth - 1)
                # Indent the representation and draw vertical lines for visual clarity and add to the original string
                worker_string = \
                    re.sub(
                        r'(^|\n)(?!$)',
                        r'\1' + f'\033[37m\u22EE\033[0m' + r'\t',
                        worker_string
                    ) \
                    if worker_string != '' else ''
                out_string += worker_string

        return out_string
