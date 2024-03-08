from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..communicator import mpi
from abc import abstractmethod, ABC


# The following class should be a singleton
# it should be extended to implement the abstract methods and then instantiated once per program
class Horovod(BaseWorker, ABC):

    def __init__(self, hvd, config: ConfigParser):
        """
        Initializer.

        Parameters
        ----------
        hvd
            horovod instance to use. this can be for example horovod.torch or horovod.tensorflow or ...
        config : ConfigParser, optional
            config for this instance
        """

        super().__init__(config)

        # set horovod
        self.hvd = hvd

        # get a new mpi group for horovod
        self.mpi_comm_name = 'horovod'
        mpi_comm_res = mpi.mpi_communicator.get_or_create_communicator(self.mpi_comm_name)
        if mpi_comm_res.is_err():
            self._log.error(f"encountered error while getting mpi communicator with error of {mpi_comm_res.get_err()}")
            raise RuntimeError("encountered error while getting mpi communicator")
        self.mpi_comm = mpi_comm_res.get()

        self.rank, self.size = mpi.mpi_communicator.get_rank_size(self.mpi_comm)

        # initialize!
        self.init()

        # perform checks
        self.checks()

        # some bookkeeping
        self.optimizer: DataMuncher = DataMuncher({})

    def checks(self):
        """Performs some checks."""

        # if not in distributed mode raise error
        if mpi.mpi_communicator.is_distributed() is False:
            self._log.error("horovod is initiated in non-distributed mode!")
            raise RuntimeError("horovod is initiated in non-distributed mode!")

        # if MPI not built multi-threading support, panic!
        if self.hvd.mpi_threads_supported() is False:
            self._log.error("mpi does not support multi threading and horovod cannot work")
            raise RuntimeError("mpi does not support multi threading and horovod cannot work")

        # if horovod not built with mpi, panic!
        if self.hvd.mpi_built() is False:
            self._log.error("horovod not built with mpi")
            raise RuntimeError("horovod not built with mpi")

        # do we have cuda support?
        if self.hvd.cuda_built() is False:
            self._log.warning("horovod not built with cuda support")

    def init(self):
        """Initializes this method"""

        # init horovod
        try:
            self.hvd.init(mpi.mpi_communicator.get_communicator(self.mpi_comm_name))
            self._log.info(f'horovod initialized for local rank {self.get_local_rank()}!')
        except BaseException as e:
            self._log.error(f'horovod initialized failed for local rank {self.get_local_rank()}')
            raise e

        # pin a gpu
        self.pin_cuda_gpu()

    @abstractmethod
    def init_training(self, *args, **kwargs):
        """initializes the training by doing initial communications."""

        raise NotImplementedError

    def get_rank(self) -> int:
        """returns the rank according to horovod."""

        return self.hvd.rank()

    def get_cross_rank(self) -> int:
        """returns the cross rank according to horovod."""

        return self.hvd.cross_rank()

    def get_size(self) -> int:
        """returns the size according to horovod."""

        return self.hvd.size()

    def get_local_rank(self) -> int:
        """returns the local rank according to horovod."""

        return self.hvd.local_rank()

    def get_local_size(self) -> int:
        """returns the local size according to horovod."""

        return self.hvd.local_size()

    def is_nccl_built(self) -> bool:
        """returns boolean on whether horovod is built with nccl."""

        return True if self.hvd.nccl_built() != 0 else False

    def is_initialized(self) -> bool:
        """returns boolean on whether horovod has been initialized already."""

        return self.hvd.is_initialized()

    @abstractmethod
    def pin_cuda_gpu(self):
        """Pin this process to the gpu of number 'local rank'."""

        raise NotImplementedError

    @abstractmethod
    def get_learning_rate_scalar(self) -> float:
        """Returns learning rate scalar based on the config."""

        raise NotImplementedError

    @abstractmethod
    def wrap_optimizer(self, name: str, *args, **kwargs):
        """
        Method to wrap and return the optimizer by hvd.DistributedOptimizer.
        This will also save the resulting optimizer with the given name in the class.

        Parameters
        ----------
        name : str
            name of the optimizer to use for storage

        Returns
        -------
        optimizer
        """

        raise NotImplementedError

