import subprocess
from ..util import util
from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.option import Option
from typing import Optional
from mpi4py import MPI
import os
from readerwriterlock import rwlock


class _MPICommunicatorSingletonClass(BaseWorker):
    """
    Class to handle OpenMPI-related tasks.

    This class is a singleton. This has to be created only once and then reused.
    """

    def __init__(self, config: ConfigParser = None):
        """
        Initializer.

        Parameters
        ----------
        config : ConfigParser
            configuration for this instance.

        """

        super().__init__(config)

        # lock for accessing elements
        _lock = rwlock.RWLockFair()
        self._lock_read = _lock.gen_rlock()
        self._lock_write = _lock.gen_wlock()

        # keep communicators
        self._communicators: DataMuncher = DataMuncher({
            "world": MPI.COMM_WORLD,
        })

        # get the rank and size
        # and get the env vars
        self._rank: int = MPI.COMM_WORLD.Get_rank()
        self._size: int = MPI.COMM_WORLD.Get_size()
        self._local_rank: int = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") or 0)
        self._local_size: int = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE") or 1)
        self._universe_size: int = int(os.getenv("OMPI_UNIVERSE_SIZE") or 1)
        self._node_rank: int = int(os.getenv("OMPI_COMM_WORLD_NODE_RANK") or 0)

        # check if we have run by mpirun and get real tty
        parent_command = \
            subprocess.check_output(
                ['ps', '-p', f'{os.getppid()}', '-o', 'cmd', '--no-headers']
            ).decode('utf-8').strip().split()[0]
        self.is_mpi_run = True if parent_command in ['mpirun', 'mpiexec'] else False
        self.mpi_tty_fd = \
            subprocess.check_output(
                ['readlink', '-f', f'/proc/{os.getppid()}/fd/1']
            ).decode('utf-8').strip() \
            if self.is_mpi_run is True \
            else util.get_tty_fd()

    def _create_logger(self):
        """
        Creates a logger for this instance.
        This has to be done to get rid of the circular dependency between this class and the logger class.
        """

        return None

    def init_logger(self):
        """
        Initializes the logging feature.
        This has to be called after initializing.
        This is to prevent circular dependency between this class and the logger class.

        """

        # create the logger
        self._log = super()._create_logger()

        # set logger name
        self._log.set_name("OpenMPI communicator")

    def update_env_vars(self):
        """Updates the internal representation of the OpenMPI related environmental variables"""

        with self._lock_write:
            self._rank: int = MPI.COMM_WORLD.Get_rank()
            self._size: int = MPI.COMM_WORLD.Get_size()
            self._local_rank: int = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") or 0)
            self._local_size: int = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE") or 1)
            self._universe_size: int = int(os.getenv("OMPI_UNIVERSE_SIZE") or 1)
            self._node_rank: int = int(os.getenv("OMPI_COMM_WORLD_NODE_RANK") or 0)

    @staticmethod
    def clone(communicator: MPI.Comm = None) -> MPI.Comm:
        """
        Clones a communicator and returns it.

        Parameters
        ----------
        communicator : MPI.Comm, optional
            The communicator to clone, if not given, world will be used

        Returns
        -------
        MPI.Comm
            The cloned communicator
        """

        communicator = communicator if communicator is not None else MPI.COMM_WORLD

        return communicator.Clone()

    def register_communicator(self, communicator: MPI.Comm, name: str) -> bool:
        """
        Saves the given communicator with the name provided.

        Parameters
        ----------
        communicator : MPI.Comm
            Communicator to save
        name : str
            name to save by

        Returns
        -------
        bool
            whether or not the save was successful

        """

        # if already exist, skip
        if self._communicators.get_option(name).is_defined():
            return False

        # save it
        with self._lock_write:
            self._communicators = \
                self._communicators.update({}, {
                    name: communicator,
                })
        return True

    def get_communicator_option(self, name: str) -> Option[MPI.Comm]:
        """
        Returns an Option with the communicator.

        Parameters
        ----------
        name : str
            name of the

        Returns
        -------
        Option[MPI.Comm]

        """

        with self._lock_read:
            comm = self._communicators.get_option(name)

        return comm

    def get_communicator(self, name: str) -> MPI.Comm:
        """
        Returns the communicator. The communicator must exist, otherwise, will raise error.

        Parameters
        ----------
        name : str
            name of the

        Returns
        -------
        Option[MPI.Comm]

        """

        comm = self._communicators.get_option(name)

        if comm.is_empty():
            raise ValueError(f"communicator with name of {name} does not exist to get")

        return comm.get()

    def get_or_create_communicator(self, name: str) -> MPI.Comm:
        """
        Returns the communicator.
        If the communicator exists, returns it. If not, create it by cloning the world, save it, and return it.

        Parameters
        ----------
        name : str
            name of the

        Returns
        -------
        Option[MPI.Comm]

        """

        comm_opt = self.get_communicator_option(name)

        # if exists, return
        if comm_opt.is_defined():
            return comm_opt.get()

        # if not exist
        # create, save, return
        comm = self.clone()
        self.register_communicator(comm, name)

        return comm

    @staticmethod
    def get_rank_size(communicator: MPI.Comm) -> (int, int):
        """
        Gets the rank and size given an MPI communicator.

        Parameters
        ----------
        communicator : MPI.Comm
            an MPI communicator from which rank and size are extracted

        Returns
        -------
        (int, int)
            tuple of (rank, size)

        """

        rank: int = communicator.Get_rank()
        size: int = communicator.Get_size()

        return rank, size

    def get_rank(self) -> int:
        """Returns the rank."""

        with self._lock_read:
            return self._rank

    def get_size(self) -> int:
        """Returns the size."""

        with self._lock_read:
            return self._size


# this is the only instance that everyone should use
# this instance has to be initialized and set at the beginning of the program
# then everyone should use this instance
mpi_communicator: Optional[_MPICommunicatorSingletonClass] = None


def init(config: ConfigParser = None):
    global mpi_communicator
    mpi_communicator = _MPICommunicatorSingletonClass(config)
    mpi_communicator.init_logger()
