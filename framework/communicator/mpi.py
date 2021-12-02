import pathlib
import subprocess
from ..util import util
from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.option import Option
from typing import Optional, Any, Sequence, List
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
        self.mpi_tty_fd: pathlib.Path = \
            pathlib.Path(subprocess.check_output(
                ['readlink', '-f', f'/proc/{os.getppid()}/fd/1']
            ).decode('utf-8').strip()) \
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

    def split(self,  communicator: MPI.Comm = None, color: int = 0, key: int = 0) -> MPI.Comm:
        """
        Splits a communicator and returns it.

        Parameters
        ----------
        communicator : MPI.Comm, optional
            The communicator to clone, if not given, world will be used
        color : int, optional
            the color for this instance
        key : int, optional
            the new key rank for this instance

        Returns
        -------
        MPI.Comm
            The cloned communicator
        """

        communicator = communicator if communicator is not None else MPI.COMM_WORLD

        return communicator.Split(color=color, key=key)

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

    def is_distributed(self) -> bool:
        """Returns true if we are in distributed mode."""

        return self.get_size() > 1

    def is_main_rank(self) -> bool:
        """Returns true if we are the main rank."""

        return self.get_rank() == 0

    def is_main_local_rank(self) -> bool:
        """Returns true if we are the main local rank."""

        return self.get_local_rank() == 0

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

        return comm.get().get()

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

    # point to point communication

    def p2p_send(self, data: Any, destination, tag: int = 0, name: str = None) -> None:
        """
        point-to-point communication for sending data

        Parameters
        ----------
        data : Any
            the data that has to be sent
        destination : int
            the rank to which the data should be sent
        tag : int, optional
            the tag corresponding to the data
        name : str, optional
            the name of the communicator to used. if not given, world will be used

        """

        communicator: MPI.Comm = self._communicators.get(name or 'world')

        communicator.send(obj=data, dest=destination, tag=tag)

    def p2p_receive(
            self,
            buffer: Optional[memoryview] = None,
            source: int = MPI.ANY_SOURCE,
            tag: int = MPI.ANY_TAG,
            status: Optional[MPI.Status] = None,
            name: str = None
    ) -> Any:
        """
        point-to-point communication for receiving data

        Parameters
        ----------
        buffer : Optional[MPI.Buffer], optional
            the buffer to be used
        source : int, optional
            the rank to which the data should be sent, defaults to any source
        tag : int, optional
            the tag corresponding to the data, defaults to any tag
        status
        name : str, optional
            the name of the communicator to used. if not given, world will be used

        Returns
        -------
        Any
            the received data

        """

        communicator: MPI.Comm = self._communicators.get(name or 'world')

        return communicator.recv(buf=buffer, source=source, tag=tag, status=status)

    # collective communications

    def barrier(self, name: str = None) -> None:
        """
        implements the call to the barrier method to wait for synchronization.

        Parameters
        ----------
        name : str, optional
            the name of the communicator to use for barrier. if not given, world will be used

        """

        communicator: MPI.Comm = self._communicators.get(name or 'world')

        communicator.barrier()

    def collective_bcast(self, data: Any, root_rank: int = 0, name: str = None) -> Any:
        """
        collective communication for broadcasting

        Parameters
        ----------
        data : Any
            the data that has to be sent
        root_rank : int
            the rank of the broadcaster
        name : str, optional
            the name of the communicator to used. if not given, world will be used

        Returns
        -------
        Any
            the received data, used for non-root ranks

        """

        communicator: MPI.Comm = self._communicators.get(name or 'world')

        return communicator.bcast(obj=data, root=root_rank)

    def collective_scatter(self, data: Sequence[Any], root_rank: int = 0, name: str = None) -> Any:
        """
        collective communication for scattering

        Parameters
        ----------
        data : Sequence[Any]
            the data that has to be sent. It has to be sequential for scattering
        root_rank : int
            the rank of the scatterer
        name : str, optional
            the name of the communicator to used. if not given, world will be used

        Returns
        -------
        Any
            the received data, used for non-root ranks

        """

        communicator: MPI.Comm = self._communicators.get(name or 'world')

        return communicator.scatter(sendobj=data, root=root_rank)

    def collective_gather(self, data: Any, root_rank: int = 0, name: str = None) -> Optional[List[Any]]:
        """
        collective communication for gathering

        Parameters
        ----------
        data : Any
            the data that has to be sent for gathering
        root_rank : int
            the rank of the gatherer
        name : str, optional
            the name of the communicator to used. if not given, world will be used

        Returns
        -------
        Optional[List[Any]]
            the received data, used for the root rank

        """

        communicator: MPI.Comm = self._communicators.get(name or 'world')

        return communicator.gather(sendobj=data, root=root_rank)

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

    def get_local_rank(self) -> int:
        """Returns the local rank."""

        with self._lock_read:
            return self._local_rank

    def get_local_size(self) -> int:
        """Returns the local size."""

        with self._lock_read:
            return self._local_size

    def get_rank(self) -> int:
        """Returns the rank."""

        # return 0

        with self._lock_read:
            return self._rank

    def get_size(self) -> int:
        """Returns the size."""

        # return 0

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
