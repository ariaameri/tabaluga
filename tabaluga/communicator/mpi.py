import pathlib
import subprocess
from ..util import util
from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.option import Option
from typing import Optional, Any, Sequence, List
from ..util.result import Result, Ok, Err
try:
    from mpi4py import MPI
except Exception as e:
    import sys
    print(f'could not import mpi4py with error of {e}', file=sys.stderr)
import os
from readerwriterlock import rwlock
from . import config


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

        # check if we have mpi4py
        try:
            from mpi4py import MPI
            self._mpi4py_loaded = True

            # keep communicators
            self._communicators: DataMuncher = DataMuncher({
                "world": MPI.COMM_WORLD,
            })

        except Exception as e:
            self._mpi4py_loaded = False

        # lock for accessing elements
        _lock = rwlock.RWLockFair()
        self._lock_read = _lock.gen_rlock()
        self._lock_write = _lock.gen_wlock()

        # set the env vars
        self._rank: int = 0
        self._size: int = 1
        self._local_rank: int = 0
        self._local_size: int = 1
        self._universe_size: int = 1
        self._node_rank: int = 0
        self.update_env_vars()

        # check if we have run by mpi and get real tty
        self.is_mpi_run, self.mpi_tty_fd = self._is_run_mpi()

    def _is_run_mpi(self) -> (bool, Option[pathlib.Path]):
        """
        Check if we are running mpi and return some info.

        Returns
        -------
        bool, Option[pathlib.Path]
            whether we are running with mpi, mpi tty path in linux

        """

        try:
            if os.getppid() > 0:
                parent_command = \
                    subprocess.check_output(
                        ['ps', '-p', f'{os.getppid()}', '-o', 'cmd', '--no-headers']
                    ).decode('utf-8').strip().split()[0]
            else:
                parent_command = ''
        except:
            parent_command = ''
        is_mpi_run = True if parent_command in ['mpirun', 'mpiexec'] else False

        if self._mpi4py_loaded is False:
            if is_mpi_run:
                raise RuntimeError('detected an mpi run but mpi4py is not loaded')

        mpi_tty_fd: Option[pathlib.Path] = \
            Result\
            .from_func(
                subprocess.check_output,
                ['readlink', '-f', f'/proc/{os.getppid()}/fd/1'],
                stderr=subprocess.DEVNULL,
            )\
            .map(lambda x: x.decode('utf-8').strip())\
            .map(lambda x: pathlib.Path(x))\
            .ok() \
            if is_mpi_run is True \
            else util.get_tty_fd()

        return is_mpi_run, mpi_tty_fd

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
            if self._mpi4py_loaded:
                self._rank: int = MPI.COMM_WORLD.Get_rank()
                self._size: int = MPI.COMM_WORLD.Get_size()
                self._local_rank: int = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") or 0)
                self._local_size: int = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE") or 1)
                self._universe_size: int = int(os.getenv("OMPI_UNIVERSE_SIZE") or 1)
                self._node_rank: int = int(os.getenv("OMPI_COMM_WORLD_NODE_RANK") or 0)
            else:
                self._rank: int = 0
                self._size: int = 1
                self._local_rank: int = 0
                self._local_size: int = 1
                self._universe_size: int = 1
                self._node_rank: int = 0

    def _check_mpi4py_loaded(self) -> None:
        """Helper method to check if mpi4py is loaded and raise exception if not."""

        if self._mpi4py_loaded is False:
            raise RuntimeError("mpi4py not loaded but mpi-requiring method called")

    def clone(self, communicator: 'MPI.Comm' = None) -> Result['MPI.Comm', Exception]:
        """
        Clones a communicator and returns it.

        Parameters
        ----------
        communicator : MPI.Comm, optional
            The communicator to clone, if not given, world will be used

        Returns
        -------
        Result['MPI.Comm', Exception]
            The cloned communicator
        """

        self._check_mpi4py_loaded()

        communicator = communicator if communicator is not None else MPI.COMM_WORLD

        return Result.from_func(communicator.Clone)

    def split(self,  communicator: 'MPI.Comm' = None, color: int = 0, key: int = 0) -> Result['MPI.Comm', Exception]:
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
        Result['MPI.Comm', Exception]
            The cloned communicator
        """

        self._check_mpi4py_loaded()

        communicator = communicator if communicator is not None else MPI.COMM_WORLD

        return Result.from_func(communicator.Split, color=color, key=key)

    def register_communicator(self, communicator: 'MPI.Comm', name: str) -> bool:
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

        self._check_mpi4py_loaded()

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

    def get_communicator_option(self, name: str) -> Option['MPI.Comm']:
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

        self._check_mpi4py_loaded()

        with self._lock_read:
            comm = self._communicators.get_value_option(name)

        return comm

    def get_communicator(self, name: str) -> 'MPI.Comm':
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

        self._check_mpi4py_loaded()

        comm = self._communicators.get_value_option(name)

        if comm.is_empty():
            raise ValueError(f"communicator with name of {name} does not exist to get")

        return comm.get()

    def get_or_create_communicator(self, name: str) -> Result['MPI.Comm', Exception]:
        """
        Returns the communicator.
        If the communicator exists, returns it. If not, create it by cloning the world, save it, and return it.

        Parameters
        ----------
        name : str
            name of the

        Returns
        -------
        Result['MPI.Comm', Exception]

        """

        self._check_mpi4py_loaded()

        comm_opt = self.get_communicator_option(name)

        # if exists, return
        if comm_opt.is_defined():
            return Ok(comm_opt.get())

        # if not exist
        # create, save, return
        comm = self.clone()
        if comm.is_err():
            return comm
        self.register_communicator(comm.get(), name)

        return comm

    # point to point communication

    def p2p_send(self, data: Any, destination, tag: int = 0, name: str = None) -> Result[None, Exception]:
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

        Returns
        -------
        Result[None, Exception]
        """

        self._check_mpi4py_loaded()

        communicator: MPI.Comm = self._communicators.get(name or 'world')

        return Result.from_func(communicator.send, obj=data, dest=destination, tag=tag)

    def p2p_receive(
            self,
            buffer: Optional[memoryview] = None,
            source: int = None,
            tag: int = None,
            status: Optional['MPI.Status'] = None,
            name: str = None
    ) -> Result[Any, Exception]:
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
        Result[Any, Exception]
            the received data

        """

        self._check_mpi4py_loaded()

        source = MPI.ANY_SOURCE if source is None else source
        tag = MPI.ANY_TAG if tag is None else tag

        communicator: Option[MPI.Comm] = self._communicators.get_value_option(name or 'world')
        if communicator.is_empty():
            return Err(RuntimeError(f"communicator '{name or 'world'}' does not exit"))

        return Result.from_func(communicator.get().recv, buf=buffer, source=source, tag=tag, status=status)

    # collective communications

    def barrier(self, name: str = None) -> Result[None, Exception]:
        """
        implements the call to the barrier method to wait for synchronization.

        Parameters
        ----------
        name : str, optional
            the name of the communicator to use for barrier. if not given, world will be used

        Returns
        -------
        Result[None, Exception]
        """

        self._check_mpi4py_loaded()

        communicator: Option[MPI.Comm] = self._communicators.get_value_option(name or 'world')
        if communicator.is_empty():
            return Err(RuntimeError(f"communicator '{name or 'world'}' does not exit"))

        return Result.from_func(communicator.get().barrier)

    def collective_bcast(self, data: Any, root_rank: int = 0, name: str = None) -> Result[Any, Exception]:
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
        Result[Any, Exception]
            the received data, used for non-root ranks

        """

        self._check_mpi4py_loaded()

        communicator: Option[MPI.Comm] = self._communicators.get_value_option(name or 'world')
        if communicator.is_empty():
            return Err(RuntimeError(f"communicator '{name or 'world'}' does not exit"))

        return Result.from_func(communicator.get().bcast, obj=data, root=root_rank)

    def collective_scatter(self, data: Sequence[Any], root_rank: int = 0, name: str = None) -> Result[Any, Exception]:
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
        Result[Any, Exception]
            the received data, used for non-root ranks

        """

        self._check_mpi4py_loaded()

        communicator: Option[MPI.Comm] = self._communicators.get_value_option(name or 'world')
        if communicator.is_empty():
            return Err(RuntimeError(f"communicator '{name or 'world'}' does not exit"))

        return Result.from_func(communicator.get().scatter, sendobj=data, root=root_rank)

    def collective_gather(self, data: Any, root_rank: int = 0, name: str = None) \
            -> Result[Optional[List[Any]], Exception]:
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
        Result[Optional[List[Any]], Exception]
            the received data, used for the root rank

        """

        self._check_mpi4py_loaded()

        communicator: Option[MPI.Comm] = self._communicators.get_value_option(name or 'world')
        if communicator.is_empty():
            return Err(RuntimeError(f"communicator '{name or 'world'}' does not exit"))

        return Result.from_func(communicator.get().gather, sendobj=data, root=root_rank)

    def collective_allgather(self, data: Any, name: str = None) \
            -> Result[List[Any], Exception]:
        """
        collective communication for all-gathering

        Parameters
        ----------
        data : Any
            the data that has to be sent for gathering
        name : str, optional
            the name of the communicator to used. if not given, world will be used

        Returns
        -------
        Result[List[Any], Exception]
            the received data

        """

        self._check_mpi4py_loaded()

        communicator: Option[MPI.Comm] = self._communicators.get_value_option(name or 'world')
        if communicator.is_empty():
            return Err(RuntimeError(f"communicator '{name or 'world'}' does not exit"))

        return Result.from_func(communicator.get().allgather, sendobj=data)

    def get_rank_size(self, communicator: 'MPI.Comm') -> (int, int):
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

        self._check_mpi4py_loaded()

        rank: int = communicator.Get_rank()
        size: int = communicator.Get_size()

        return rank, size

    def get_local_rank(self) -> int:
        """Returns the local rank."""

        return self._local_rank

    def get_local_size(self) -> int:
        """Returns the local size."""

        return self._local_size

    def get_rank(self) -> int:
        """Returns the rank."""

        return self._rank

    def get_size(self) -> int:
        """Returns the size."""

        return self._size

    def get_node_rank(self) -> int:
        """Returns the node rank."""

        return self._node_rank


def init_with_config(config: ConfigParser):
    return _MPICommunicatorSingletonClass(config)


# this is an instance that everyone can use
# this is the only instance that everyone should use
# this instance has to be initialized and set at the beginning of the program
# then everyone should use this instance
mpi_communicator: _MPICommunicatorSingletonClass = init_with_config(config.mpi_config or ConfigParser({}))
