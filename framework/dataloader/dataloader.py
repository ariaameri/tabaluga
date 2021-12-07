import math
import pathlib
from abc import ABC, abstractmethod
from pathlib import Path
from ..base import base
from ..util.config import ConfigParser
from ..communicator import mpi
from typing import List, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from enum import Enum


# an enum to hold the content type of data to be loaded
class ContentTypes(Enum):

    FILE = 'file'
    MONGO = 'mongo'


# a mapping between the column concepts and their names
metadata_columns = {
    'folder_path': 'folder_path',
    'folder_name': 'folder_name',
    'file_name': 'file_name',
    'file_extension': 'file_extension',
    'path': 'path',
    'content_type': 'content_type',
    'syncable': 'syncable',
    'original_index': 'original_index',
    'data_raw': 'data_raw',
}


class DataManager(base.BaseEventManager, ABC):
    """This abstract class manages the DataLoader or DataLoader Managers.

    It is responsible to distribute the train/validation/test metadata among the data loaders."""

    def __init__(self, config: ConfigParser = None):
        """Initializer for the data manager.

        Parameters
        ----------
        config : ConfigParser
            The configurations for the data manager.
        """

        super().__init__(config)

        self._check_process_config()

        # Get the input type
        self._input_type: str = self._config.get_or_else('input_type', 'folder_path')

        # Get random seed and shuffle boolean
        self._seed = self._config.get_or_else('seed', None)
        self._shuffle: bool = self._config.get_or_else('shuffle', False)

        # Get test and validation ratios
        self._test_ratio: float = self._config.get_or_else('test_ratio', 0)
        self._val_ratio: float = self._config.get_or_else('val_ratio', 0)

        # Set batch size
        self.batch_size: int = self._config.get('batch_size')

        # Pandas data frame to hold the metadata of the data
        self.metadata: pd.DataFrame
        self.test_metadata: pd.DataFrame
        self.val_metadata: pd.DataFrame
        self.train_metadata: pd.DataFrame

        # Train, val, and test DataLoaderManager placeholders
        self.workers['train']: DataLoaderManager
        self.workers['validation']: DataLoaderManager
        self.workers['test']: DataLoaderManager

        # if we are the main rank, we always want to create a metadata
        if mpi.mpi_communicator.is_main_rank():
            # Create and modify general and train, val, test metadata
            self.create_metadata()
            self.modify_metadata()
            self._generate_train_val_test_metadata()

        # if in distributed mode, do the initial setup
        self.syncer: Optional[Syncer] = None
        if mpi.mpi_communicator.is_distributed():

            # make a syncer to sync metadata and data among processes
            self.syncer: Syncer = Syncer(self._config.get_or_empty('syncer'))

            if mpi.mpi_communicator.is_main_rank():
                # if we are the main rank, keep a copy of the original metadata
                self.metadata_original = self.metadata
                self.test_metadata_original = self.test_metadata
                self.val_metadata_original = self.val_metadata
                self.train_metadata_original = self.train_metadata

                # also, set the metadata for the syncer
                self.syncer.set_metadata(
                    metadata=self.metadata_original,
                )

            # first, run the initial synchronization between processes
            self.syncer.sync_initial()

            # now, sync the train/val/test metadata
            self.sync_train_val_test_metadata()

        # Create workers
        self.create_workers()

    def get_syncer(self):

        return self.syncer

    def sync_train_val_test_metadata(self):
        """Syncs the metadata across processes."""

        if mpi.mpi_communicator.is_distributed() is False:
            raise RuntimeError("not in distributed mode to sync train/val/test metadata.")

        if self.syncer.is_distributor():

            # if we have to shuffle, shuffle
            if self._shuffle is True:
                self.shuffle_each_original_metadata()

            # now pass the metadata to the syncer
            self.syncer.set_train_val_test_metadata(
                train_metadata=self.train_metadata_original,
                validation_metadata=self.val_metadata_original,
                test_metadata=self.test_metadata_original,
            )

        # get the metadata from the syncer
        train_metadata, val_metadata, test_metadata = self.syncer.sync_train_val_test_metadata()

        # reset the indices
        train_metadata = MetadataManipulator.reset_level_0_indices(train_metadata)
        val_metadata = MetadataManipulator.reset_level_0_indices(val_metadata)
        test_metadata = MetadataManipulator.reset_level_0_indices(test_metadata)

        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.test_metadata = test_metadata

    def _check_process_config(self) -> None:
        """Reconfigures the config file for this class."""

        # Spread multithreading one level down
        # Didn't choose only one level down because 'train' and 'val' entries might not exist
        if self._config.get_option('multithreading').is_defined():
            self._config = self._config.update(
                {},
                {'$set_on_insert': {
                    'train.multithreading': self._config.get('multithreading'),
                    'val.multithreading': self._config.get('multithreading'),
                }})

    def create_metadata(self) -> None:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Read and create the data metadata
        if self._input_type == 'folder_path':
            metadata_generator = FolderReader(self._config)
        elif self._input_type == 'mongo':
            raise NotImplementedError
        else:
            raise ValueError(f"input type of {self._input_type} not recognized")

        # get the metadata
        self.metadata = metadata_generator.build_and_get_metadata()

    def modify_metadata(self) -> None:
        """Modifies the metadata that this instance holds."""

        metadata_manipulator = MetadataManipulator(metadata=self.metadata)

        # regroup the metadata
        self.metadata = metadata_manipulator.modify()

    def _generate_train_val_test_metadata(self) -> None:
        """The method creates train, validation, and test metadata."""

        # Get count of each set
        total_data_count = self.metadata.index.get_level_values(0).unique().size
        test_count = int(total_data_count * self._test_ratio)
        val_count = int((total_data_count - test_count) * self._val_ratio)
        train_count = total_data_count - test_count - val_count

        # Find the indices of each set
        np.random.seed(self._seed)
        indices = np.arange(total_data_count) \
            if self._shuffle is False \
            else np.random.permutation(total_data_count)
        test_indices = indices[:test_count]
        val_indices = indices[test_count:(test_count+val_count)]
        train_indices = indices[(test_count+val_count):]

        # Create the train, validation, and test metadata
        self.test_metadata = self.metadata.loc[test_indices]
        self.val_metadata = self.metadata.loc[val_indices]
        self.train_metadata = self.metadata.loc[train_indices]

        # Update the column names of the data frames
        assert(metadata_columns['original_index'] == 'original_index')  # this is because df.assign can only get kwargs
        [self.train_metadata, self.val_metadata, self.test_metadata] = \
            [
                df
                .assign(original_index=df.index.get_level_values(0))
                .rename(
                    index={
                        key: value
                        for value, key
                        in enumerate(df.index.get_level_values(0).unique(), start=0)
                    }
                )
                for df
                in [self.train_metadata, self.val_metadata, self.test_metadata]
            ]

    def shuffle_each_metadata(self) -> None:
        """Shuffles each of the metadata separately."""

        # set the seed
        np.random.seed(self._seed)

        # unfortunately, because python does not have referencing, we cannot iterate over values and have to shuffle
        # each metadata separately

        # shuffle train_metadata
        metadata_count = self.train_metadata.index.get_level_values(0).unique().size
        indices = np.arange(metadata_count) \
            if self._shuffle is False \
            else np.random.permutation(metadata_count)
        self.train_metadata = self.train_metadata[indices]

        # shuffle val_metadata
        metadata_count = self.val_metadata.index.get_level_values(0).unique().size
        indices = np.arange(metadata_count) \
            if self._shuffle is False \
            else np.random.permutation(metadata_count)
        if len(indices) > 0:
            self.val_metadata = self.val_metadata[indices]

        # shuffle val_metadata
        metadata_count = self.test_metadata.index.get_level_values(0).unique().size
        indices = np.arange(metadata_count) \
            if self._shuffle is False \
            else np.random.permutation(metadata_count)
        if len(indices) > 0:
            self.test_metadata = self.test_metadata[indices]

    def shuffle_each_original_metadata(self) -> None:
        """Shuffles each of the metadata separately."""

        # only shuffle if we are distributed and we are the main rank
        if mpi.mpi_communicator.is_distributed() is False or mpi.mpi_communicator.is_main_rank() is False:
            return

        # set the seed
        np.random.seed(self._seed)

        # unfortunately, because python does not have referencing, we cannot iterate over values and have to shuffle
        # each metadata separately

        # shuffle train_metadata_original
        metadata_count = self.train_metadata_original.index.get_level_values(0).unique().size
        indices = np.arange(metadata_count) \
            if self._shuffle is False \
            else np.random.permutation(metadata_count)
        self.train_metadata_original = self.train_metadata_original[indices]

        # shuffle val_metadata_original
        metadata_count = self.val_metadata_original.index.get_level_values(0).unique().size
        indices = np.arange(metadata_count) \
            if self._shuffle is False \
            else np.random.permutation(metadata_count)
        if len(indices) > 0:
            self.val_metadata_original = self.val_metadata_original[indices]

        # shuffle val_metadata_original
        metadata_count = self.test_metadata_original.index.get_level_values(0).unique().size
        indices = np.arange(metadata_count) \
            if self._shuffle is False \
            else np.random.permutation(metadata_count)
        if len(indices) > 0:
            self.test_metadata_original = self.test_metadata_original[indices]

    def set_batch_size(self, batch_size: int) -> None:
        """Sets the batch size and thus finds the total number of batches in one epoch.

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        self.batch_size = batch_size

        # Set batch size of all the workers
        for worker in self.workers:
            worker.set_batch_size(batch_size)


class MetadataManipulator(base.BaseWorker):
    """Class to modify and manipulate metadata."""

    def __init__(self, metadata: pd.DataFrame, config: ConfigParser = None):

        super().__init__(config)

        # metadata storage
        self.metadata = metadata

    def modify(self) -> pd.DataFrame:
        """Runs the whole modification pipeline and returns the result."""

        # regroup based on the file name
        metadata = self.regroup_metadata(criterion=metadata_columns['file_name'])

        return metadata

    def regroup_metadata(self, criterion=None) -> pd.DataFrame:
        """Groups the metadata.

        Each group of data (e.g. containing data and label) should have.
        Each group must have its own unique index, where indices are range.
        Each group will be recovered by metadata.loc[index]

        Parameters
        ----------
        criterion : str or List[str]
            The name of the columns based on which the metadata should be categorized

        Returns
        -------
        pd.DataFrame
            the result

        """

        if criterion is None:
            return self.metadata

        # check if criterion is a column concept
        if isinstance(criterion, str) and criterion not in self.metadata.columns:
            raise ValueError(f"criterion '{criterion}' does not exists as a metadata column.")
        elif isinstance(criterion, list) and any([(c not in self.metadata.columns) for c in criterion]):
            raise ValueError(f"part of criterion '{criterion}' does not exists as a metadata column.")

        # Group based on the criterion
        metadata = self.metadata.groupby(criterion).apply(lambda x: x.reset_index(drop=True))

        # Rename the indices to be range
        # Also rename the index level 0 name to be 'index' (instead of `criterion`)
        metadata = self.reset_level_0_indices(metadata)
        metadata.index.names = [None, *metadata.index.names[1:]]

        self.metadata = metadata

        return metadata

    @staticmethod
    def reset_level_0_indices(metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Resets the level 0 indices to start from 0 and returns the result.

        Parameters
        ----------
        metadata : pd.DataFrame
            the data frame to reset the indices

        Returns
        -------
        pd.DataFrame
            the result

        """

        metadata = metadata.rename(
            index={key: value for value, key in enumerate(metadata.index.get_level_values(0).unique(), start=0)}
        )

        return metadata


class FolderReader(base.BaseWorker):
    """This class is a helper class that reads metadata from a path (folder)."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # Folders containing the data
        self._folders: List[str] = []

        # check and populate the folder metadata
        self._check_populate_folder_metadata()

    def _check_populate_folder_metadata(self):
        """Checks given folders for sanity and existence."""

        # Folders containing the data
        self._folders: List[str] = self._config.get_or_else('folders', None)

        # Check if folder list is given
        if self._folders is None:
            raise ValueError("No folders given to read files from!")
        for folder in self._folders:
            if Path(folder).exists() is False:
                raise FileNotFoundError(f'The folder {folder} does not exist!')

    def build_and_get_metadata(self) -> pd.DataFrame:
        """
        This method goes over the specified folders, read files and creates and returns a pandas data frame from them.

        Returns
        -------
        pd.DataFrame
            pandas dataframe of the information

        """

        # File names in nested lists
        # only one level deep
        file_nested_paths = [list(Path(folder).iterdir()) for folder in self._folders]

        # Flatten the file names and make absolute paths
        file_paths = [file_path
                      for file_paths in file_nested_paths
                      for file_path in file_paths]
        # Check each file based on the criteria
        file_paths = [file_path for file_path in file_paths if self._check_file(file_path)]
        # Retrieve the folder path and file names
        folder_paths = [file_name.parent for file_name in file_paths]
        folder_names = [folder_path.name for folder_path in folder_paths]
        file_names = [file_path.stem for file_path in file_paths]
        file_extensions = [file_path.suffix.lower() for file_path in file_paths]

        # Create data frame of all the files in the folder
        metadata = pd.DataFrame({
            metadata_columns['folder_path']: [str(item) for item in folder_paths],
            metadata_columns['folder_name']: folder_names,
            metadata_columns['file_name']: file_names,
            metadata_columns['file_extension']: file_extensions,
            metadata_columns['path']: [str(item) for item in file_paths],
            metadata_columns['content_type']: ContentTypes.FILE.value,
            metadata_columns['syncable']: True,
        })

        return metadata

    def _check_file(self, file_path: Path) -> bool:
        """"Helper function to check a single file.

        Parameters
        ----------
        file_path : Path
            The path of the file
        """

        # Check that the file is not a folder
        check = file_path.is_file()

        # Check criteria for the file name
        check &= self._filter_file_name(file_path.name)

        return check

    def _filter_file_name(self, file_name: str) -> bool:
        """"Helper function to filter a single file based on its name and criteria.

        Parameters
        ----------
        file_name : str
            The path of the file
        """

        check = False if file_name.startswith('.') else True
        # Take care of MacOS special files
        check &= file_name not in ['Icon\r']

        return check


class Syncer(base.BaseWorker):
    """Class to synchronize the data among processes in distributed mode."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # cache some distributed-mode-related data
        self.rank: int = mpi.mpi_communicator.get_rank()
        self.dist_size: int = mpi.mpi_communicator.get_size()
        self.is_main_rank: bool = mpi.mpi_communicator.is_main_rank()
        self.distributor = True if self.is_main_rank is True else False
        self.receiver = True if self.is_main_rank is False else False

        # get the batch size for syncing
        self.batch_size = self._config.get_or_else('batch_size', 16)

        # book keeping
        self.thread_count: int = self._config.get_or_else('multithreading_count', 10)
        # whether we should force the data syncing initially
        self.force_sync = self._config.get_or_else('force_sync', False)

        # placeholder for the metadata
        self.metadata: Optional[pd.DataFrame] = None
        self.train_metadata: Optional[pd.DataFrame] = None
        self.validation_metadata: Optional[pd.DataFrame] = None
        self.test_metadata: Optional[pd.DataFrame] = None

        # get a new mpi communicator
        self.mpi_comm_name = 'data_manager'
        mpi_comm = mpi.mpi_communicator.get_or_create_communicator('data_manager')

        # make a new communicator for broadcasting
        self.mpi_comm_main_local_name = 'data_manager_main_local'
        self.mpi_comm_main_local = mpi.mpi_communicator.split(
            communicator=mpi_comm,
            color=0 if mpi.mpi_communicator.is_main_local_rank() is True else 1
        )
        mpi.mpi_communicator.register_communicator(self.mpi_comm_main_local, name=self.mpi_comm_main_local_name)

    def set_metadata(self, metadata: pd.DataFrame):
        """
        Sets the complete metadata to be used for initial synchronization.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata data frame

        Returns
        -------
        self

        """

        self.metadata = metadata

        return self

    def set_train_val_test_metadata(
            self,
            train_metadata: pd.DataFrame,
            validation_metadata: pd.DataFrame,
            test_metadata: pd.DataFrame,
    ):
        """
        Sets the train/val/test metadata..

        Parameters
        ----------
        train_metadata : pd.DataFrame
            the train metadata data frame
        validation_metadata : pd.DataFrame
            the validation metadata data frame
        test_metadata : pd.DataFrame
            the test metadata data frame

        Returns
        -------
        self

        """

        self.train_metadata = train_metadata
        self.validation_metadata = validation_metadata
        self.test_metadata = test_metadata

        return self

    def is_distributor(self) -> bool:
        """Returns a boolean regarding whether we are the distributor."""

        return self.distributor

    def is_receiver(self) -> bool:
        """Returns a boolean regarding whether we are a receiver."""

        return self.receiver

    def sync_initial(self):
        """
        syncs everything.
        this method should be called the first time syncing is happening.

        Returns
        -------

        """

        # sync the whole metadata
        metadata: pd.DataFrame = self._sync_metadata()

        # sync the local data that is syncable
        self._sync_local_data(metadata)

    def _sync_metadata(self) -> pd.DataFrame:
        """Syncs the metadata and returns it."""

        # make sure we have all the metadata if we are the main rank
        if self.distributor:
            if self.metadata is None:
                raise ValueError("metadata are not set in the main rank. please set them before calling this function.")

        # let everyone get the metadata
        metadata: pd.DataFrame = \
            mpi.mpi_communicator.collective_bcast(
                data=self.metadata,
                root_rank=0,
                name=self.mpi_comm_name,
            )

        return metadata

    def _sync_local_data(self, metadata: pd.DataFrame = None):
        """
        Reads the metadata and syncs the local data within.

        Parameters
        ----------
        metadata : pd.DataFrame, optional
            metadata to use for syncing. if not given, will default to the original metadata in the instance

        Returns
        -------

        """

        # get the metadata
        metadata = metadata if metadata is not None else self.metadata

        # get the portion of the metadata that is syncable
        syncable_selection = metadata[metadata_columns['syncable']] == True
        metadata = metadata[syncable_selection]

        if self.force_sync is True:
            # force sync if necessary
            self._sync_local_data_broadcast(metadata)
        else:
            # sync selectively
            self._sync_local_data_selective(metadata)

    def _sync_local_data_broadcast(self, metadata: pd.DataFrame):
        """
        Broadcasts all the local data to all the processes and makes sure everyone has it.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for syncing

        """

        # we should not continue if we are the main local rank
        if mpi.mpi_communicator.is_main_local_rank() is False:
            return

        # get the rank and size
        rank, size = \
            mpi.mpi_communicator.get_rank_size(mpi.mpi_communicator.get_communicator(self.mpi_comm_main_local_name))

        # log
        if self.is_distributor():
            self._log.info(f"syncing {len(metadata)} local data via force broadcasting with {size-1} workers")

        # make a thread pool ot be used for loading the data
        thread_pool = ThreadPoolExecutor(self.thread_count)

        # go over the data in batch_size chunks
        for start_idx in range(0, len(metadata), self.batch_size):

            # load if we are the distributor
            if self.distributor:
                # first, load the data into a new dataframe
                metadata_updated: Optional[pd.DataFrame] = \
                    self._load_local_data_raw(
                        metadata.iloc[start_idx:(start_idx+self.batch_size)],
                        thread_pool,
                    )
            else:
                metadata_updated = None

            # now broadcast
            metadata_updated: pd.DataFrame = \
                mpi.mpi_communicator.collective_bcast(
                    data=metadata_updated,
                    root_rank=0,
                    name=self.mpi_comm_main_local_name,
                )

            # save only if we are not the main rank
            if self.receiver:
                self._save_local_data_raw(metadata_updated, thread_pool)

        # log
        if self.is_distributor():
            self._log.info(f"done syncing {len(metadata)} local data via force broadcasting with {size-1} workers")

    def _check_local_data(self, metadata: pd.DataFrame, thread_pool: ThreadPoolExecutor):
        """
        Checks the local machine for existence of data in the given data frame and returns a new data frame with the
        items that do not exist.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for checking for data
        thread_pool : ThreadPoolExecutor
            the thread pool to use for checking the data

        Returns
        -------
        pd.DataFrame
            a new metadata with rows of the original data frame whose data do not exist on the local machine

        """

        def check_single_data_raw(path: str) -> bool:
            """Checks if the given path exists and returns a boolean result of it."""

            dest = pathlib.Path(path)

            return dest.exists() and dest.is_file()

        # find a selector of the metadata for the files that do exist
        selector = list(
            thread_pool.map(check_single_data_raw, metadata[metadata_columns['path']])
        )
        # not the selection
        selector = [not item for item in selector]

        # get the new metadata and return it based on the criteria
        return metadata.iloc[selector]

    def _load_local_data_raw(self, metadata: pd.DataFrame, thread_pool: ThreadPoolExecutor) -> pd.DataFrame:
        """
        Loads the local data and returns them.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for loading the data
        thread_pool : ThreadPoolExecutor
            the thread pool to use for loading the data

        Returns
        -------
        pd.DataFrame
            a new metadata with the data loaded in it

        """

        def load_single_data_raw(path: str) -> bytes:
            """Loads a single raw data from the given path in binary form and returns it."""

            with open(path, 'rb') as file:
                return file.read()

        # Load the data
        data = list(
            thread_pool.map(load_single_data_raw, metadata[metadata_columns['path']])
        )

        # assign them to a new column
        assert(metadata_columns['data_raw'] == 'data_raw')  # this is because df.assign can only get kwargs
        metadata = metadata.assign(data_raw=data)

        return metadata

    def _save_local_data_raw(self, metadata: pd.DataFrame, thread_pool: ThreadPoolExecutor) -> None:
        """
        saves the data

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for saving the data
        thread_pool : ThreadPoolExecutor
            the thread pool to use for saving the data

        """

        def save_single_data_raw(path: str, data: bytes) -> None:
            """saves a single raw data to the given path in binary form."""

            # create the folder if it does not exist
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

            # now save the result
            with open(path, 'wb') as file:
                file.write(data)

        # save the data
        list(
            thread_pool.map(
                lambda x: save_single_data_raw(x[0], x[1]),
                zip(metadata[metadata_columns['path']], metadata[metadata_columns['data_raw']])
            )
        )

    def _sync_local_data_selective(self, metadata: pd.DataFrame):
        """
        Syncs only the missing portion of the local data with all the processes and makes sure everyone has all.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for syncing

        """

        # make a thread pool to be used for the subthreads
        thread_pool = ThreadPoolExecutor(self.thread_count)

        if self.distributor:
            # make a thread pool to be used for interacting with other ranks
            thread_communicator_pool = ThreadPoolExecutor(self.thread_count)

            list(
                thread_communicator_pool.map(
                    lambda rank: self._sync_local_data_with_rank(rank, metadata, thread_pool),
                    range(1, self.dist_size)
                )
            )
        else:
            # just receive the data!
            self._sync_local_data_with_rank(self.rank, metadata, thread_pool)

    def _sync_local_data_with_rank(self, rank: int, metadata: pd.DataFrame, thread_pool: ThreadPoolExecutor):
        """
        Syncs only the missing portion of the local data with the given rank.

        Parameters
        ----------
        rank : int
            destination rank
        metadata : pd.DataFrame
            the metadata to use for syncing
        thread_pool : ThreadPoolExecutor
            the thread pool to use for the operations

        """

        if self.receiver:
            # find out the missing data and send it to the main rank
            metadata_missing_files = self._check_local_data(metadata, thread_pool)
            mpi.mpi_communicator.p2p_send(
                data=metadata_missing_files,
                destination=0,
                name=self.mpi_comm_name,
            )
        elif self.distributor:
            # get the data sent by this rank
            metadata_missing_files = \
                mpi.mpi_communicator.p2p_receive(
                    source=rank,
                    name=self.mpi_comm_name,
                )
        else:
            raise RuntimeError("we should not have ended up here!")

        # log
        if self.is_distributor() and len(metadata_missing_files) > 0:
            self._log.info(f"syncing {len(metadata_missing_files)} local data selectively with rank {rank}")

        # go over the data in batch_size chunks
        for start_idx in range(0, len(metadata_missing_files), self.batch_size):

            # load if we are the distributor
            if self.distributor:
                # first, load the data into a new dataframe
                metadata_updated: Optional[pd.DataFrame] = \
                    self._load_local_data_raw(
                        metadata_missing_files.iloc[start_idx:(start_idx + self.batch_size)],
                        thread_pool,
                    )
                # now send to the rank
                mpi.mpi_communicator.p2p_send(
                    data=metadata_updated,
                    destination=rank,
                    tag=start_idx,
                    name=self.mpi_comm_name,
                )
            else:
                # just receive the data and store them
                metadata_updated = \
                    mpi.mpi_communicator.p2p_receive(
                        source=0,
                        tag=start_idx,
                        name=self.mpi_comm_name,
                    )
                self._save_local_data_raw(metadata_updated, thread_pool)

        # log
        if self.is_distributor() and len(metadata_missing_files) > 0:
            self._log.info(f"done syncing {len(metadata_missing_files)} local data selectively with rank {rank}")

    def sync_train_val_test_metadata(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Splits the train/val/test metadata into chunks for each process, scatters them, and returns each chunk.
        It should be noted that after this operation, all nodes will have exactly the same amount of data.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            the data frames for the metadata of train, val, and test

        """

        if self.is_distributor():

            if any([item is None for item in [self.train_metadata, self.validation_metadata, self.test_metadata]]):
                raise ValueError("metadata are not set in the main rank. please set them before calling this function.")

            # split the training metadata
            train_zero_level_indices = self.train_metadata.index.get_level_values(0).unique()
            chunk_size = math.floor(train_zero_level_indices.size / self.dist_size)
            # note that the last chunk, which has a number of data not equal to the other chunks, will be dropped and
            # not synced
            train_split_indices = \
                [
                    train_zero_level_indices[start_idx:(start_idx+chunk_size)]
                    for start_idx
                    in range(0, train_zero_level_indices.size, chunk_size)
                ]
            train_split_metadata = \
                [
                    self.train_metadata.loc[chunk_indices]
                    for chunk_indices
                    in train_split_indices
                ]
            # remove the extra training data
            train_split_metadata = train_split_metadata[:self.dist_size]

            # split the validation metadata
            # we only want the distributor to have the validation data and not the others
            val_split_metadata = [self.validation_metadata]
            val_split_metadata.extend([
                self.validation_metadata.iloc[:0]  # this will surely result in an empty dataframe
                for _
                in range(1, self.dist_size)
            ])

            # split the test metadata
            # we only want the distributor to have the test data and not the others
            test_split_metadata = [self.test_metadata]
            test_split_metadata.extend([
                self.test_metadata.iloc[:0]  # this will surely result in an empty dataframe
                for _
                in range(1, self.dist_size)
            ])
        else:
            train_split_metadata = None
            val_split_metadata = None
            test_split_metadata = None

        # scatter all the metadata
        train_metadata = \
            mpi.mpi_communicator.collective_scatter(
                data=train_split_metadata,
                root_rank=0,
                name=self.mpi_comm_name
            )
        val_metadata = \
            mpi.mpi_communicator.collective_scatter(
                data=val_split_metadata,
                root_rank=0,
                name=self.mpi_comm_name
            )
        test_metadata = \
            mpi.mpi_communicator.collective_scatter(
                data=test_split_metadata,
                root_rank=0,
                name=self.mpi_comm_name
            )

        return train_metadata, val_metadata, test_metadata


class DataLoaderManager(base.BaseEventManager, ABC):
    """This abstract class manages the data loaders and gets input from DataManager."""

    def __init__(self, metadata: pd.DataFrame, config: ConfigParser = None):
        """Initializer for data loader manager.

        Parameters
        ----------
        metadata : pd.DataFrame
            The metadata for the data to be loaded
        config : ConfigParser
            The configuration for the instance
        """

        super().__init__(config)

        # Set the metadata
        self.metadata = metadata

        # Modify the metadata
        self.modify_metadata()

        # Book keeping for iterator, batch size and number of iterations (batches in an epoch)
        self._iterator_count = 0
        self.batch_size: int = -1
        self.number_of_iterations: int = -1

        # Create workers
        self.create_workers()

    def modify_metadata(self) -> None:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Make a selection of the metadata
        selection = [self._check_file(file_path) for file_path in self.metadata[metadata_columns['path']]]

        # Update the metadata
        self.metadata = self.metadata.iloc[selection]

    def _regroup_metadata(self, criterion=None, reset_index: bool = True) -> None:
        """Groups the metadata.

        Each group of data (e.g. containing data and label) should have.
        Each group must have its own unique index, where indices are range.
        Each group will be recovered by metadata.loc[index]

        Parameters
        ----------
        criterion : str or List[str]
            The name of the columns based on which the metadata should be categorized
        reset_index : bool, optional
            Whether or not to reset the level-0 indexing to range. If not given, will reset

        """

        if criterion is None:
            return

        # Group based on the criterion
        metadata = self.metadata.groupby(criterion).apply(lambda x: x.reset_index(drop=True))

        # Rename the indices to be range
        # Also rename the index level 0 name to be 'index' (instead of criterion)
        if reset_index:
            metadata = metadata.rename(
                index={key: value for value, key in enumerate(metadata.index.get_level_values(0).unique(), start=0)}
            )
            metadata.index.names = [None, *metadata.index.names[1:]]

        self.metadata = metadata

    def _check_file(self, file_path: str) -> bool:
        """"Helper function to check a single file.

        Parameters
        ----------
        file_path : str
            The path of the file
        """

        # Check criteria for the file name
        check = self._filter_file_name(file_path.split('/')[-1])

        return check

    def _filter_file_name(self, file_name: str) -> bool:
        """"Helper function to filter a single file based on its name and criteria.

        Parameters
        ----------
        file_name : str
            The path of the file
        """

        return True

    def set_batch_size(self, batch_size: int) -> None:
        """Sets the batch size and thus finds the total number of batches in one epoch.

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        self.batch_size = batch_size

        # Book keeping for the returned number of iterations from workers
        number_of_iterations_workers = []

        # Set batch size of all the workers and get their number_of_iterations
        for worker in self.workers:
            number_of_iterations_workers.append(worker.set_batch_size(batch_size))

        # Check if all the returned number of iterations are the same
        assert all([no == number_of_iterations_workers[0] for no in number_of_iterations_workers]) is True, \
            'Returned number_of_iterations from all DataLoader\'s must be the same.'

        # Set the number of iterations
        self.number_of_iterations = number_of_iterations_workers[0]

    def __len__(self) -> int:
        """Returns the length of the instance.

        IMPORTANT: The length of all workers must be the same.

        """

        return self.number_of_iterations

    def __iter__(self):
        """Returns an iterable, self."""

        return self

    def __next__(self):
        """Returns the next set of data.

        Returns
        -------
        A collection of next set of data

        """

        # if the batch size is more that the amount of data left, go to beginning and return None
        if self._iterator_count > self.number_of_iterations:
            self._iterator_count = 0
            return StopIteration

        # Load the data
        data = self.__getitem__(self._iterator_count)

        return data

    @abstractmethod
    def __getitem__(self, item: int):
        """Returns the item-th batch of the data.

        Parameters
        ----------
        item : int
            The index of the batch of the data to be returned

        Returns
        -------
        A list of the data loaded

        """

        raise NotImplementedError


class DataLoader(base.BaseEventWorker, ABC):
    """This abstract class loads the data."""

    def __init__(self, metadata: pd.DataFrame, config: ConfigParser = None):
        """Initializer for data loader.

        Parameters
        ----------
        metadata : pd.DataFrame
            The metadata for the data to be loaded
        config : ConfigParser
            The configuration for the instance
        """

        super().__init__(config)

        # Set the metadata
        self.metadata: pd.DataFrame = metadata

        # Flag for if we should load the data with multithreading
        self.multithreading: bool = self._config.get_or_else('multithreading', True)
        self.thread_count: int = self._config.get_or_else('multithreading_count', 5)

        # Book keeping for the batch size and thus the number of iterations (batches) in each epoch
        self.batch_size: int = -1
        self.number_of_iterations: int = -1

        # Book keeping for iterator
        self._iterator_count = 0

    def modify_metadata(self) -> None:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Make a selection of the metadata
        selection = [self._check_file(file_path) for file_path in self.metadata[metadata_columns['path']]]

        # Update the metadata
        self.metadata = self.metadata.iloc[selection]

    def _check_file(self, file_path: str) -> bool:
        """"Helper function to check a single file.

        Parameters
        ----------
        file_path : str
            The path of the file
        """

        # Check criteria for the file name
        check = self._filter_file_name(file_path.split('/')[-1])

        return check

    def _filter_file_name(self, file_name: str) -> bool:
        """"Helper function to filter a single file based on its name and criteria.

        Parameters
        ----------
        file_name : str
            The path of the file
        """

        return True

    def set_batch_size(self, batch_size: int) -> int:
        """Sets the batch size and thus finds the total number of batches in one epoch.

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        self.batch_size = batch_size
        self.number_of_iterations = len(self.metadata) // batch_size

        return self.number_of_iterations

    def load_data(self, metadata: pd.DataFrame):
        """Loads data provided in the metadata data frame.

        Parameters
        ----------
        metadata : pd.DataFrame
            Panda's data frame containing the data metadata to be loaded

        Returns
        -------
        Loaded data

        """

        # Load the data
        # Load data with multithreading
        if self.multithreading is True:
            # Load the data with threads
            thread_pool = ThreadPoolExecutor(self.thread_count)
            data = list(
                    thread_pool.map(lambda row: self.load_single_data(row[1]), metadata.iterrows())
                )
        else:
            data = [
                self.load_single_data(row[1]) for row in metadata.iterrows()
            ]

        data = self.load_data_post(data)

        return data

    @abstractmethod
    def load_single_data(self, row: pd.Series):
        """Loads a single file whose path is given.

        Parameters
        ----------
        row : pd.Series
            Pandas row entry of that specific data

        Returns
        -------
        Loaded data

        """

        raise NotImplementedError

    @abstractmethod
    def load_data_post(self, data: List):
        """Reforms the data already loaded into the desired format.

        Parameters
        ----------
        data : List
            The already loaded data in a list

        Returns
        -------
        Loaded data in the desired format

        """

        raise NotImplementedError

    def __len__(self) -> int:
        """Gives the total number of iterations this data loader will go through.

        Returns
        -------
        Total number of batches in each epoch

        """

        return self.number_of_iterations

    def __iter__(self):
        """Returns an iterable, self."""

        return self

    def __next__(self):
        """Returns the next set of data.

        Returns
        -------
        A collection of next set of data

        """

        # if the batch size is more that the amount of data left, go to beginning and return None
        if self._iterator_count > self.number_of_iterations:
            self._iterator_count = 0
            raise StopIteration

        # Load the data
        data = self.__getitem__(self._iterator_count)

        return data

    def __getitem__(self, item: int):
        """Returns the item-th batch of the data.

        Parameters
        ----------
        item : int
            The index of the batch of the data to be returned

        Returns
        -------
        A list of the data loaded

        """

        # Check if item count is sensible
        assert item < self.number_of_iterations, \
            f'Requested number of images to be loaded goes beyond the end of available data.'

        # Find the corresponding metadata
        begin_index = item * self.batch_size
        metadata = self.metadata.iloc[begin_index:(begin_index + self.batch_size)]

        # Load the images
        data = self.load_data(metadata)

        return data




































