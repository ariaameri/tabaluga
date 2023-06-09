import concurrent.futures
import math
import pathlib
import re
import select
import threading
from abc import ABC, abstractmethod
from pathlib import Path
import colored
from jointstemplant.util.util import OnAndEnabled
from ..base import base
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.data_muncher import UPDATE_MODIFIERS as UM, UPDATE_OPERATIONS as UO, UPDATE_CONDITIONALS as UC
from ..util.data_muncher import FILTER_OPERATIONS as FO, FILTER_MODIFIERS as FM
from ..util.option import Some, Option, nothing
from ..communicator import mpi
from typing import List, Optional, Callable, Any, Dict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from ..util.result import Result, Err, Ok


# an enum to hold the content type of data to be loaded
class ContentTypes(Enum):

    FILE = 'file'
    MONGO = 'mongo'


# a mapping between the column concepts and their names
metadata_columns = {
    'folder_path': 'folder_path',
    'folder_parent_path': 'folder_parent_path',
    'folder_name': 'folder_name',
    'file_name': 'file_name',
    'file_extension': 'file_extension',
    'path': 'path',
    'content_type': 'content_type',
    'metadata_sync_choice': 'metadata_sync_choice',
    'syncable': 'syncable',
}
_metadata_columns_internal = {
    'original_index': 'original_index',
    'data_raw': 'data_raw',
    '__criterion': '__criterion',
}

metadata_columns_COCO = {
    'coco_id': 'coco_id',
    'coco_dataset_id': 'coco_dataset_id',
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
        _seed = self._config.get_or_else('seed', None)
        self._seed = _seed if _seed is not None else np.random.randint(0, 100000)
        self._shuffle: bool = self._config.get_or_else('shuffle.enabled', False)
        self._shuffle_add_node_rank: bool = self._config.get_or_else('shuffle.add_node_rank', True)

        # Get test and validation ratios
        self._test_ratio: float = self._config.get_or_else('test_ratio', 0)
        self._val_ratio: float = self._config.get_or_else('val_ratio', 0)

        # Set batch size
        self.batch_size: int = self._config.get('batch_size')
        # log
        self._log.info(
            f"performing data loading with batch size of "
            f"{colored.fg('cyan')}{self.batch_size}{colored.attr('reset')}"
        )
        # bookkeeping for the training batch size
        self._batch_size_train: int = self.batch_size
        # bookkeeping for the val/test batch size
        self._batch_size_val: int = self._config.get_or_else('batch_size_val', None)
        self._batch_size_test: int = self._config.get_or_else('batch_size_test', None)

        # build the metadata generator
        self.metadata_generator = self._build_metadata_generator()

        # Pandas data frame to hold the metadata of the data
        self.train_metadata: pd.DataFrame
        self.val_metadata: pd.DataFrame
        self.test_metadata: pd.DataFrame
        if (res := self._create_metadata()).is_err():
            raise res.get_err()
        self.train_metadata, self.val_metadata, self.test_metadata = res.get()

        # keep a copy of the original metadata
        self.test_metadata_original = self.test_metadata
        self.val_metadata_original = self.val_metadata
        self.train_metadata_original = self.train_metadata

        # Train, val, and test DataLoaderManager placeholders
        self.workers['train']: DataLoaderManager
        self.workers['validation']: DataLoaderManager
        self.workers['test']: DataLoaderManager

        # if in distributed mode, do the initial setup
        if mpi.mpi_communicator.is_distributed():

            # make a syncer to sync metadata among processes
            self._metadata_syncer: MetadataSyncer = MetadataSyncer(self._config.get_or_empty('metadata_syncer'))

            # now, sync the train/val/test metadata
            if (res := self._sync_all_metadata_distributed()).is_err():
                self._log.error(f"error while syncing the metadata with error of {res.get_err()}")
                raise RuntimeError("error while syncing metadata")
            self.train_metadata, self.val_metadata, self.test_metadata = res.get()

        # Create workers
        self.create_workers()

        # the shared multithreading pool
        self._multithreading = self._config.get_or_else('multithreading', False)
        self.thread_pool: Optional[ThreadPoolExecutor] = None

        if self._multithreading:
            self._multithreading_count = \
                self.metadata_generator.get_best_multithreading_count(self.train_metadata)\
                    .get_or_else(self._config.get_or_else('multithreading_count', 5))

            if self._multithreading is True:
                self.thread_pool = ThreadPoolExecutor(
                    self._multithreading_count, thread_name_prefix="tabaluga-datamanager-thread-pool"
                )
        self._distribute_shared_multithreading_pool()

        # set the batch sizes
        self.set_batch_size(self.batch_size)

    def _distribute_shared_multithreading_pool(self):
        """distributes the shared multithreading pool among all workers."""

        if self._multithreading:
            for worker in self._get_dataloader_workers():
                worker.set_shared_multithreading_pool(self.thread_pool)

    @abstractmethod
    def _get_dataloader_workers(self) -> List:
        """Returns the list of workers that the shared multithreading pool can be set to."""

        pass

    def get_syncer(self):

        return self._metadata_syncer

    def _sync_all_metadata_distributed(self, rand_seed_add: int = 0) -> Result[Any, Exception]:
        """
        Syncs the metadata across processes.

        Parameters
        ----------
        rand_seed_add : int, optional
            number to add with the random seed generator to have different random number generation each time

        Returns
        -------
        Result[(pd.DataFrame, pd.DataFrame, pd.DataFrame), Exception]
            train/val/test metadata
        """

        train_metadata_original = self.train_metadata_original
        val_metadata_original = self.val_metadata_original
        test_metadata_original = self.test_metadata_original

        if mpi.mpi_communicator.is_distributed() is False:
            return Ok((train_metadata_original, val_metadata_original, test_metadata_original))

        if self._metadata_syncer.is_distributor():

            # if we have to shuffle, shuffle
            if self._shuffle is True:
                train_metadata_original = self._shuffle_metadata(train_metadata_original, rand_seed_add)
                val_metadata_original = self._shuffle_metadata(val_metadata_original, rand_seed_add)
                test_metadata_original = self._shuffle_metadata(test_metadata_original, rand_seed_add)

        # get the metadata from the syncer
        if (res := self._metadata_syncer.sync_train_val_test_metadata(
                train_metadata_original,
                val_metadata_original,
                test_metadata_original,
                self.batch_size,
        )).is_err():
            self._log.error(f"error happened while syncing the metadata between nodes with error of {res.get_err()}")
            return Err(RuntimeError("failed syncing metadata"))
        train_metadata, val_metadata, test_metadata = res.get()

        # reset the indices
        train_metadata = MetadataManipulator.reset_level_0_indices(train_metadata)
        val_metadata = MetadataManipulator.reset_level_0_indices(val_metadata)
        test_metadata = MetadataManipulator.reset_level_0_indices(test_metadata)

        return Ok((train_metadata, val_metadata, test_metadata))

    def _check_process_config(self) -> None:
        """Reconfigures the config file for this class."""

        pass

    def _build_folder_reader(self) -> 'FolderReader':
        """Builds and return the FolderReader instance for this instance."""

        return FolderReader(self._config.get_or_empty('folder_reader'))

    def _build_metadata_generator(self):
        """Builds and return the metadata generator"""

        # Read and create the data metadata
        if self._input_type == 'folder_path':
            metadata_generator = self._build_folder_reader()
        elif self._input_type == 'mongo':
            raise NotImplementedError
        else:
            raise ValueError(f"input type of {self._input_type} not recognized")

        return metadata_generator

    def _create_metadata(self) -> Result[Any, Exception]:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        train_metadata = val_metadata = test_metadata = None

        if mpi.mpi_communicator.is_main_rank():
            # get the metadata
            metadata = self.metadata_generator.build_and_get_metadata()
            metadata = self._modify_metadata(metadata)
            train_metadata, val_metadata, test_metadata = \
                self._generate_train_val_test_metadata(metadata)
            train_metadata, val_metadata, test_metadata = \
                self._add_additional_metadata(
                    train_metadata,
                    val_metadata,
                    test_metadata
                )
        # ask the metadata generator to sync among the nodes
        for md in [train_metadata, val_metadata, test_metadata]:
            if (res := self.metadata_generator.sync(md)).is_err():
                self._log.error(f"error while asking the metadata generator to sync with error of {res.get_err()}")
                return res

        return Ok((train_metadata, val_metadata, test_metadata))

    def _modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Modifies the metadata that this instance holds."""

        metadata_manipulator = MetadataManipulator(metadata=metadata)

        # regroup the metadata
        metadata = metadata_manipulator.modify()

        return metadata

    def _generate_train_val_test_metadata(self, metadata: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """The method creates train, validation, and test metadata."""

        metadata_sync, metadata_unsync = MetadataManipulator.separate_sync_choices_metadata(metadata)

        # Get count of each set
        total_data_count = metadata_sync.index.get_level_values(0).unique().size
        test_count = int(total_data_count * self._test_ratio)
        val_count = int((total_data_count - test_count) * self._val_ratio)
        train_count = total_data_count - test_count - val_count

        # store the previous state of the random generator
        # Find the indices of each set
        rng_state = np.random.get_state()
        seed = self._seed
        if self._seed is not None:

            # add rank
            if self._shuffle_add_node_rank is True:
                seed += mpi.mpi_communicator.get_rank()

            np.random.seed(seed)

        indices = np.arange(total_data_count) \
            if self._shuffle is False \
            else np.random.permutation(total_data_count)
        test_indices = metadata_sync.index.get_level_values(0).unique()[indices[:test_count]]
        val_indices = metadata_sync.index.get_level_values(0).unique()[indices[test_count:(test_count+val_count)]]
        train_indices = metadata_sync.index.get_level_values(0).unique()[indices[(test_count+val_count):]]

        # Create the train, validation, and test metadata
        test_metadata = MetadataManipulator.join_metadata_idx_sort([metadata_unsync, metadata_sync.loc[test_indices]])
        val_metadata = MetadataManipulator.join_metadata_idx_sort([metadata_unsync, metadata_sync.loc[val_indices]])
        train_metadata = MetadataManipulator.join_metadata_idx_sort([metadata_unsync, metadata_sync.loc[train_indices]])

        # Update the column names of the data frames
        assert(_metadata_columns_internal['original_index'] == 'original_index')  # this is because df.assign can only get kwargs
        [train_metadata, val_metadata, test_metadata] = \
            [
                df
                .assign(original_index=df.index.get_level_values(0))
                .rename(
                    index={
                        key: value
                        for value, key
                        in enumerate(df.index.get_level_values(0).unique(), start=0)
                    },
                    level=0,
                )
                for df
                in [train_metadata, val_metadata, test_metadata]
            ]

        # restore the random generator state
        if self._seed is not None:
            np.random.set_state(rng_state)

        return train_metadata, val_metadata, test_metadata

    def _add_additional_metadata(
            self,
            train_metadata,
            val_metadata,
            test_metadata
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """Adds the additional metadata to the metadata within"""

        def reindex(df: pd.DataFrame, index_begin: int) -> pd.DataFrame:
            """Reindexes the 0 level indices of the given pandas dataframe so that the indexing starts from
            the given value"""

            return df.rename(
                index={
                    key: value
                    for value, key
                    in enumerate(df.index.get_level_values(0).unique(), start=index_begin)
                },
                level=0,
            )

        def add_original_index(df: pd.DataFrame) -> pd.DataFrame:
            """adds `original_index` column to the dataframe."""

            df[_metadata_columns_internal['original_index']] = -1

            return df

        def process_df(new_df: pd.DataFrame, old_df: pd.DataFrame) -> pd.DataFrame:
            """Processes the new df and concats it with the old one and returns the result."""

            new_df = self._modify_metadata(new_df)
            new_df = add_original_index(new_df)
            new_df = reindex(new_df, len(old_df.index.get_level_values(0).unique()))
            df = pd.concat([old_df, new_df])

            return df

        # get the new train metadata, process it and add it to the train metadata
        train_metadata = \
            process_df(
                self.metadata_generator.build_and_get_add_train_metadata(),
                train_metadata,
            )

        # get the new val metadata, process it and add it to the val metadata
        val_metadata = \
            process_df(
                self.metadata_generator.build_and_get_add_val_metadata(),
                val_metadata,
            )

        # get the new test metadata, process it and add it to the test metadata
        test_metadata = \
            process_df(
                self.metadata_generator.build_and_get_add_test_metadata(),
                test_metadata,
            )

        return train_metadata, val_metadata, test_metadata

    def _shuffle_all_metadata(self, rand_seed_add: int = 0) -> None:
        """
        Shuffles each of the metadata separately.

        Parameters
        ----------
        rand_seed_add : int, optional
            number to add with the random seed generator to have different random number generation each time

        """

        self.train_metadata = self._shuffle_metadata(self.train_metadata, rand_seed_add)
        self.val_metadata = self._shuffle_metadata(self.val_metadata, rand_seed_add)
        self.test_metadata = self._shuffle_metadata(self.test_metadata, rand_seed_add)

    def _shuffle_metadata(self, metadata: pd.DataFrame, rand_seed_add: int = 0) -> pd.DataFrame:
        """
        Shuffles the given metadata if necessary

        Parameters
        ----------
        metadata : pd.DataFrame
            metadata to shuffle
        rand_seed_add : int, optional
            number to add with the random seed generator to have different random number generation each time

        """

        if self._shuffle is False:
            return metadata

        # store the previous state of the random generator and set the seed
        rng_state = np.random.get_state()

        seed = self._seed
        # add the given number
        seed += rand_seed_add
        # add rank
        if self._shuffle_add_node_rank is True:
            seed += mpi.mpi_communicator.get_rank()

        np.random.seed(seed)

        # shuffle the metadata
        metadata_count = metadata.index.get_level_values(0).unique().size
        indices = np.random.permutation(metadata_count)
        if len(indices) > 0:
            metadata = metadata.loc[indices]
            metadata = MetadataManipulator.reset_level_0_indices(metadata)

        # restore the random generator state
        np.random.set_state(rng_state)

        return metadata

    def sync_metadata(self, rand_seed_add: int = 0) -> None:
        """
        Does the necessary things to sync the metadata. Should be called at each epoch.

        Parameters
        ----------
        rand_seed_add : int, optional
            number to add with the random seed generator to have different random number generation each time

        """

        # first, sync all metadata across nodes in distributed mode
        if (res := self._sync_all_metadata_distributed(rand_seed_add)).is_err():
            self._log.error(f"error while syncing the metadata with error of {res.get_err()}")
            raise RuntimeError("error while syncing metadata")
        self.train_metadata, self.val_metadata, self.test_metadata = res.get()

        # shuffle
        self._shuffle_all_metadata(rand_seed_add)

        # send the new metadata to all the workers
        self._distribute_metadata_to_workers()

    def set_batch_size(self, batch_size: int) -> None:
        """Sets the batch size

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        self.batch_size = batch_size

        # Set batch size of all the workers
        for worker in self._get_dataloader_workers():
            worker.set_batch_size(self.batch_size)

        self._log.info(f"batch size set to {colored.fg('cyan')}{batch_size}{colored.attr('reset')}")

    @abstractmethod
    def _distribute_metadata_to_workers(self) -> None:
        """distribute the metadata to workers."""

        pass


class MetadataManipulator(base.BaseWorker):
    """Class to modify and manipulate metadata."""

    def __init__(self, metadata: pd.DataFrame, config: ConfigParser = None):

        super().__init__(config)

        # metadata storage
        self.metadata = metadata

    def modify(self, check_for_sanity: bool = True) -> pd.DataFrame:
        """Runs the whole modification pipeline and returns the result."""

        metadata = self.metadata

        if len(self.metadata) > 0:
            # we create a criterion helper column based on the data that we have to assist the groupby call of pandas
            # to groupby int instead of anything else
            criterion_field_name = _metadata_columns_internal['__criterion']
            criterion_set = self.metadata[criterion_field_name].unique()
            mapping = {criterion: idx_groupby for idx_groupby, criterion in enumerate(criterion_set)}
            self.metadata['__criterion_help'] = \
                self.metadata.apply(lambda row: mapping[row[criterion_field_name]], axis=1)

            # regroup based on the file name
            metadata = self.regroup_metadata(self.metadata, criterion='__criterion_help')
            self.metadata = metadata = metadata.drop(['__criterion_help'], axis=1)

            if check_for_sanity:
                self._check_for_sanity(metadata)

        return metadata

    @staticmethod
    def regroup_metadata(metadata: pd.DataFrame, criterion: str = None, reset_index: bool = True) -> pd.DataFrame:
        """Groups the metadata.

        Each group of data (e.g. containing data and label) should have.
        Each group must have its own unique index, where indices are range.
        Each group will be recovered by metadata.loc[index]

        Parameters
        ----------
        metadata : pd.DataFrame
            metadata to process
        criterion : str or List[str]
            The name of the columns based on which the metadata should be categorized
        reset_index : bool
            whether to reset the level 0 indices

        Returns
        -------
        pd.DataFrame
            the result

        """

        if criterion is None or metadata.empty:
            return metadata

        # check if criterion is a column concept
        if isinstance(criterion, str) and criterion not in metadata.columns:
            raise ValueError(f"criterion '{criterion}' does not exists as a metadata column.")
        elif isinstance(criterion, list) and any([(c not in metadata.columns) for c in criterion]):
            raise ValueError(f"part of criterion '{criterion}' does not exists as a metadata column.")

        new_indices = []
        new_indices_count = {}
        for k in metadata[criterion]:
            n = new_indices_count.get(k, 0)
            new_indices_count[k] = n + 1
            new_indices.append((k, n))
        metadata = metadata.copy(True)
        metadata.index = pd.MultiIndex.from_tuples(new_indices)
        # sort the indices
        metadata = metadata.loc[metadata.index.sort_values()]

        if reset_index:
            # Rename the indices to be range
            # Also rename the index level 0 name to be 'index' (instead of `criterion`)
            metadata = MetadataManipulator.reset_level_0_indices(metadata)
            metadata.index.names = [None, *metadata.index.names[1:]]

        return metadata

    @staticmethod
    def check_for_sanity(metadata: pd.DataFrame, logger) -> None:
        """Checks the given metadata and makes some info if necessary."""

        # get the bundle counts and report
        idx_0_to_1_count: Dict[int, int] = {}
        for (l_0, _) in metadata.index:
            idx_0_to_1_count[l_0] = (idx_0_to_1_count.get(l_0) or 0) + 1

        from collections import Counter
        count = Counter(idx_0_to_1_count.values())
        if (length := len(count.values())) == 0:
            logger.error(f"ended up in a weird situation: data bundles seem to not exist!")
        elif length == 1:
            logger.info(f"data bundles contain {list(count.keys())[0]} elements each")
        else:
            warn = \
                "data bundles contain different amount of elements. I found:" \
                "\n\t (bundle size) (count)" \
                "\n\t\t (samples' path)"
            # get the pd max_colwidth and set it to something big
            pd_max_col_width = pd.get_option('display.max_colwidth')
            pd.set_option('display.max_colwidth', 1000)
            for c, v in count.items():
                warn += f"\n\t- {c}: {v}"
                # get the elements that have c counts
                c_keys = [k for k, v in idx_0_to_1_count.items() if v == c][:10]
                # sample them
                sample_df_str = str(metadata.loc[c_keys][metadata_columns['path']])
                from tabaluga.framework.util.util import REGEX_INDENT_NEW_LINE_ONLY
                sample_df_str = REGEX_INDENT_NEW_LINE_ONLY.sub('\n\t\t ', sample_df_str)

                warn += f"\n\t\t {sample_df_str}"

            warn += "\n"

            # revert pandas setting
            pd.set_option('display.max_colwidth', pd_max_col_width)

            logger.warning(warn)

    def _check_for_sanity(self, metadata: pd.DataFrame) -> None:
        """Checks the given metadata and makes some info if necessary."""

        return self.check_for_sanity(metadata, self._log)

    @staticmethod
    def reset_level_0_indices(metadata: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
        """
        Resets the level 0 indices to start from 0 and returns the result.

        Parameters
        ----------
        metadata : pd.DataFrame
            the data frame to reset the indices
        offset : int
            the offset for setting the index

        Returns
        -------
        pd.DataFrame
            the result

        """

        metadata = metadata.rename(
            index={key: value for value, key in enumerate(metadata.index.get_level_values(0).unique(), start=offset)},
            level=0,
        )

        return metadata

    @staticmethod
    def bundle_exist_name(metadata: pd.DataFrame, exist_names: List[str], criterion_column: str):
        """
        Takes the two level indexed metadata, for each of the levels checks if the criterion column contains all names.
        If not, removes that bundle (level 0 index) and returns the result.

        Parameters
        ----------
        metadata : pd.DataFrame
            the two-level indexed metadata
        exist_names : List[str]
            the names that must exist in each bundle according to criterion_column values
        criterion_column : str
            the column name to use to search for exist_names

        Returns
        -------
        pd.DataFrame
            the result

        """

        names_count: int = len(exist_names)
        locs_drop = []

        for i0 in metadata.index.get_level_values(0).unique():
            loc = metadata.loc[i0]

            # if the bundle has a smaller number of elements from the "exist" names, drop it
            if len(loc) < names_count:
                locs_drop.append(i0)
                continue

            # if it does not have all names, then it should be removed
            loc_criterion = loc[criterion_column].values
            for name in exist_names:
                if name not in loc_criterion:
                    locs_drop.append(i0)
                    break

        metadata = metadata.drop(locs_drop)

        return metadata

    @staticmethod
    def separate_sync_choices_metadata(metadata: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        metadata_sync = metadata[metadata[metadata_columns['metadata_sync_choice']] == True]
        metadata_unsync = metadata[metadata[metadata_columns['metadata_sync_choice']] == False]

        return metadata_sync, metadata_unsync

    @staticmethod
    def join_metadata_idx_sort(metadatas: List[pd.DataFrame]) -> pd.DataFrame:

        return pd.concat(metadatas).sort_index(level=0)

    @staticmethod
    def join_to_metadata_level_0_idx_sort(metadata_main: pd.DataFrame, metadata_others: List[pd.DataFrame]) -> pd.DataFrame:

        level_0_indices = metadata_main.index.get_level_values(0).unique()

        metadata_others = [m[m.index.get_level_values(0) in level_0_indices] for m in metadata_others]

        res = MetadataManipulator.join_metadata_idx_sort([metadata_main, *metadata_others])

        return res


class FolderReader(base.BaseWorker):
    """This class is a helper class that reads metadata from a path (folder)."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # Folders containing the data
        self._folders: List[str] = []

        # how to create criterion
        _criterion: List[str] = \
            self._config.get_value_option('criterion_generation.criterion')\
            .expect('criterion config does not exist')
        self._criterion_hash: bool = self._config.get_or_else('criterion_generation.hash', True)
        self._criterion_function = self._check_process_criterion(_criterion)

        # list of extensions to ignore and accept
        self._extension_ignore: List[re.Pattern] = [
            re.compile(item) for item in self._config.get_or_else('extension_ignore', [])
        ]
        self._extension_accept: List[re.Pattern] = [
            re.compile(item) for item in self._config.get_or_else('extension_accept', [])
        ]

        # Folders containing the data
        self._folders: List[str] = self._config.get_value_option('folders').expect('folder config does not exist')
        self._add_train_folders: List[str] = self._config.get_or_else('additional_train_folders', [])
        self._add_val_folders: List[str] = self._config.get_or_else('additional_val_folders', [])
        self._add_test_folders: List[str] = self._config.get_or_else('additional_test_folders', [])
        self._add_train_folders = [] if self._add_train_folders == [None] else self._add_train_folders
        self._add_val_folders = [] if self._add_val_folders == [None] else self._add_val_folders
        self._add_test_folders = [] if self._add_test_folders == [None] else self._add_test_folders

        # File names in nested lists
        self._deep_folders: bool = self._config.get_or_else('deep', False)

        self._file_type = self._config.get_value_option('type').expect('type for file type not provided')

        # whether we want to find the best multithreading count
        self._find_best_multithreading_count_enabled = self._config.get_or_else('find_best_multithreading_count', False)

    def get_type(self) -> str:

        return self._file_type

    def _check_process_criterion(self, criterion: List[str]) -> Callable[[pd.Series], str]:
        """Checks if the given criterion is ok"""

        # the criterion values should be from the columns' names
        for criteria in criterion:
            if criteria not in metadata_columns.keys():
                self._log.error(
                    f"criteria '{criteria}' not accepted. Only the following criterion are accepted:"
                    + '\n\t - '
                    + '\n\t - '.join(metadata_columns.keys())
                    + '\n'
                )
                raise ValueError("unknown criteria")

        if self._criterion_hash:
            import hashlib
            criterion_function: Callable[[pd.Series], str] = \
                lambda row: \
                hashlib\
                .md5(
                    '+++'.join([row[metadata_columns[item]] for item in criterion])
                    .encode()
                )\
                .hexdigest()
        else:
            criterion_function: Callable[[pd.Series], str] = \
                lambda row: '+++'.join([row[metadata_columns[item]] for item in criterion])

        return criterion_function

    def _check_populate_folder_metadata(self, folders: List[str]):
        """Checks given folders for sanity and existence."""

        # Check if folder list is given
        if folders is None:
            raise ValueError("No folders given to read files from!")
        for folder in folders:
            if not isinstance(folder, str):
                raise ValueError(f"given folder {folder} is not string")
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

        self._check_populate_folder_metadata(self._folders)
        metadata = self._build_and_get_metadata_folders(self._folders)

        return metadata

    def build_and_get_add_train_metadata(self) -> pd.DataFrame:
        """
        This method goes over the specified folders for the additional train data folder, read files and creates
        and returns a pandas data frame from them.

        Returns
        -------
        pd.DataFrame
            pandas dataframe of the information

        """

        self._check_populate_folder_metadata(self._add_train_folders)
        metadata = self._build_and_get_metadata_folders(self._add_train_folders)

        return metadata

    def build_and_get_add_val_metadata(self) -> pd.DataFrame:
        """
        This method goes over the specified folders for the additional validation data folder, read files and creates
        and returns a pandas data frame from them.

        Returns
        -------
        pd.DataFrame
            pandas dataframe of the information

        """

        self._check_populate_folder_metadata(self._add_val_folders)
        metadata = self._build_and_get_metadata_folders(self._add_val_folders)

        return metadata

    def build_and_get_add_test_metadata(self) -> pd.DataFrame:
        """
        This method goes over the specified folders for the additional test data folder, read files and creates
        and returns a pandas data frame from them.

        Returns
        -------
        pd.DataFrame
            pandas dataframe of the information

        """

        self._check_populate_folder_metadata(self._add_test_folders)
        metadata = self._build_and_get_metadata_folders(self._add_test_folders)

        return metadata

    def _build_and_get_metadata_folders(self, folders: List[str] = None) -> pd.DataFrame:
        """
        This method goes over the specified folders, read files and creates and returns a pandas data frame from them.

        Parameters
        ----------
        folders : List[str], optional
            list of folders to look into. if not given, the default folder is assumed

        Returns
        -------
        pd.DataFrame
            pandas dataframe of the information

        """

        folders = folders if folders is not None else self._folders

        # File names in nested lists
        if self._deep_folders is False:
            # only one level deep
            file_nested_paths = [list(Path(folder).iterdir()) for folder in folders]
        else:
            # grab everything!
            file_nested_paths = [list(Path(folder).glob('**/*')) for folder in folders]

        # Flatten the file names and make absolute paths
        file_paths = [file_path
                      for file_paths in file_nested_paths
                      for file_path in file_paths]
        # Check each file based on the criteria
        file_paths = [file_path for file_path in file_paths if self._check_file(file_path)]
        # Retrieve the folder path and file names
        folder_paths = [file_name.parent for file_name in file_paths]
        folder_parent_paths = [folder_path.parent for folder_path in folder_paths]
        folder_names = [folder_path.name for folder_path in folder_paths]
        file_names = [file_path.stem for file_path in file_paths]
        file_extensions = [file_path.suffix.lower() for file_path in file_paths]

        # Create data frame of all the files in the folder
        metadata = pd.DataFrame({
            metadata_columns['folder_path']: [str(item) for item in folder_paths],
            metadata_columns['folder_parent_path']: [str(item) for item in folder_parent_paths],
            metadata_columns['folder_name']: folder_names,
            metadata_columns['file_name']: file_names,
            metadata_columns['file_extension']: file_extensions,
            metadata_columns['path']: [str(item) for item in file_paths],
            metadata_columns['content_type']: ContentTypes.FILE.value,
            metadata_columns['metadata_sync_choice']: True,
            metadata_columns['syncable']: True,
        })

        # conform the data according to the file format
        metadata = self._conform_to_type(metadata)

        # add criterion column
        metadata = self._add_criterion_column(metadata)

        return metadata

    def _conform_to_type(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Conforms the provided metadata based on the file format and returns the result or panic."""

        if self._file_type == 'separate_files':
            _ = FolderReaderExecutorSeparateFiles(self._config.get_or_empty("separate_files")).process(metadata)
        elif self._file_type == 'coco':
            _ = FolderReaderExecutorCOCO(self._config.get_or_empty("coco")).process(metadata)
        else:
            raise Exception(f"unknown file type of {self._file_type}")

        if _.is_err():
            raise Exception(f"could not conform to type {self._file_type} with error of '{_.get_err()}'")

        return _.get()

    def _add_criterion_column(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the criterion column and returns it

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata

        Returns
        -------
        pd.DataFrame
            the new metadata with criterion column updated

        """

        if len(metadata) > 0:
            metadata[_metadata_columns_internal['__criterion']] = \
                metadata.apply(self._criterion_function, axis=1)

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

        # criteria for the file extension
        check &= self._filter_file_extension(
            pathlib.Path(file_name).suffix[1:].lower()
        )  # remove the leading . from the ext

        return check

    def _filter_file_extension(self, extension: str) -> bool:
        """
        Method to check whether the extension is acceptable or not.

        Parameters
        ----------
        extension : str
            the extension

        Returns
        -------
        bool
            whether the extension is acceptable

        """

        if any(pattern.match(extension) is not None for pattern in self._extension_ignore):
            return False

        if self._extension_accept and all(pattern.match(extension) is None for pattern in self._extension_accept):
            return False

        return True

    def get_best_multithreading_count(self, metadata: pd.DataFrame) -> Option[int]:
        """
        Finds the best number of threads to load the data.
        Will return `nothing` if not enabled

        Parameters
        ----------
        metadata : pd.DataFrame
            metadata to test with

        Returns
        -------
        Option[int]
            returns Some[int] if enabled and `nothing` if not
        """

        if self._find_best_multithreading_count_enabled:
            paths = [pathlib.Path(path) for path in metadata[metadata_columns['path']]]
            self._log.info(f"performing multithread count finder on training data with size of {len(paths)}")
            multithreading_count = DataLoaderMultiThreadFinder(
                paths=paths,
                config=self._config.get_or_empty('multithread_count_finder'),
            ).find_best_thread_count()

            return Some(multithreading_count)

        return nothing

    def sync(self, metadata: pd.DataFrame = None) -> Result[None, Exception]:
        """Syncs the data across nodes."""

        if not mpi.mpi_communicator.is_distributed():
            return Ok(None)

        # make a syncer to sync the files
        file_syncer: FileSyncer = FileSyncer(self._config.get_or_empty('file_syncer'))
        res = file_syncer.sync(metadata)

        return res


class FolderReaderExecutor(base.BaseWorker, ABC):
    """Abstract class to represent the executor on the folder reader metadata, such as a middleware."""

    @abstractmethod
    def process(self, metadata: pd.DataFrame) -> Result[pd.DataFrame, Exception]:
        """
        Processes the metadata and returns the result.

        Parameters
        ----------
        metadata : pd.DataFrame
            input metadata

        Returns
        -------
        Result[pd.DataFrame, Exception]
            resulting metadata, result wrapped

        """

        raise NotImplementedError


class FolderReaderExecutorSeparateFiles(FolderReaderExecutor):
    """Folder reader metadata for 'separate files' format."""

    def process(self, metadata: pd.DataFrame) -> Result[pd.DataFrame, Exception]:

        # do nothing and return the result
        return Ok(metadata)


class FolderReaderExecutorCOCO(FolderReaderExecutor):
    """Folder reader metadata for 'coco' format."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

    def _filter_only_jsons(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Filters out only the labels.json files in the metadata and returns the result"""

        return metadata[
            (metadata[metadata_columns['file_extension']] == '.json') &
            (metadata[metadata_columns['file_name']] == 'labels')
            ]

    def _read_json(self, path: pathlib.Path):
        """Loads the json file given the path"""

        with open(path) as file:
            import json
            the_json = json.load(file)

        return the_json

    def _construct_syncable_metadata(self, coco_json: Dict[str, Dict]) -> pd.DataFrame:
        """
        Constructs the metadata for syncing (among nodes) based on the provided coco json.

        Parameters
        ----------
        coco_json : Dict[str, Dict]
            coco labels.json in dictionary format

        Returns
        -------
        pd.DataFrame
            resulting dataframe

        """

        coco_images_info = coco_json['images']

        # Check each file based on the criteria
        file_paths = [pathlib.Path(coco_image_info['path']) for coco_image_info in coco_images_info]
        # Retrieve the folder path and file names
        folder_paths = [file_name.parent for file_name in file_paths]
        folder_parent_paths = [folder_path.parent for folder_path in folder_paths]
        folder_names = [folder_path.name for folder_path in folder_paths]
        file_names = [file_path.stem for file_path in file_paths]
        file_extensions = [file_path.suffix.lower() for file_path in file_paths]
        dataset_ids = [int(coco_image_info['dataset_id']) for coco_image_info in coco_images_info]
        ids = [int(coco_image_info['id']) for coco_image_info in coco_images_info]

        metadata = pd.DataFrame({
            metadata_columns['folder_path']: [str(item) for item in folder_paths],
            metadata_columns['folder_parent_path']: [str(item) for item in folder_parent_paths],
            metadata_columns['folder_name']: folder_names,
            metadata_columns['file_name']: file_names,
            metadata_columns['file_extension']: file_extensions,
            metadata_columns['path']: [str(item) for item in file_paths],
            metadata_columns_COCO['coco_id']: ids,
            metadata_columns_COCO['coco_dataset_id']: dataset_ids,
            metadata_columns['content_type']: ContentTypes.FILE.value,
            metadata_columns['metadata_sync_choice']: True,
            metadata_columns['syncable']: False,
        })

        return metadata

    def _construct_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs metadata from the provided metadata that must contain only labels.json info.

        Parameters
        ----------
        metadata : pd.DataFrame
            metadata for the labels.json of coco

        Returns
        -------
        pd.DataFrame
            resulting dataframe consisting of the input metadata and the constructed image metadata

        """

        metadata = metadata.copy()
        metadata[metadata_columns_COCO['coco_dataset_id']] = -1

        coco_metadatas = []
        for idx, p in metadata[metadata_columns['path']].items():
            coco_json = self._read_json(pathlib.Path(p))
            m = self._construct_syncable_metadata(coco_json)
            coco_metadatas.append(m)
            coco_dataset_id = \
                m[~m[metadata_columns_COCO['coco_dataset_id']].isna()][metadata_columns_COCO['coco_dataset_id']].iloc[0]
            metadata.loc[idx, metadata_columns_COCO['coco_dataset_id']] = coco_dataset_id

        metadata[metadata_columns['metadata_sync_choice']] = False

        metadata = pd.concat([metadata, *coco_metadatas])

        metadata = metadata.reset_index(drop=True)

        return metadata

    def process(self, metadata: pd.DataFrame) -> Result[pd.DataFrame, Exception]:
        """Processes the provided metadata, extract image info from the coco json metadata provided and returns the
        result"""

        metadata = self._filter_only_jsons(metadata)

        coco_metadata = self._construct_metadata(metadata)

        return Ok(coco_metadata)


class FileSyncer(OnAndEnabled):

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # make a new communicator for broadcasting
        self.mpi_comm_main_local_name = 'file_syncer_main_local'
        if (
                mpi_comm_main_local_res := mpi.mpi_communicator.split(
                    # communicator=mpi_comm_res.get(),
                    color=0 if mpi.mpi_communicator.is_main_local_rank() is True else 1
                )
        ).is_err():
            self._log.error(f"could not get the mpi split with error of {mpi_comm_main_local_res.get_err()}")
            raise RuntimeError("could not get the mpi split")
        self.mpi_comm_main_local = mpi_comm_main_local_res.get()
        mpi.mpi_communicator.register_communicator(self.mpi_comm_main_local, name=self.mpi_comm_main_local_name)

        # cache some distributed-mode-related data
        self.rank, self.dist_size = mpi.mpi_communicator.get_rank_size(self.mpi_comm_main_local)
        self.is_main_rank: bool = self.rank == 0
        self.distributor = True if self.is_main_rank is True else False
        self.receiver = True if self.is_main_rank is False else False

        # get the batch size for syncing
        self.batch_size = self._config.get_or_else('batch_size', 16)

        # bookkeeping
        self.thread_count: int = self._config.get_or_else('multithreading_count', 10)
        # whether we should force the data syncing initially
        self.force_sync = self._config.get_or_else('force_sync', False)

    def is_distributor(self) -> bool:
        """Returns a boolean regarding whether we are the distributor."""

        return self.distributor

    def is_receiver(self) -> bool:
        """Returns a boolean regarding whether we are a receiver."""

        return self.receiver

    def _sync_metadata(self, metadata: Optional[pd.DataFrame] = None) -> Result[pd.DataFrame, Exception]:
        """
        Syncs the metadata and returns it.

        Parameters
        ----------
        metadata : Optional[pd.DataFrame]
            must be provided if we are the main rank

        Returns
        -------
        Result[pd.DataFrame, Exception]
        """

        # make sure we have all the metadata if we are the main rank
        if self.distributor and metadata is None:
            return Err(
                RuntimeError("metadata are not set in the main rank. please set them before calling this function.")
            )

        # let everyone get the metadata
        res: Result[pd.DataFrame, Exception] = mpi.mpi_communicator.collective_bcast(
                data=metadata,
                root_rank=0,
                name=self.mpi_comm_main_local_name,
            )

        return res

    def _check_local_data(self, metadata: pd.DataFrame, thread_pool: ThreadPoolExecutor) -> pd.DataFrame:
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

    def _load_local_data_raw(self, metadata: pd.DataFrame, thread_pool: ThreadPoolExecutor) \
            -> Result[pd.DataFrame, Exception]:
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
        Result[pd.DataFrame, Exception]
            a new metadata with the data loaded in it

        """

        def load_single_data_raw(path: str) -> Result[bytes, Exception]:
            """Loads a single raw data from the given path in binary form and returns it."""

            try:
                with open(path, 'rb') as file:
                    return Ok(file.read())
            except Exception as e:
                return Err(RuntimeError(f"cannot open {path} with error of {e}"))

        # Load the data
        data_res = list(
            thread_pool.map(load_single_data_raw, metadata[metadata_columns['path']])
        )
        for res in data_res:
            if res.is_err():
                self._log.error(f"{res.get_err()}")
                return Err(RuntimeError("could not read a file"))

        data = [item.get() for item in data_res]

        # assign them to a new column
        assert(_metadata_columns_internal['data_raw'] == 'data_raw')  # this is because df.assign can only get kwargs
        res = Result.from_func(metadata.assign, data_raw=data)

        return res

    def _save_local_data_raw(self, metadata: pd.DataFrame, thread_pool: ThreadPoolExecutor) -> Result[None, Exception]:
        """
        saves the data

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for saving the data
        thread_pool : ThreadPoolExecutor
            the thread pool to use for saving the data

        Returns
        -------
        Result[None, Exception]
        """

        def save_single_data_raw(path: str, data: bytes) -> Result[None, Exception]:
            """saves a single raw data to the given path in binary form."""

            try:
                # create the folder if it does not exist
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

                # now save the result
                with open(path, 'wb') as file:
                    file.write(data)

                return Ok(None)

            except Exception as e:
                return Err(RuntimeError(f"could not save file at {path} with error of {e}"))

        # save the data
        save_res = list(
            thread_pool.map(
                lambda x: save_single_data_raw(x[0], x[1]),
                zip(metadata[metadata_columns['path']], metadata[_metadata_columns_internal['data_raw']])
            )
        )
        for res in save_res:
            if res.is_err():
                self._log.error(f"{res.get_err()}")
                return Err(RuntimeError("could not save a file"))

        return Ok(None)

    def _sync_missing_metadata(
            self,
            rank: int,
            metadata: pd.DataFrame,
            thread_pool: ThreadPoolExecutor
    ) -> Result[pd.DataFrame, Exception]:
        """
        Finds and syncs the metadata of the missing files

        Parameters
        ----------
        rank : int
            destination rank
        metadata : pd.DataFrame
            the metadata to use for syncing
        thread_pool : ThreadPoolExecutor
            the thread pool to use for the operations

        Returns
        -------
        Result[pd.DataFrame, Exception]
        """

        if self.receiver:
            # find out the missing data and send it to the main rank
            metadata_missing_files = self._check_local_data(metadata, thread_pool)
            if (
                    res := mpi.mpi_communicator.p2p_send(
                        data=metadata_missing_files,
                        destination=0,
                        name=self.mpi_comm_main_local_name,
                    )
            ).is_err():
                return res
            else:
                return Ok(metadata_missing_files)

        elif self.distributor:
            # get the data sent by this rank
            res = mpi.mpi_communicator.p2p_receive(
                source=rank,
                name=self.mpi_comm_main_local_name,
            )
            return res

        else:
            raise RuntimeError("we should not have ended up here!")

    def _sync_data(
            self,
            rank: int,
            metadata_missing_files: pd.DataFrame,
            thread_pool: ThreadPoolExecutor,
    ) -> Result[None, Exception]:
        """
        Syncs data according to the metadata of the missing files

        Parameters
        ----------
        rank : int
            destination rank
        metadata : pd.DataFrame
            the metadata to use for syncing
        thread_pool : ThreadPoolExecutor
            the thread pool to use for the operations

        Returns
        -------
        Result[None, Exception]
        """

        # go over the data in batch_size chunks
        for start_idx in range(0, len(metadata_missing_files), self.batch_size):

            # load if we are the distributor
            if self.distributor:
                # first, load the data into a new dataframe
                if (
                        metadata_updated_res := self._load_local_data_raw(
                            metadata_missing_files.iloc[start_idx:(start_idx + self.batch_size)],
                            thread_pool,
                        )
                ).is_err():
                    return metadata_updated_res

                # now send to the rank
                if (
                        res := mpi.mpi_communicator.p2p_send(
                            data=metadata_updated_res.get(),
                            destination=rank,
                            tag=start_idx,
                            name=self.mpi_comm_main_local_name,
                        )
                ).is_err():
                    return res

            else:
                # just receive the data and store them
                if (
                        metadata_updated_res := mpi.mpi_communicator.p2p_receive(
                            source=0,
                            tag=start_idx,
                            name=self.mpi_comm_main_local_name,
                        )
                ).is_err():
                    return metadata_updated_res

                if (
                        res := self._save_local_data_raw(
                            metadata_updated_res.get(),
                            thread_pool
                        )
                ).is_err():
                    return res

        return Ok(None)

    def _sync_local_data_with_rank(
            self,
            rank: int,
            metadata: pd.DataFrame,
            thread_pool: ThreadPoolExecutor
    ) -> Result[None, Exception]:
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

        Returns
        -------
        Result[None, Exception]
        """

        if (
            res := self._sync_missing_metadata(
                rank,
                metadata,
                thread_pool,
            )
        ).is_err():
            return res
        metadata_missing_files = res.get()

        # log
        if self.is_distributor() and len(metadata_missing_files) > 0:
            self._log.info(f"syncing {len(metadata_missing_files)} local data selectively with rank {rank}")

        # sync the data
        if (
            res := self._sync_data(
                rank,
                metadata_missing_files,
                thread_pool,
            )
        ).is_err():
            return res

        # log
        if self.is_distributor() and len(metadata_missing_files) > 0:
            self._log.info(f"done syncing {len(metadata_missing_files)} local data selectively with rank {rank}")

        return Ok(None)

    def _sync_local_data_selective(self, metadata: pd.DataFrame) -> Result[None, Exception]:
        """
        Syncs only the missing portion of the local data with all the processes and makes sure everyone has all.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for syncing

        Returns
        -------
        Result[pd.DataFrame, Exception]
        """

        if self.distributor:
            try:
                # make a thread pool to be used for interacting with other ranks
                thread_pool = \
                    ThreadPoolExecutor(
                        self.thread_count,
                        thread_name_prefix="tabaluga-syncer-local-data-selective-distributor"
                    )
            except Exception as e:
                return Err(e)

            result: List[Result[None, Exception]] = list(
                thread_pool.map(
                    lambda rank: self._sync_local_data_with_rank(rank, metadata, thread_pool),
                    range(1, self.dist_size)
                )
            )

        else:

            try:
                # make a thread pool to be used for the subthreads
                thread_pool = ThreadPoolExecutor(
                    self.thread_count,
                    thread_name_prefix="tabaluga-syncer-local-data-selective"
                )
            except Exception as e:
                return Err(e)

            # just receive the data!
            result: List[Result[None, Exception]] = \
                [self._sync_local_data_with_rank(self.rank, metadata, thread_pool)]

        # shutdown the threadpool as it is no longer needed
        thread_pool.shutdown(wait=True)

        for item in result:
            if item.is_err():
                return Err(RuntimeError(f"error while syncing data with error of {item.get_err()}"))

        return Ok(None)

    def _sync_local_data_broadcast(self, metadata: pd.DataFrame) -> Result[None, Exception]:
        """
        Broadcasts all the local data to all the processes and makes sure everyone has it.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata to use for syncing

        """

        # we should not continue if we are not the main local rank
        if mpi.mpi_communicator.is_main_local_rank() is False:
            return Ok(None)

        # get the rank and size
        rank, size = \
            mpi.mpi_communicator.get_rank_size(mpi.mpi_communicator.get_communicator(self.mpi_comm_main_local_name))

        # log
        if self.is_distributor():
            self._log.info(f"syncing {len(metadata)} local data via force broadcasting with {size-1} workers")

        # make a thread pool ot be used for loading the data
        thread_pool = ThreadPoolExecutor(self.thread_count, thread_name_prefix="tabaluga-syncer-local-data-broadcast")

        # go over the data in batch_size chunks
        for start_idx in range(0, len(metadata), self.batch_size):

            # load if we are the distributor
            if self.distributor:
                # first, load the data into a new dataframe
                if (
                        metadata_updated_res := self._load_local_data_raw(
                            metadata.iloc[start_idx:(start_idx+self.batch_size)],
                            thread_pool,
                        )
                ).is_err():
                    return metadata_updated_res
                metadata_updated = metadata_updated_res.get()
            else:
                metadata_updated = None

            # now broadcast
            if (
                    metadata_updated_res := mpi.mpi_communicator.collective_bcast(
                        data=metadata_updated,
                        root_rank=0,
                        name=self.mpi_comm_main_local_name,
                    )
            ).is_err():
                return metadata_updated_res

            metadata_updated = metadata_updated_res.get()

            # save only if we are not the main rank
            if self.receiver:
                if (
                        res := self._save_local_data_raw(metadata_updated, thread_pool)
                ).is_err():
                    return res

        # shutdown the threadpool as it is no longer needed
        thread_pool.shutdown(wait=True)

        # log
        if self.is_distributor():
            self._log.info(f"done syncing {len(metadata)} local data via force broadcasting with {size-1} workers")

    def _sync_local_data(self, metadata: pd.DataFrame) -> Result[None, Exception]:
        """
        Reads the metadata and syncs the local data within.

        Parameters
        ----------
        metadata : pd.DataFrame, optional
            metadata to use for syncing

        Returns
        -------
        Result[pd.DataFrame, Exception]
        """

        # get the portion of the metadata that is syncable
        syncable_selection = metadata[metadata_columns['syncable']] == True
        metadata = metadata[syncable_selection]

        if self.force_sync is True:
            # force sync if necessary
            res = self._sync_local_data_broadcast(metadata)
        else:
            # sync selectively
            res = self._sync_local_data_selective(metadata)

        return res

    def sync(self, metadata: pd.DataFrame = None) -> Result[None, Exception]:
        """
        Syncs the data of the given metadata.

        Parameters
        ----------
        metadata : pd.DataFrame
            the metadata of the data to be synced, must be provided if main rank, for others, it does not matter

        Returns
        -------
        Result[None, Exception]
            the result of the sync

        """

        if self._is_enabled_main_local_rank() and self.dist_size > 1:

            if self.distributor and metadata is None:
                return Err(
                    RuntimeError("metadata are not set in the main rank. please set them before calling this function.")
                )

            # sync the whole metadata
            res: Result[pd.DataFrame, Exception] = self._sync_metadata(metadata)
            if res.is_err():
                return res

            # sync the local data that is syncable
            self._sync_local_data(res.get())

        mpi.mpi_communicator.barrier()

        return Ok(None)


class MetadataSyncer(base.BaseWorker):

    def __init__(self, config: ConfigParser = None):
        super().__init__(config)

        # cache some distributed-mode-related data
        self.rank: int = mpi.mpi_communicator.get_rank()
        self.dist_size: int = mpi.mpi_communicator.get_size()
        self.is_main_rank: bool = mpi.mpi_communicator.is_main_rank()
        self.distributor = True if self.is_main_rank is True else False
        self.receiver = True if self.is_main_rank is False else False

        # whether we should split the train data among nodes
        self.all_nodes_all_train_data = self._config.get_or_else('all_nodes_all_train_data', False)

        # if we want the train metadata to have the same length across nodes
        self.same_train_metadata_length_nodes = self._config.get_or_else('same_train_metadata_length_nodes', True)

        # get a new mpi communicator
        self.mpi_comm_name = 'metadata_syncer'
        if (
                mpi_comm_res := mpi.mpi_communicator.get_or_create_communicator('metadata_syncer')
        ).is_err():
            self._log.error(f"could not get the mpi communicator with error of {mpi_comm_res.get_err()}")
            raise RuntimeError("could not get the mpi communicator")

    def is_distributor(self) -> bool:
        """Returns a boolean regarding whether we are the distributor."""

        return self.distributor

    def is_receiver(self) -> bool:
        """Returns a boolean regarding whether we are a receiver."""

        return self.receiver

    def _get_batch_sizes(self, batch_size: int) -> Result[List[int], Exception]:

        return mpi.mpi_communicator.collective_gather(batch_size, name=self.mpi_comm_name)

    def _get_counts_based_on_list(self, total_count: int, weights: List[int]) -> List[int]:

        total = sum(weights)
        common_fact = [int(int(c / total * total_count) / c) for c in weights]
        main_common_fact = min(common_fact)
        final_counts = [int(main_common_fact * c) for c in weights]

        return final_counts

    def _generate_train_val_test_metadata(
            self,
            train_metadata: pd.DataFrame,
            val_metadata: pd.DataFrame,
            test_metadata: pd.DataFrame,
            batch_size: int,
    ) -> Result[Any, Exception]:
        """
        Generates and returns train/val/test metadata for each of the ranks.

        Returns
        -------
        Result[(List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]), Exception]
            resulting dataframes
        """

        res = self._get_batch_sizes(batch_size)
        if res.is_err():
            return res
        batch_sizes = res.get()

        if self.is_distributor():

            # if we should not split the train data
            if self.all_nodes_all_train_data is True:

                # make the train data
                train_split_metadata = [train_metadata] * self.dist_size

            # if we should split the train data among the nodes
            else:
                train_metadata_sync, train_metadata_unsync = \
                    MetadataManipulator.separate_sync_choices_metadata(train_metadata)

                # split the training metadata
                # split according to the batch size so that each get the same number of iterations
                train_zero_level_indices = train_metadata_sync.index.get_level_values(0).unique()
                chunk_sizes = self._get_counts_based_on_list(train_zero_level_indices.size, batch_sizes)
                chunk_sizes_cumsum = list(np.cumsum([0, *chunk_sizes]))
                train_split_indices = \
                    [
                        train_zero_level_indices[chunk_sizes_cumsum[idx]:chunk_sizes_cumsum[idx+1]]
                        for idx
                        in range(0, len(chunk_sizes_cumsum) - 1)
                    ]
                train_split_metadata = \
                    [
                        train_metadata_sync.loc[chunk_indices]
                        for chunk_indices
                        in train_split_indices
                    ]

                # check that all train metadata have the same length
                if self.same_train_metadata_length_nodes:
                    md_lengths = [
                        md.index.get_level_values(0).unique().size / b
                        for md, b
                        in zip(train_split_metadata, batch_sizes)
                    ]
                    if not all([size == md_lengths[0] for size in md_lengths[1:]]):
                        self._log.error(
                            f"the train metadata across nodes have different number of iterations:"
                            f"\n\t - "
                            + "\n\t - ".join([f"{node}: {size}" for node, size in enumerate(md_lengths)])
                            + "\n"
                        )
                        return Err(RuntimeError("train metadata have different sizes"))

                train_split_metadata = \
                    [
                        MetadataManipulator.join_metadata_idx_sort([m, train_metadata_unsync])
                        for m
                        in train_split_metadata
                    ]

            # split the validation metadata
            # we only want the distributor to have the validation data and not the others
            val_metadata_sync, val_metadata_unsync = \
                MetadataManipulator.separate_sync_choices_metadata(train_metadata)
            val_split_metadata = [val_metadata_sync]
            val_split_metadata.extend([
                val_metadata_sync.iloc[:0]  # this will surely result in an empty dataframe
                for _
                in range(1, self.dist_size)
            ])
            val_split_metadata = [
                MetadataManipulator.join_metadata_idx_sort([val_metadata_unsync, m])
                for m
                in val_split_metadata
            ]

            # split the test metadata
            # we only want the distributor to have the test data and not the others
            test_metadata_sync, test_metadata_unsync = \
                MetadataManipulator.separate_sync_choices_metadata(val_metadata)
            test_split_metadata = [test_metadata]
            test_split_metadata.extend([
                test_metadata_sync.iloc[:0]  # this will surely result in an empty dataframe
                for _
                in range(1, self.dist_size)
            ])
            test_split_metadata = [
                MetadataManipulator.join_metadata_idx_sort([test_metadata_unsync, m])
                for m
                in test_split_metadata
            ]

        else:
            train_split_metadata = None
            val_split_metadata = None
            test_split_metadata = None

        return Ok((train_split_metadata, val_split_metadata, test_split_metadata))

    def sync_train_val_test_metadata(
            self,
            train_metadata: pd.DataFrame,
            val_metadata: pd.DataFrame,
            test_metadata: pd.DataFrame,
            batch_size: int,
    ) -> Result[Any, Exception]:
        """
        Splits the train/val/test metadata into chunks for each process, scatters them, and returns each chunk.
        It should be noted that after this operation, all nodes will have exactly the same amount of data.

        Returns
        -------
        Result[(pd.DataFrame, pd.DataFrame, pd.DataFrame), Exception]
            the data frames for the metadata of train, val, and test

        """

        if (
                res := self._generate_train_val_test_metadata(
                    train_metadata,
                    val_metadata,
                    test_metadata,
                    batch_size,
                )
        ).is_err():
            return res
        train_split_metadata, val_split_metadata, test_split_metadata = res.get()

        # scatter all the metadata
        if (
                train_metadata_res := mpi.mpi_communicator.collective_scatter(
                    data=train_split_metadata,
                    root_rank=0,
                    name=self.mpi_comm_name
                )
        ).is_err():
            return train_metadata_res

        if (
                val_metadata_res := mpi.mpi_communicator.collective_scatter(
                    data=val_split_metadata,
                    root_rank=0,
                    name=self.mpi_comm_name
                )
        ).is_err():
            return val_metadata_res

        if (
                test_metadata_res := mpi.mpi_communicator.collective_scatter(
                    data=test_split_metadata,
                    root_rank=0,
                    name=self.mpi_comm_name
                )
        ).is_err():
            return test_metadata_res

        return Ok((train_metadata_res.get(), val_metadata_res.get(), test_metadata_res.get()))


class DataLoaderMultiThreadFinder(base.BaseWorker):

    def __init__(self, paths: List[pathlib.Path], config: ConfigParser = None):

        super().__init__(config)

        self.paths = paths

        # get the batch size for syncing
        self.min_thread_count = self._config.get_or_else('min_thread_count', 1)
        self.max_thread_count = self._config.get_or_else('max_thread_count', 50)
        self.error_margin_allowed = self._config.get_or_else('error_margin_allowed', .1)
        self.average_count = self._config.get_or_else('average_count', 10)

    @staticmethod
    def _load_file(path: pathlib.Path) -> bytes:
        with open(path.as_posix(), mode='rb') as f:
            return f.read()

    def _loader(self, paths: List[pathlib.Path], multi_thread_count: int) -> List[concurrent.futures.Future]:

        pool = ThreadPoolExecutor(
            max_workers=multi_thread_count,
        )

        tasks = [pool.submit(self._load_file, item) for item in paths]

        [task.result() for task in tasks]

        return tasks

    def _runner(self, multithread_count: int) -> float:

        import time
        start_time = time.time()
        for _ in range(self.average_count):
            self._loader(self.paths, multithread_count)
        delta_time_sync = time.time() - start_time
        self._log.info(
            f"elapsed time for loading files with thread count of {multithread_count: 4} "
            f"is {delta_time_sync / self.average_count * 1e3: 10.2f} ms")

        return delta_time_sync

    def find_best_thread_count(self) -> int:

        result = [
            {
                'thread_count': thread_count,
                'time': self._runner(
                    multithread_count=thread_count,
                ),
            }
            for thread_count
            in range(self.min_thread_count, self.max_thread_count)
        ]

        # find the minimum time taken
        min_time = min([res['time'] for res in result])
        # now, allow up to some percentage of the time taken then choose the one with the lowest thread count
        sorted_result = sorted(
            [
                res
                for res
                in result
                if res['time'] / min_time <= (1 + self.error_margin_allowed)
            ],
            key=lambda x: x['thread_count']
        )
        best_thread_count = sorted_result[0]['thread_count']
        best_thread_count_time = sorted_result[0]['time']
        self._log.info(
            f"best thread count found is {best_thread_count} "
            f"with time of {best_thread_count_time / self.average_count * 1e3:5.3f} ms to load all the data"
        )

        # return the one with minimum thread count
        return best_thread_count


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
        self.metadata_original, self.metadata = self.set_metadata(metadata, False)

        # placeholder for the shared multithreading pool
        self._shared_multithreading_pool: Optional[ThreadPoolExecutor] = None

        # bookkeeping for iterator, batch size and number of iterations (batches in an epoch)
        self._iterator_count = 0
        self.batch_size: int = -1
        self.number_of_iterations: int = -1

        # for loading ahead batches
        self._loaded_data_mu = threading.Lock()
        self._loaded_data = DataMuncher()  # will be a mapping from str(batch) to the loaded data
        self._load_ahead_batch_count = self._config.get_or_else('load_ahead_batch_count', 3)
        self._load_ahead_enabled = self._load_ahead_batch_count > 0
        if self._load_ahead_enabled:
            self._log.info(f"loading batches ahead with size of {self._load_ahead_batch_count}")
            import multiprocessing
            self._r_data_load_ahead, self._w_data_load_ahead = multiprocessing.Pipe(duplex=False)
            self._load_ahead_thread = threading.Thread(
                name='tabaluga-dataloader-manager-batch-ahead-load',
                target=self._load_ahead_batch_thread,
                daemon=True,
            )
            self._load_ahead_thread.start()

        # bookkeeping to know if we wrap around the dataset while loading
        self._data_loading_wrap_around = True

        # Create workers
        self.create_workers()

    def set_shared_multithreading_pool(self, pool: ThreadPoolExecutor):
        """
        Sets the shared multithreading pool to be used for data loading and spreads it to the workers.

        Parameters
        ----------
        pool : ThreadPoolExecutor
            the thread pool

        """

        self._shared_multithreading_pool = pool
        for worker in self._get_dataloader_workers():
            worker.set_shared_multithreading_pool(pool)

    @abstractmethod
    def _get_dataloader_workers(self) -> List:
        """Returns the list of workers that the shared multithreading pool can be set to."""

        pass

    def _modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Make a selection of the metadata
        selection = [self._check_file(file_path) for file_path in metadata[metadata_columns['path']]]

        # Update the metadata
        metadata = metadata.iloc[selection]

        return metadata

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

        # bookkeeping for the returned number of iterations from workers
        number_of_iterations_workers = []

        # Set batch size of all the workers and get their number_of_iterations
        for worker in self._get_dataloader_workers():
            number_of_iterations_workers.append(worker.set_batch_size(batch_size))

        # Check if all the returned number of iterations are the same
        assert all([no == number_of_iterations_workers[0] for no in number_of_iterations_workers]) is True, \
            (
                f'Returned number_of_iterations from all DataLoader\'s must be the same.\n'
                f'I received number of iterations from the workers as follows:\n'
                f'\t- ' +
                "\n\t- ".join(
                    [f"{name}: {no}" for name, no in zip(self.workers.get_names(), number_of_iterations_workers)]
                ) +
                f"\n\n"
                f"Possibly, the amount of data you have is not the same across different parent folders.\n"
            )

        # Set the number of iterations
        self.number_of_iterations = number_of_iterations_workers[0]

        # # set the wrap around
        # if self._batch_size_effective > self.batch_size_report:
        #     self._data_loading_wrap_around = True
        # else:
        #     self._data_loading_wrap_around = False

    def get_batch_size(self) -> int:
        """
        Returns the batch size.

        This returns the batch size

        Returns
        -------
        int
            batch size
        """

        return self.batch_size

    def get_number_of_iterations(self) -> int:
        """
        Returns the number of iterations.

        This method returns number of iterations of report as it is the count that we want to iterate over globally.

        Returns
        -------
        int
            batch size
        """

        return self.number_of_iterations

    def __len__(self) -> int:
        """Returns the length of the instance.

        IMPORTANT: The length of all workers must be the same.

        """

        return self.get_number_of_iterations()

    def __iter__(self):
        """Returns an iterable, self."""

        # reset the iteration counter
        self._iterator_count = 0

        return self

    def __next__(self):
        """Returns the next set of data.

        Returns
        -------
        A collection of next set of data

        """

        # if the batch size is more that the amount of data left, go to beginning and return None
        if self._iterator_count > self.get_number_of_iterations():
            self._iterator_count = 0
            return StopIteration

        # Load the data
        data = self.__getitem__(self._iterator_count)

        return data

    def _reset_load_ahead_data(self):
        """Resets the loaded ahead data"""

        self._loaded_data = DataMuncher()

    def on_train_epoch_begin(self, info: DataMuncher = DataMuncher()):

        # reset the load ahead data as they are obsolete!
        self._reset_load_ahead_data()

        super().on_train_epoch_begin(info)

    def on_val_epoch_begin(self, info: DataMuncher = DataMuncher()):

        # reset the load ahead data as they are obsolete!
        self._reset_load_ahead_data()

        super().on_val_epoch_begin(info)

    def on_test_epoch_begin(self, info: DataMuncher = DataMuncher()):

        # reset the load ahead data as they are obsolete!
        self._reset_load_ahead_data()

        super().on_test_epoch_begin(info)

    def _load_batch_wrap(self, batch: int, *args, **kwargs):
        """Helper method that corrects the batch number according to the wraparound"""

        if batch >= self.get_number_of_iterations():
            if self._data_loading_wrap_around:
                batch = batch % self.get_number_of_iterations()
            else:
                self._log.error(
                    f"requested batch {batch} is beyond the number of batches of {self.get_number_of_iterations()}"
                )
                raise RuntimeError(
                    f"requested batch {batch} is beyond the number of batches of {self.get_number_of_iterations()}"
                )

        return self.load_batch(batch)

    @abstractmethod
    def load_batch(self, item: int):
        """
        loads a batch and returns the result

        Parameters
        ----------
        item : int
            batch number

        Returns
        -------
        result

        """

        raise NotImplementedError

    def _load_ahead_batches(self, batches: List[int]) -> None:
        """
        loads the batches given as input and stores them. This method is used in the look ahead loading.

        Parameters
        ----------
        batches : List[int]
            list of batches that has to be loaded

        """

        # look for the batches we have and load the ones we do not
        already_loaded_data_option = [
            (batch, self._loaded_data.get_value_option(str(batch)))
            for batch
            in batches
        ]

        # the batches that we will need
        old_loaded_data = {
            str(loaded_data[0]): loaded_data[1].get()
            for loaded_data
            in already_loaded_data_option
            if loaded_data[1].is_defined()
        }

        # load the batches that are not already loaded and save them
        new_data = {}
        for loaded_data in already_loaded_data_option:
            # skip if we already have the data
            if loaded_data[1].is_defined():
                continue
            # load the new batch
            new_data[str(loaded_data[0])] = self._load_batch_wrap(loaded_data[0])
            # update the loaded data
            new_loaded_data = DataMuncher({**old_loaded_data, **new_data})
            with self._loaded_data_mu:  # most probably, we will not have to wait here
                self._loaded_data = new_loaded_data

    def _load_ahead_batch_thread(self) -> None:
        """method for the load ahead thread to run"""

        while True:

            # get the batch size that was loaded
            r, _, _ = select.select([self._r_data_load_ahead], [], [])
            batch = r[0].recv()

            # find the batch sizes that should be loaded
            start_idx = batch + 1
            end_idx = batch + 1 + self._load_ahead_batch_count
            batches_to_be_loaded = list(range(start_idx, end_idx))

            # correct the batch indices if necessary
            if self._data_loading_wrap_around is False and end_idx >= self.get_number_of_iterations():
                batches_to_be_loaded = list(range(start_idx,  self.get_number_of_iterations()))

            # do the batch ahead loading
            self._load_ahead_batches(batches_to_be_loaded)

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

        return self.get(batch=item)

    def get(self, batch: int, *args, **kwargs):

        if self._load_ahead_enabled:
            # try to load from the already loaded data and if not exist, load manually

            with self._loaded_data_mu:
                data = self._loaded_data.get_value_option(str(batch))

            # if not already loaded, load it
            if data.is_empty():
                data = self._load_batch_wrap(batch, *args, **kwargs)
            else:
                data = data.get()

            # now, let the load ahead know
            self._w_data_load_ahead.send(batch)

        else:
            data = self._load_batch_wrap(batch, *args, **kwargs)

        return data

    def set_metadata(self, metadata: pd.DataFrame, distribute: bool = True) -> (pd.DataFrame, pd.DataFrame):
        """Sets the internal metadata and returns the same thing."""

        metadata_original = self.metadata_original = metadata
        metadata = self.metadata = self._modify_metadata(metadata)

        if distribute:
            self._distribute_metadata_to_workers()

        return metadata_original, metadata

    @abstractmethod
    def _distribute_metadata_to_workers(self) -> None:
        """distribute the metadata to workers."""

        pass


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
        self.metadata: pd.DataFrame = self.set_metadata(metadata)

        # Flag for if we should load the data with multithreading
        self.multithreading: bool = self._config.get_or_else('multithreading', True)
        self.use_own_multithreading: bool = self._config.get_or_else('use_own_multithreading', False)
        self.thread_count: int = self._config.get_or_else('multithreading_count', 5)
        self.use_shared_multithreading: bool = self._config.get_or_else('use_shared_multithreading', True)
        # placeholder for the multithreading pool
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        if self.use_own_multithreading is True:
            self.thread_pool = ThreadPoolExecutor(
                self.thread_count,
                thread_name_prefix="tabaluga-dataloader-thread-pool"
            )

        # for loading ahead batches
        self._loaded_data_mu = threading.Lock()
        self._loaded_data = DataMuncher()  # will be a mapping from str(batch) to the loaded data
        self._load_ahead_batch_count = self._config.get_or_else('load_ahead_batch_count', 3)
        self._load_ahead_enabled = self._load_ahead_batch_count > 0
        import multiprocessing
        self._r_data_load_ahead: Optional[multiprocessing.connection.Connection] = None
        self._w_data_load_ahead: Optional[multiprocessing.connection.Connection] = None
        self._load_ahead_thread: Optional[threading.Thread] = None
        if self._load_ahead_enabled:
            self._log.info(f"loading batches ahead with size of {self._load_ahead_batch_count}")
            self._r_data_load_ahead, self._w_data_load_ahead = multiprocessing.Pipe(duplex=False)
            self._load_ahead_thread = threading.Thread(
                name='tabaluga-dataloader-batch-ahead-load',
                target=self._load_ahead_batch_thread,
                daemon=True,
            )
            self._load_ahead_thread.start()

        # bookkeeping for the batch size and thus the number of iterations (batches) in each epoch
        self.batch_size: int = -1
        self.number_of_iterations: int = -1
        # in case the effective batch size is bigger than the report, we have to wrap around the metadata while loading
        self._data_loading_wrap_around = False

        # bookkeeping for iterator
        self._iterator_count = 0

    def set_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Sets the internal metadata and returns the same thing."""

        metadata = self.metadata = self._modify_metadata(metadata)

        return metadata

    def set_shared_multithreading_pool(self, pool: ThreadPoolExecutor):
        """
        Sets the shared multithreading pool to be used for data loading

        Parameters
        ----------
        pool : ThreadPoolExecutor
            the thread pool

        """
        if self.use_shared_multithreading:
            self.thread_pool = pool

    def _modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Make a selection of the metadata
        selection = [self._check_file(file_path) for file_path in metadata[metadata_columns['path']]]

        # Update the metadata
        metadata = metadata.iloc[selection]

        return metadata

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

    def _get_metadata_len(self) -> int:
        """Finds and returns the length of metadata."""

        return len(self.metadata)

    def set_batch_size(self, batch_size: int) -> int:
        """Sets the batch size and thus finds the total number of batches in one epoch.

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        self.batch_size = batch_size
        self.number_of_iterations = self._get_metadata_len() // batch_size

        # reset the loaded data
        self._loaded_data = DataMuncher()

        # set the wrap around
        # if self._batch_size_effective > self.batch_size_report:
        #     self._data_loading_wrap_around = True
        # else:
        #     self._data_loading_wrap_around = False

        return self.number_of_iterations

    def get_batch_size(self) -> int:
        """
        Returns the batch size.

        This returns the batch size

        Returns
        -------
        int
            batch size
        """

        return self.batch_size

    def get_number_of_iterations(self) -> int:
        """
        Returns the number of iterations.

        This method returns number of iterations of report as it is the count that we want to iterate over globally.

        Returns
        -------
        int
            batch size
        """

        return self.number_of_iterations

    def get_number_of_iterations_batch_size_1(self) -> int:
        """
        Returns the number of iterations if we were to use batch size 1.

        Returns
        -------
        int
            number of iterations
        """

        return self._get_metadata_len()

    def _load_data(self, metadata: pd.DataFrame):
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
            if self.thread_pool is None:
                self._log.error('multithreading is on but no multithreading pool is provided. maybe check the config')
                raise RuntimeError(
                    'multithreading is on but no multithreading pool is provided. maybe check the config'
                )
            # Load the data with threads
            data = list(
                    self.thread_pool.map(lambda row: self._load_single_data(row[1]), metadata.iterrows())
                )
        else:
            data = [
                self._load_single_data(row[1]) for row in metadata.iterrows()
            ]

        data = self._load_data_post(data)

        return data

    @abstractmethod
    def _load_single_data(self, row: pd.Series):
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
    def _load_data_post(self, data: List):
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

    def load_batch(self, item: int):
        """
        loads a batch and returns the result

        Parameters
        ----------
        item : int
            batch number

        Returns
        -------
        result

        """

        # Check if item count is sensible
        if item >= self.get_number_of_iterations() and self._data_loading_wrap_around is False:
            raise RuntimeError(f'Requested number of images to be loaded goes beyond the end of available data.')

        # Find the corresponding metadata
        begin_index = (item * self.batch_size) % self._get_metadata_len()
        end_index = begin_index + self.batch_size
        metadata = self.metadata.loc[begin_index:(end_index-1)]
        if end_index > self._get_metadata_len() - 1 and self._data_loading_wrap_around is True:
            remainder = self.batch_size - (self._get_metadata_len() - 1 - begin_index)
            metadata2 = self.metadata.loc[:(remainder-1)]
            metadata = pd.concat([metadata, metadata2])

        # Load the images
        data = self._load_data(metadata)

        return data

    def load_batch_size_1(self, item: int):
        """
        loads a batch with batch size of 1 and returns the result, which is a single element

        Parameters
        ----------
        item : int
            batch number

        Returns
        -------
        result

        """

        # Check if item count is sensible
        if item >= self._get_metadata_len():
            raise RuntimeError(f'Requested number of batch to be loaded goes beyond the end of available data.')

        # Find the corresponding metadata
        begin_index = item % self._get_metadata_len()
        end_index = begin_index + 1
        metadata = self.metadata.loc[begin_index:(end_index-1)]

        # Load the images
        data = self._load_data(metadata)

        return data

    def _load_ahead_batches(self, batches: List[int]) -> None:
        """
        loads the batches given as input and stores them. This method is used in the look ahead loading.

        Parameters
        ----------
        batches : List[int]
            list of batches that has to be loaded

        """

        # look for the batches we have and load the ones we do not
        already_loaded_data_option = [
            (batch, self._loaded_data.get_value_option(str(batch)))
            for batch
            in batches
        ]

        # the batches that we will need
        old_loaded_data = {
            str(loaded_data[0]): loaded_data[1].get()
            for loaded_data
            in already_loaded_data_option
            if loaded_data[1].is_defined()
        }

        # load the batches that are not already loaded and save them
        new_data = {}
        for loaded_data in already_loaded_data_option:
            # skip if we already have the data
            if loaded_data[1].is_defined():
                continue
            # load the new batch
            new_data[str(loaded_data[0])] = self.load_batch(loaded_data[0])
            # update the loaded data
            new_loaded_data = DataMuncher({**old_loaded_data, **new_data})
            with self._loaded_data_mu:  # most probably, we will not have to wait here
                self._loaded_data = new_loaded_data

    def _load_ahead_batch_thread(self) -> None:
        """method for the load ahead thread to run"""

        while True:

            # get the batch size that was loaded
            r, _, _ = select.select([self._r_data_load_ahead], [], [])
            batch = r[0].recv()

            # find the batch sizes that should be loaded
            start_idx = batch + 1
            end_idx = batch + 1 + self._load_ahead_batch_count
            batches_to_be_loaded = list(range(start_idx, end_idx))

            # correct the batch indices if necessary
            if self._data_loading_wrap_around is False and end_idx >= self.get_number_of_iterations():
                batches_to_be_loaded = list(range(start_idx,  self.get_number_of_iterations()))

            # do the batch ahead loading
            self._load_ahead_batches(batches_to_be_loaded)

    def __len__(self) -> int:
        """Gives the total number of iterations this data loader will go through.

        Returns
        -------
        Total number of batches in each epoch

        """

        return self.get_number_of_iterations()

    def __iter__(self):
        """Returns an iterable, self."""

        # reset the counter
        self._iterator_count = 0

        return self

    def __next__(self):
        """Returns the next set of data.

        Returns
        -------
        A collection of next set of data

        """

        # if the batch size is more that the amount of data left, go to beginning and return None
        if self._iterator_count > self.get_number_of_iterations():
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

        if self._load_ahead_enabled:
            # try to load from the already loaded data and if not exist, load manually

            with self._loaded_data_mu:
                data = self._loaded_data.get_value_option(str(item))

            # if not already loaded, load it
            if data.is_empty():
                data = self.load_batch(item)
            else:
                data = data.get()

            # now, let the load ahead know
            self._w_data_load_ahead.send(item)

        else:
            data = self.load_batch(item)

        return data
