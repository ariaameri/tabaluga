import concurrent.futures
import json
import pathlib
import re
import select
import threading
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Any, Dict

import colored
import numpy as np
import pandas as pd
import polars as pl

from ..base import base
from ..communicator import mpi
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.option import Some, Option, nothing
from ..util.result import Result, Err, Ok
from opentelemetry import trace
from tabaluga.util.tracer import TRACER_NAME
from tabaluga.util.util import EventMode

# Acquire a tracer
_tracer = trace.get_tracer(TRACER_NAME)

# a mapping between the column concepts and their names
@dataclass(frozen=True)
class MetadataColumns:
    folder_path: str = 'folder_path'
    folder_parent_path: str = 'folder_parent_path'
    folder_name: str = 'folder_name'
    file_name: str = 'file_name'
    file_extension: str = 'file_extension'
    path: str = 'path'
    # content_type: str = 'content_type'
    # metadata_sync_choice: str = 'metadata_sync_choice'
    # syncable: str = 'syncable'
    # id: str = 'id'
    bundle_id: str = 'bundle_id'
    # index: str = 'index'


@dataclass(frozen=True)
class MetadataColumnsTypes:
    folder_path = str
    folder_parent_path = str
    folder_name = str
    file_name = str
    file_extension = str
    path = str
    # content_type = str
    # metadata_sync_choice = pl.Boolean
    # syncable = pl.Boolean
    # id = pl.UInt64
    bundle_id = None
    # index = pl.Int64


metadata_columns = MetadataColumns()
metadata_columns_type = MetadataColumnsTypes()


@dataclass(frozen=True)
class MetadataColumnsCOCO:
    image_id: str = 'image_id'


@dataclass(frozen=True)
class MetadataColumnsCOCOType:
    image_id = pl.UInt64


metadata_columns_COCO = MetadataColumnsCOCO()
metadata_columns_COCO_type = MetadataColumnsCOCOType()


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
        self._shuffle_enabled: bool = self._config.get_or_else('shuffle.enabled', False)
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
        self._batch_size_val: int = self._config.get_or_else('batch_size_val', self.batch_size)
        self._batch_size_test: int = self._config.get_or_else('batch_size_test', self.batch_size)

        # build the data generator
        self._data_infos = self._config.get("data")
        self._data_processors, self._bundle_ids, self._bundle_id_2_processor = (
            self._build_dataloaders().expect("failed building dataloaders")
        )

        self._train_ids, self._val_ids, self._test_ids = self._get_train_val_test_indices()
        self._train_ids_full = self._train_ids  # for later distributing the indices

        # Create workers
        self.create_workers()

        # the shared multithreading pool
        self._multithreading = self._config.get_or_else('multithreading', False)
        self._thread_pool: Optional[ThreadPoolExecutor] = None

        if self._multithreading:
            multithreading_count = self._config.get_or_else('multithreading_count', 5)
            self._thread_pool = ThreadPoolExecutor(
                multithreading_count, thread_name_prefix="tabaluga-datamanager-thread-pool"
            )
            self._distribute_shared_multithreading_pool()

        self._distribute_train_val_test_ids()

    @_tracer.start_as_current_span(
        "tabaluga.data_manager.build_data_loaders"
    )
    def _build_dataloaders(self):

        data = []
        bundle_count = 0
        bundle_mapping = {}
        bundle_ids = []
        for data_info in self._data_infos:

            keys = list(data_info.keys())

            if len(keys) != 1:
                raise ValueError("data must have a single key")

            if keys[0] == "separate_files":
                data_processor = SeparateFilesData(ConfigParser(data_info["separate_files"]))
            elif keys[0] == "coco":
                data_processor = CocoData(ConfigParser(data_info["coco"]))
            else:
                return Err(ValueError(f"unsupported data type of {keys[0]}"))

            if (res := data_processor.sync()).is_err():
                return res
            # update the cumulative bundle ids
            current_bundle_count = data_processor.get_bundle_count()
            new_ids = list(range(bundle_count, bundle_count + current_bundle_count))
            bundle_ids.extend(new_ids)
            data_processor.set_bundle_ids(new_ids)
            for new_id in new_ids:
                bundle_mapping[new_id] = data_processor
            bundle_count += current_bundle_count
            data.append(data_processor)

        return Ok((data, bundle_ids, bundle_mapping))

    def _get_train_val_test_indices(self):

        bundle_ids = self._bundle_ids
        if self._shuffle_enabled:
            rand_state = np.random.get_state()
            np.random.seed(self._seed)
            bundle_ids = list(np.random.permutation(bundle_ids))
            np.random.set_state(rand_state)

        test_count = int(len(bundle_ids) * self._test_ratio)
        val_count = int((len(bundle_ids) - test_count) * self._val_ratio)

        test_ids = bundle_ids[:test_count]
        val_ids = bundle_ids[test_count:(test_count+val_count)]
        train_ids = bundle_ids[(test_count+val_count):]

        return train_ids, val_ids, test_ids

    @abstractmethod
    def _distribute_train_val_test_ids(self):
        """distributes the shared multithreading pool among all workers."""

        pass

    def _distribute_shared_multithreading_pool(self):
        """distributes the shared multithreading pool among all workers."""

        if self._multithreading:
            for worker in self._get_dataloader_workers():
                worker.set_shared_multithreading_pool(self._thread_pool)

    @abstractmethod
    def _get_dataloader_workers(self) -> List:
        """Returns the list of workers that the shared multithreading pool can be set to."""

        pass

    @abstractmethod
    def _get_dataloader_workers_train(self) -> List:
        """Returns the list of train data loader"""

        pass

    @abstractmethod
    def _get_dataloader_workers_val(self) -> List:
        """Returns the list of val data loader"""

        pass

    @abstractmethod
    def _get_dataloader_workers_test(self) -> List:
        """Returns the list of test data loader"""

        pass

    def _shuffle(self, rand_seed_add: int):

        if self._shuffle_enabled:
            rand_state = np.random.get_state()
            np.random.seed(self._seed + rand_seed_add)
            self._train_ids_full = list(np.random.permutation(self._train_ids_full))
            self._train_ids = list(np.random.permutation(self._train_ids))
            np.random.set_state(rand_state)

    def set_batch_size(self, batch_size: int, event_mode: EventMode = None) -> None:
        """Sets the batch size

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        if event_mode is None:
            self.batch_size = batch_size
            self._batch_size_train = batch_size
            self._batch_size_val = batch_size
            self._batch_size_test = batch_size
            # Set batch size of all the workers
            for worker in self._get_dataloader_workers():
                worker.set_batch_size(self.batch_size)
        elif event_mode == EventMode.train:
            self.batch_size = batch_size
            self._batch_size_train = batch_size
            for worker in self._get_dataloader_worker_train():
                worker.set_batch_size(self.batch_size)
        elif event_mode == EventMode.val:
            self.batch_size = batch_size
            self._batch_size_val = batch_size
            for worker in self._get_dataloader_worker_val():
                worker.set_batch_size(self.batch_size)
        elif event_mode == EventMode.test:
            self.batch_size = batch_size
            self._batch_size_test = batch_size
            for worker in self._get_dataloader_worker_test():
                worker.set_batch_size(self.batch_size)

        self._log.info(f"batch size set to {colored.fg('cyan')}{batch_size}{colored.attr('reset')}")

    def sync_and_shuffle(self, rand_seed_add: int):

        # first, shuffle the data
        self._shuffle(rand_seed_add)

        if not mpi.mpi_communicator.is_distributed():
            self._distribute_train_val_test_ids()
            return Ok(None)

        def split_ids_for_nodes(ids: List, batch_size: int, node_count: int) -> List[List]:

            total_count = len(ids)
            total_batch_count = int(total_count / batch_size)
            batch_count_per_node = int(total_batch_count / node_count)
            item_count_per_node = int(batch_count_per_node * batch_size)

            ids_nodes = []
            for start_idx in range(0, item_count_per_node * node_count, item_count_per_node):
                ids_nodes.append(ids[start_idx:(start_idx+item_count_per_node)])

            return ids_nodes

        if (res_train_ids_full := mpi.mpi_communicator.collective_bcast(self._train_ids_full, root_rank=0)).is_err():
            return res_train_ids_full

        train_ids = (
            split_ids_for_nodes(self._train_ids_full, self._batch_size_train, mpi.mpi_communicator.get_size())
            if mpi.mpi_communicator.is_main_rank() else None
        )
        if (res_train_ids := mpi.mpi_communicator.collective_scatter(train_ids, root_rank=0)).is_err():
            return res_train_ids

        self._train_ids_full = res_train_ids_full.get()
        self._train_ids = res_train_ids.get()
        self._val_ids = self._val_ids if mpi.mpi_communicator.is_main_rank() else []
        self._test_ids = self._test_ids if mpi.mpi_communicator.is_main_rank() else []

        self._distribute_train_val_test_ids()
        return Ok(None)


class DataLoaderManager(base.BaseEventManager, ABC):
    """This abstract class manages the data loaders and gets input from DataManager."""

    def __init__(self, idx_2_processor: Dict[int, 'Data'], config: ConfigParser = None):

        super().__init__(config)

        self._idx_2_processor: Dict[int, 'Data'] = idx_2_processor

        # placeholder for the shared multithreading pool
        self._shared_multithreading_pool: Optional[ThreadPoolExecutor] = None

        # bookkeeping for iterator, batch size and number of iterations (batches in an epoch)
        self._iterator_count = 0
        self.batch_size: int = -1
        self.number_of_iterations: int = -1

        self._indices: List[int] = []

        # placeholder for the multithreading pool
        self._thread_pool: Optional[ThreadPoolExecutor] = None

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

        self._thread_pool = pool

    @abstractmethod
    def _get_dataloader_workers(self) -> List:
        """Returns the list of workers that the shared multithreading pool can be set to."""

        pass

    def set_indices(self, indices: List[int]):

        self._indices = indices

        self._update_num_iterations()

    def set_batch_size(self, batch_size: int) -> None:
        """Sets the batch size and thus finds the total number of batches in one epoch.

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        self.batch_size = batch_size

        self._update_num_iterations()

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

    def _update_num_iterations(self):

        self.number_of_iterations = int(len(self._indices) / self.batch_size)

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

        return self._load_batch(batch)

    @_tracer.start_as_current_span(
        "tabaluga.data_loader_manager.load_batch"
    )
    def _load_batch(self, item: int):
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

        span = trace.get_current_span()
        span.set_attributes({
            "batch_size": self.batch_size,
            "batch": item,
        })
        span_context = span.get_span_context()

        start_idx = item * self.batch_size
        if self._thread_pool is not None:
            data = [
                self._thread_pool.submit(self._idx_2_processor[idx].load_bundle, idx, span_context)
                for idx
                in self._indices[start_idx:(start_idx+self.batch_size)]
            ]
            data = [_.result() for _ in data]
        else:
            data = [
                self._idx_2_processor[idx].load_bundle(idx)
                for idx in
                self._indices[start_idx:(start_idx+self.batch_size)]
            ]

        return self._prepare_loaded_data(data)

    @abstractmethod
    def _prepare_loaded_data(self, data: List[DataMuncher]):
        pass

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
                batches_to_be_loaded = list(range(start_idx, self.get_number_of_iterations()))

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

        with _tracer.start_as_current_span(
            "tabaluga.data_loader_manager.get_data",
            attributes={
                "batch_size": self.batch_size,
                "batch": batch,
            }
        ) as span:
            if self._load_ahead_enabled:
                # try to load from the already loaded data and if not exist, load manually

                with self._loaded_data_mu:
                    data = self._loaded_data.get_value_option(str(batch))

                # if not already loaded, load it
                if data.is_empty():
                    data = self._load_batch_wrap(batch, *args, **kwargs)
                else:
                    span.set_attributes("from", "memory")
                    data = data.get()

                # now, let the load ahead know
                self._w_data_load_ahead.send(batch)

            else:
                data = self._load_batch_wrap(batch, *args, **kwargs)

        return data


class Data(base.BaseWorker, ABC):

    @abstractmethod
    def get_bundle_count(self) -> int:
        pass

    @abstractmethod
    def set_bundle_ids(self, ids: List[int]) -> Result[None, Exception]:
        pass

    @abstractmethod
    def load_bundle(self, bundle_id: int, otel_context=None) -> DataMuncher:
        pass

    @abstractmethod
    def sync(self):
        pass


class SeparateFilesData(Data):

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # Folders containing the data
        self._folders: List[str] = []

        # list of extensions to ignore and accept
        self._extension_ignore: List[re.Pattern] = [
            re.compile(item) for item in self._config.get_or_else('extension_ignore', [])
        ]
        self._extension_accept: List[re.Pattern] = [
            re.compile(item) for item in self._config.get_or_else('extension_accept', [])
        ]

        self._iwholepath_regex: List[re.Pattern] = [
            re.compile(item) for item in self._config.get_or_else('iwholepath_regex', [])
        ]
        self._iwholepath_regex_ignore: List[re.Pattern] = [
            re.compile(item) for item in self._config.get_or_else('iwholepath_regex_ignore', [])
        ]

        # Folders containing the data
        self._folders: List[str] = self._config.get_value_option('folders').expect('folder config does not exist')
        # self._add_train_folders: List[str] = self._config.get_or_else('additional_train_folders', [])
        # self._add_val_folders: List[str] = self._config.get_or_else('additional_val_folders', [])
        # self._add_test_folders: List[str] = self._config.get_or_else('additional_test_folders', [])
        # self._add_train_folders = [] if self._add_train_folders == [None] else self._add_train_folders
        # self._add_val_folders = [] if self._add_val_folders == [None] else self._add_val_folders
        # self._add_test_folders = [] if self._add_test_folders == [None] else self._add_test_folders

        # File names in nested lists
        self._deep_folders: bool = self._config.get_or_else('deep', False)

        # how to create criterion
        self._criterion: List[str] = \
            self._config.get_value_option('criterion_generation.criterion')\
            .expect('criterion config does not exist')
        self._criterion_hash: bool = self._config.get_or_else('criterion_generation.hash', True)
        self._check_criterion(self._criterion)

        self._metadata = self._build_and_get_metadata(self._folders)

    def _check_criterion(self, criterion: List[str]):
        """Checks if the given criterion is ok"""

        # the criterion values should be from the columns' names
        for criteria in criterion:
            if criteria not in metadata_columns.__dict__.keys():
                self._log.error(
                    f"criteria '{criteria}' not accepted. Only the following criterion are accepted:"
                    + '\n\t - '
                    + '\n\t - '.join(metadata_columns.__dict__.keys())
                    + '\n'
                )
                raise ValueError("unknown criteria")

    def _build_and_get_metadata(self, folders: List[str]) -> pl.DataFrame:
        """
        This method goes over the specified folders, read files and creates and returns a pandas data frame from them.

        Returns
        -------
        pl.DataFrame
            pandas dataframe of the information

        """

        self._check_populate_folder_metadata(folders)
        metadata = self._build_and_get_metadata_folders(folders)
        metadata = self._bundle_metadata(metadata)

        return metadata

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

    def _check_file(self, file_path: Path) -> bool:
        """"Helper function to check a single file.

        Parameters
        ----------
        file_path : Path
            The path of the file
        """

        # Check that the file is not a folder
        if file_path.is_file() is False:
            return False

        if self._filter_file_path(file_path) is False:
            return False

        # Check criteria for the file name
        if self._filter_file_name(file_path.name) is False:
            return False

        return True

    def _filter_file_path(self, file_path: Path) -> bool:
        """
        Checks the file path to see if it is acceptable or not.

        Parameters
        ----------
        file_path : Path

        Returns
        -------
        bool

        """

        if self._iwholepath_regex and \
                (any([reg.match(file_path.as_posix().lower()) is not None for reg in self._iwholepath_regex]) is False):
            return False

        for reg in self._iwholepath_regex_ignore:
            if reg.match(file_path.as_posix().lower()) is not None:
                return False

        return True

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

    def _build_and_get_metadata_folders(self, folders: List[str] = None) -> pl.DataFrame:
        """
        This method goes over the specified folders, read files and creates and returns a pandas data frame from them.

        Parameters
        ----------
        folders : List[str], optional
            list of folders to look into. if not given, the default folder is assumed

        Returns
        -------
        pl.DataFrame
            dataframe of the information

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
        file_paths = [
            file_path
            for file_paths in file_nested_paths
            for file_path in file_paths
            if self._check_file(file_path)
        ]
        # Retrieve the folder path and file names
        folder_paths = [file_name.parent for file_name in file_paths]
        folder_parent_paths = [folder_path.parent for folder_path in folder_paths]
        folder_names = [folder_path.name for folder_path in folder_paths]
        file_names = [file_path.stem for file_path in file_paths]
        file_extensions = [file_path.suffix.lower() for file_path in file_paths]

        # Create data frame of all the files in the folder
        metadata = pl.DataFrame(
            data={
                metadata_columns.folder_path: [str(item) for item in folder_paths],
                metadata_columns.folder_parent_path: [str(item) for item in folder_parent_paths],
                metadata_columns.folder_name: folder_names,
                metadata_columns.file_name: file_names,
                metadata_columns.file_extension: file_extensions,
                metadata_columns.path: [str(item) for item in file_paths],
                # metadata_columns.metadata_sync_choice: True,
                # metadata_columns.syncable: True,
                # metadata_columns.id: ids,
            },
            schema={
                metadata_columns.folder_path: metadata_columns_type.folder_path,
                metadata_columns.folder_parent_path: metadata_columns_type.folder_parent_path,
                metadata_columns.folder_name: metadata_columns_type.folder_name,
                metadata_columns.file_name: metadata_columns_type.file_name,
                metadata_columns.file_extension: metadata_columns_type.file_extension,
                metadata_columns.path: metadata_columns_type.path,
                # metadata_columns.metadata_sync_choice: metadata_columns_type.metadata_sync_choice,
                # metadata_columns.syncable: metadata_columns_type.syncable,
                # metadata_columns.id: metadata_columns_type.id,
            }
        )

        return metadata

    def _add_bundle_id_column(self, metadata: pl.DataFrame) -> pl.DataFrame:
        """
        Creates the bundle id column and returns it

        Parameters
        ----------
        metadata : pl.DataFrame
            the metadata

        Returns
        -------
        pl.DataFrame
            the new metadata with the new column updated

        """

        if len(metadata) > 0:
            import hashlib
            metadata = metadata.with_columns(
                pl.concat_str(
                    pl.col([getattr(metadata_columns, item) for item in self._criterion]), separator="+++"
                )
                .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
                .alias(metadata_columns.bundle_id)
            )
        else:
            # add empty str literal cause the column type is str
            metadata = metadata.with_columns(
                pl.lit("").alias(metadata_columns.bundle_id),
            )

        return metadata

    def _check_print_bundle_sanity(self, metadata: pl.DataFrame) -> None:
        """Checks the given metadata and makes some info if necessary."""

        count_df = metadata.groupby(metadata_columns.bundle_id).count()

        from collections import Counter
        count = Counter(count_df['count'])
        if (length := len(count.values())) == 0:
            self._log.error(f"ended up in a weird situation: data bundles seem to not exist!")
        elif length == 1:
            self._log.info(f"data bundles contain {list(count.keys())[0]} elements each")
        else:
            warn = \
                "data bundles contain different amount of elements. I found:" \
                "\n\t (bundle size) (count)" \
                "\n\t\t (samples' path)"
            for c, v in count.items():
                warn += f"\n\t- {c}: {v}"
                # get the elements that have c counts
                bundle_ids = count_df.filter(pl.col('count') == c)[:10][metadata_columns.bundle_id]
                sample_df_str = "- " + "\n- ".join(
                    list(metadata.filter((pl.col(metadata_columns.bundle_id).is_in(bundle_ids)))[metadata_columns.path])
                )
                from tabaluga.util.util import REGEX_INDENT_NEW_LINE_ONLY
                sample_df_str = REGEX_INDENT_NEW_LINE_ONLY.sub('\n\t\t ', sample_df_str)

                warn += f"\n\t\t {sample_df_str}"

            warn += "\n"

            self._log.warning(warn)

    def _bundle_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:

        # add criterion column
        metadata = self._add_bundle_id_column(metadata)

        self._check_print_bundle_sanity(metadata)

        return metadata

    def get_bundle_count(self) -> int:

        return self._metadata.get_column(metadata_columns.bundle_id).n_unique()

    def set_bundle_ids(self, ids: List[int]) -> Result[None, Exception]:

        bundle_ids = self._metadata.get_column(metadata_columns.bundle_id).unique(maintain_order=True)
        if len(bundle_ids) != len(ids):
            return Err(Exception("got wrong number of ids"))
        mapping = {old: new for old, new in zip(bundle_ids, ids)}
        self._metadata = self._metadata.with_columns(pl.col(metadata_columns.bundle_id).replace(mapping))

        return Ok(None)

    _processor = nothing

    def set_processor(self, processor):
        self._processor = Some(processor)

    def load_bundle(self, bundle_id: int) -> DataMuncher:
        raise NotImplementedError

    def sync(self) -> Result[None, Exception]:

        if not mpi.mpi_communicator.is_distributed():
            return Ok(None)

        if (res := mpi.mpi_communicator.collective_bcast(self._metadata, root_rank=0)).is_err():
            return res
        self._metadata = res.get()

        return Ok(None)


class CocoData(Data):

    def __init__(self, config: ConfigParser):

        super().__init__(config)

        self._coco_path = (
            self._config
            .get_value_option('coco_annotation_path')
            .expect("could not find coco_annotation_path in the config")
        )
        self._coco_json = (
            DataMuncher(
                self._read_json(pathlib.Path(self._coco_path))
                .expect(f"error reading coco json at path {self._coco_path}")
            )
        )
        self._categories = (
            self._coco_json
            .get_value_option('categories')
            .expect("could not find categories in the coco json")
        )
        self._coco_annotations = (
            self._gather_annotations(
                self._coco_json
                .get_value_option('annotations')
                .expect("could not find annotations in the coco json")
            )
            .expect(f"error gathering coco annotations of path {self._coco_path}")
        )
        self._img_dir_relative_path = (
            self._config
            .get_value_option('image_dir_relative_path')
            .expect("could not find image_dir_relative_path in the config")
        )
        self._img_dir_path = pathlib.Path(self._coco_path).parent / self._img_dir_relative_path

        self._metadata = (
            self._build_metadata()
            .expect(f"error while building metadata for coco at path {self._coco_path}")
        )

    @staticmethod
    def _read_json(path: pathlib.Path) -> Result[Dict, Exception]:

        with open(path.as_posix()) as f:
            return Ok(json.load(f))

    def _build_metadata(self) -> Result[pl.DataFrame, Exception]:

        try:
            img_paths = [
                self._img_dir_path / img.get("file_name") for img in self._coco_json.get('images')
            ]

            img_ids = [
                int(img.get("id")) for img in self._coco_json.get('images')
            ]

            for img_path in img_paths:
                if not img_path.exists():
                    raise RuntimeError(f"cannot find the image at path {img_path}")

            metadata = pl.DataFrame(
                data={
                    metadata_columns_COCO.image_id: img_ids,
                    metadata_columns.folder_path: [str(_.parent) for _ in img_paths],
                    metadata_columns.folder_parent_path: [_.parent.parent.name for _ in img_paths],
                    metadata_columns.folder_name: [_.parent.name for _ in img_paths],
                    metadata_columns.file_name: [_.name for _ in img_paths],
                    metadata_columns.file_extension: [_.suffix for _ in img_paths],
                    metadata_columns.path: [str(item) for item in img_paths],
                },
                schema={
                    metadata_columns_COCO.image_id: metadata_columns_COCO_type.image_id,
                    metadata_columns.folder_path: metadata_columns_type.folder_path,
                    metadata_columns.folder_parent_path: metadata_columns_type.folder_parent_path,
                    metadata_columns.folder_name: metadata_columns_type.folder_name,
                    metadata_columns.file_name: metadata_columns_type.file_name,
                    metadata_columns.file_extension: metadata_columns_type.file_extension,
                    metadata_columns.path: metadata_columns_type.path,
                }
            )
        except Exception as e:
            return Err(e)

        return Ok(metadata)

    @staticmethod
    def _gather_annotations(annotations) -> Result[Dict[int, Dict[int, List]], Exception]:

        from collections import defaultdict
        coco_annotations = defaultdict(list)

        try:
            for anno in annotations:
                coco_annotations[int(anno['image_id'])].append(anno)

            coco_annotations_final = {img_id: defaultdict(list) for img_id in coco_annotations.keys()}
            for img_id, annos in coco_annotations.items():
                for anno in annos:
                    coco_annotations_final[img_id][anno['category_id']].append(anno)
        except Exception as e:
            return Err(e)

        return Ok(coco_annotations_final)

    def get_bundle_count(self) -> int:

        return len(self._metadata)

    def set_bundle_ids(self, ids: List[int]) -> Result[None, Exception]:

        if len(self._metadata) != len(ids):
            return Err(Exception("got wrong number of ids"))
        self._metadata = self._metadata.with_columns(pl.Series(name=metadata_columns.bundle_id, values=ids))

        return Ok(None)

    _processor = nothing

    @staticmethod
    def set_processor(processor):
        CocoData._processor = Some(processor)

    def load_bundles(self, bundle_ids: List[int]) -> List[DataMuncher]:

        metadata_rows = self._metadata.filter(pl.col(metadata_columns.bundle_id).is_in(bundle_ids))
        image_ids = list(metadata_rows[metadata_columns_COCO.image_id])
        annos = [self._coco_annotations.get(image_id, {}) for image_id in image_ids]
        res = self._processor.map(lambda x: x.process(metadata_rows, annos, self._categories)).get()

        return res

    def load_bundle(self, bundle_id: int, otel_context=None) -> DataMuncher:

        metadata_row = self._metadata.filter(pl.col(metadata_columns.bundle_id) == bundle_id)
        image_id = metadata_row[metadata_columns_COCO.image_id][0]
        anno = self._coco_annotations.get(image_id, {})
        with _tracer.start_as_current_span("tabaluga.data_loader.coco.processor.load"):
            res = self._processor.map(
                lambda x: x.process(metadata_row, anno, self._categories, otel_context=otel_context)
            ).get()

        return res

    def sync(self) -> Result[None, Exception]:

        if not mpi.mpi_communicator.is_distributed():
            return Ok(None)

        if (res := mpi.mpi_communicator.collective_bcast(self._metadata, root_rank=0)).is_err():
            return res
        self._metadata = res.get()
        if (res := mpi.mpi_communicator.collective_bcast(self._coco_json.dict_representation(), root_rank=0)).is_err():
            return res
        self._coco_json = DataMuncher(res.get())
        if (res := mpi.mpi_communicator.collective_bcast(self._categories, root_rank=0)).is_err():
            return res
        self._categories = res.get()
        if (res := mpi.mpi_communicator.collective_bcast(self._coco_annotations, root_rank=0)).is_err():
            return res
        self._coco_annotations = res.get()
        if (res := mpi.mpi_communicator.collective_bcast(self._img_dir_relative_path, root_rank=0)).is_err():
            return res
        self._img_dir_relative_path = res.get()

        return Ok(None)










































