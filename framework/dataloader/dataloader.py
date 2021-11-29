from abc import ABC, abstractmethod
from pathlib import Path
from ..base import base
from ..util.config import ConfigParser
from typing import List
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

        # Create and modify general and train, val, test metadata
        self.create_metadata()
        self.modify_metadata()
        self._generate_train_val_test_metadata()

        # Create workers
        self.create_workers()

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
        metadata = metadata.rename(
            index={key: value for value, key in enumerate(metadata.index.get_level_values(0).unique(), start=0)}
        )
        metadata.index.names = [None, *metadata.index.names[1:]]

        self.metadata = metadata

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




































