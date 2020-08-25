from abc import ABC, abstractmethod

from ..base import base
from ..util.config import ConfigParser
import os
from typing import List
import pandas as pd
import numpy as np


class DataManager(base.BaseEventManager, ABC):
    """This abstract class manages the DataLoader or DataLoader Managers.

    It is responsible to distribute the train/validation/test metadata among the data loaders."""

    def __init__(self, config: ConfigParser):
        """Initializer for the data manager.

        Parameters
        ----------
        config : ConfigParser
            The configurations for the data manager.
        """

        super().__init__(config)

        # Folders containing the data
        self._folders: List[str] = []

        # Get the input type
        self._input_type: str = config.get('input_type')

        # Get random seed and shuffle boolean
        self._seed = config.get_or_else('seed', None)
        self._shuffle: bool = config.get_or_else('shuffle', False)

        # Get test and validation ratios
        self._test_ratio: float = config.get_or_else('test_ratio', 0)
        self._val_ratio: float = config.get_or_else('val_ratio', 0)

        # Set batch size
        self.batch_size: int = config.get('batch_size')

        # Pandas data frame to hold the metadata of the data
        self.metadata: pd.DataFrame
        self.test_metadata: pd.DataFrame
        self.val_metadata: pd.DataFrame
        self.train_metadata: pd.DataFrame

        # Train, val, and test DataLoaderManager placeholders
        self.workers['train']: DataLoaderManager
        self.workers['val']: DataLoaderManager
        self.workers['test']: DataLoaderManager

        # Create general and train, val, test metadata
        self.create_metadata()

        # Create workers
        self.create_workers()

    def create_metadata(self) -> None:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Read and create the data metadata
        if self._input_type == 'folder_path':
            self._build_metadata_from_folder()
        elif self._input_type == 'mongo':
            self._build_metadata_from_mongo()

        # Regroup the metadata based on the criteria of file name
        self._regroup_metadata('file_name')

        # Generate the train, validation, and test metadata
        self._generate_train_val_test_metadata()

    def _check_file(self, file_path: str) -> bool:
        """"Helper function to check a single file.

        Parameters
        ----------
        file_path : str
            The path of the file
        """

        # Check that the file is not a folder
        check = os.path.isfile(file_path)

        # Check criteria for the file name
        check = check & self._filter_file_name(file_path.split('/')[-1])

        return check

    def _filter_file_name(self, file_name: str) -> bool:
        """"Helper function to filter a single file based on its name and criteria.

        Parameters
        ----------
        file_name : str
            The path of the file
        """

        check = False if file_name.startswith('.') else True

        return check

    def _build_metadata_from_folder(self) -> None:
        """This method goes over the specified folders, read files and creates a pandas data frame from them."""

        # Folders containing the data
        self._folders: List[str] = self._config.get('folders')

        # Check if folder list is given
        if self._folders is None:
            raise Exception("No folders given to read files from!")
        for folder in self._folders:
            if os.path.exists(folder) is False:
                raise Exception(f'The folder {folder} does not exist!')

        file_nested_names = [os.listdir(folder) for folder in self._folders]  # File names in nested lists
        # Flatten the file names and make absolute paths
        file_paths = [os.path.join(folder_name, file_name)
                      for file_list, folder_name in zip(file_nested_names, self._folders)
                      for file_name in file_list]
        # Check each file based on the criteria
        file_paths = [file_path for file_path in file_paths if self._check_file(file_path)]
        # Retrieve the folder path and file names
        folder_paths = [os.path.dirname(file_name)
                        for file_name in file_paths]
        folder_names = [os.path.split(folder_path)[1]
                        for folder_path in folder_paths]
        file_names = [os.path.split(file_path)[1]
                      for file_path in file_paths]
        file_extension = [file_name.split('.')[1].lower()
                          for file_name in file_names]
        file_names = [file_name.split('.')[0]
                      for file_name in file_names]

        # Create data frame of all the files in the folder
        self.metadata = pd.DataFrame({
            'folder_path': folder_paths,
            'folder_name': folder_names,
            'file_name': file_names,
            'file_extension': file_extension,
            'path': file_paths
        })

    def _build_metadata_from_mongo(self) -> None:
        """Creates metadata based on the information retrived from mongoDB."""

        pass

    def _regroup_metadata(self, criterion=None) -> None:
        """Groups the metadata.

        Each group of data (e.g. containing data and label) should have.
        Each group must have its own unique index, where indices are range.
        Each group will be recovered by metadata.loc[index]

        Parameters
        ----------
        criterion : str or List[str]
            The name of the columns based on which the metadata should be categorized

        """

        if criterion is None:
            return

        # Group based on the criterion
        metadata = self.metadata.groupby(criterion).apply(lambda x: x.reset_index(drop=True))

        # Rename the indices to be range
        # Also rename the index level 0 name to be 'index' (instead of `criterion`)
        metadata = metadata.rename(
            index={key: value for value, key in enumerate(metadata.index.get_level_values(0).unique(), start=0)}
        )
        metadata.index.names = [None, *metadata.index.names[1:]]

        self.metadata = metadata

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


class DataLoaderManager(base.BaseEventManager, ABC):
    """This abstract class manages the data loaders and gets input from DataManager."""

    def __init__(self, config: ConfigParser, metadata: pd.DataFrame):
        """Initializer for data loader manager.

        Parameters
        ----------
        config : ConfigParser
            The configuration for the instance
        metadata : pd.DataFrame
            The metadata for the data to be loaded
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
        selection = [self._check_file(file_path) for file_path in self.metadata['path']]

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

    def __init__(self, config: ConfigParser, metadata: pd.DataFrame):
        """Initializer for data loader.

        Parameters
        ----------
        config : ConfigParser
            The configuration for the instance
        metadata : pd.DataFrame
            The metadata for the data to be loaded
        """

        super().__init__(config)

        # Set the metadata
        self.metadata: pd.DataFrame = metadata

        # Book keeping for the batch size and thus the number of iterations (batches) in each epoch
        self.batch_size: int = -1
        self.number_of_iterations: int = -1

        # Book keeping for iterator
        self._iterator_count = 0

    def modify_metadata(self) -> None:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Make a selection of the metadata
        selection = [self._check_file(file_path) for file_path in self.metadata['path']]

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

    @abstractmethod
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
