from ..base import base
from ..util.config import ConfigParser
import os
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataManager(base.BaseManager):
    """This abstract class manages the DataLoader or DataLoader Managers.

    It is responsible to distribute the train/validation/test metadata among the data loaders."""

    def __init__(self, config: ConfigParser):
        """Initializer for the data manager.

        Parameters
        ----------
        config : ConfigParser
            The configurations for the data manager.
        """

        super().__init__()

        # Save the config
        self._config = config

        # Folders containing the data
        self._folders: List[str] = []

        # Get the input type
        self._input_type: str = config.input_type

        # Get random seed and shuffle boolean
        self._seed = config.seed if config.seed is not None else None
        self._shuffle: bool = config.shuffle if config.shuffle is not None else False

        # Get test and validation ratios
        self._test_ratio: float = config.test_ratio if config.test_ratio is not None else 0
        self._val_ratio: float = config.val_ratio if config.val_ratio is not None else 0

        # Pandas data frame to hold the metadata of the data
        self.metadata: pd.DataFrame
        self.test_metadata: pd.DataFrame
        self.val_metadata: pd.DataFrame
        self.train_metadata: pd.DataFrame

    def create_metadata(self) -> None:
        """Checks how to create metadata from input source and create train, validation, and test metadata."""

        # Read and create the data metadata
        if self._input_type == 'folder_path':
            self._build_metadata_from_folder()
        elif self._input_type == 'mongo':
            self._build_metadata_from_mongo()

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
        self._folders: List[str] = self._config.folders

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
        file_names = [file_name.split('/')[-1]
                      for file_name in file_paths]

        # Create data frame of all the files in the folder
        self.metadata = pd.DataFrame({
            'folder': folder_paths,
            'filename': file_names,
            'path': file_paths
        })

    def _build_metadata_from_mongo(self) -> None:
        """Creates metadata based on the information retrived from mongoDB."""

        pass

    def _generate_train_val_test_metadata(self) -> None:
        """The method creates train, validation, and test metadata."""

        # Get count of each set
        total_data_count = len(self.metadata)
        test_count = int(total_data_count * self._test_ratio)
        val_count = int((total_data_count - test_count) * self._test_ratio)
        train_count = total_data_count - test_count - val_count

        # Find the indices of each set
        indices = np.arange(total_data_count) \
            if self._shuffle is False \
            else np.random.permutation(total_data_count)
        test_indices = indices[:test_count]
        val_indices = indices[test_count:(test_count+val_count)]
        train_indices = indices[(test_count+val_count):]

        # Create the train, validation, and test metadata
        self.test_metadata = self.metadata.iloc[test_indices]
        self.val_metadata = self.metadata.iloc[val_indices]
        self.train_metadata = self.metadata.iloc[train_indices]

        # Update the column names of the data frames
        [self.train_metadata, self.val_metadata, self.test_metadata] = \
            [df.assign(original_index=df.index).reset_index(drop=True)
             for df
             in [self.train_metadata, self.val_metadata, self.test_metadata]]


class DataLoaderManager(base.BaseManager):
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

        super().__init__()

        # Set the config and metadata
        self._config = config
        self.metadata = metadata

        # Modify the metadata
        self.modify_metadata()

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


class DataLoader(base.BaseWorker):
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

        super().__init__()

        # Set the config and metadata
        self._config: ConfigParser = config
        self.metadata: pd.DataFrame = metadata

        # Book keeping for the batch size and thus the number of iterations (batches) in each epoch
        self.batch_size: int
        self.number_of_iterations: int

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

    def set_batch_size(self, batch_size: int) -> None:
        """Sets the batch size and thus finds the total number of batches in one epoch.

        Parameter
        ---------
        batch_size : int
            Batch size

        """

        self.batch_size = batch_size
        self.number_of_iterations = len(self.metadata) // batch_size

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

        pass

    def __len__(self) -> int:
        """Gives the total number of iterations this data loader will go through.

        Returns
        -------
        Total number of batches in each epoch

        """

        return self.number_of_iterations

    def __next__(self):
        """Returns the next set of images.

        Returns
        -------
        A list of next set of images : List[np.ndarray]

        """

        # if the batch size is more that the amount of data left, go to beginning
        if self._iterator_count >= self.number_of_iterations:
            self._iterator_count = 0
        # Find the correct begin index and slice the metadata
        begin_index = self._iterator_count * self.batch_size
        metadata = self.metadata.iloc[begin_index:(begin_index + self.batch_size)]

        # Load the images
        images = self.load_data(metadata)

        return images

    def __getitem__(self, item: int):
        """Returns the item-th batch of the image data.

        Parameters
        ----------
        item : int
            The index of the batch of the image data to be returned

        Returns
        -------
        A list of the image data loaded : List[np.ndarray]

        """

        # Check if item count is sensible
        assert item < self.number_of_iterations, \
            f'Requested number of images to be loaded goes beyond the end of available data.'

        # Find the corresponding metadata
        begin_index = item * self.batch_size
        metadata = self.metadata.iloc[begin_index:(begin_index + self.batch_size)]

        # Load the images
        images = self.load_data(metadata)

        return images
