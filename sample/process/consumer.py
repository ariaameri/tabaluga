from framework.process.process import ProcessManager, Process
from framework.process.cuda import CUDAInformation
from framework.process.image import BackgroundToColor, ImageResizer, ImageNormalizer, BWHCToBCWH, OneHotDecoder
from framework.util.config import ConfigParser
from typing import Dict


class SampleProcessManager(ProcessManager):
    """Simple ProcessManager class that manages processes and pre- and post-process managers."""

    def __init__(self, config: ConfigParser):
        """Initializes the instance.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed

        """

        super().__init__(config)

        self.create_workers()

    # def __str__(self):
    #     """Short explanation of the instance."""
    #
    #     string = f'Image process manager'
    #
    #     return string

    def create_workers(self):
        """Creates the pre- and post-processing managers as workers."""

        self.workers['cuda'] = CUDAInformation(self._config.get_or_else('cuda_information', ConfigParser()))
        self.workers['preprocess'] = SampleImagePreprocessManager(self._config.get('preprocess'))

    def on_begin(self, info: Dict = None):
        """at the beginning of everything"""

        self.workers['cuda'].process()

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) batch, process the loaded train data."""

        data = info['data']
        processed_data = self.workers['preprocess'].on_batch_begin({'data': data})

        return processed_data

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, process the loaded val data."""

        data = info['data']
        processed_data = self.workers['preprocess'].on_val_begin({'data': data})

        return processed_data


class SampleImagePreprocessManager(ProcessManager):
    """A simple class to manage Preprocess instances."""

    def __init__(self, config: ConfigParser):
        """Initializer.

        Parameters
        ----------
        config : ConfigParser
            The configuration needed for this instance and its workers.

        """

        super().__init__(config)

        self.create_workers()

    def create_workers(self):
        """Creates Preprocess instances."""

        self.workers['labels_background_to_color'] = BackgroundToColor(ConfigParser())

        self.workers['image_resizer'] = ImageResizer(self._config.get('resize'))

        self.workers['image_normalizer'] = ImageNormalizer()

        self.workers['image_bwhc_to_bcwh'] = BWHCToBCWH()

        self.workers['label_one_hot_decoder'] = OneHotDecoder(ConfigParser({"axis": 1}))

    def on_batch_begin(self, info: Dict = None):
        """On beginning of (train) epoch, process the loaded train image data."""

        # data = info['data']['data']
        data = info['data']

        # labels = data.get('labels')
        # labels = self.workers['labels_background_to_color'].process(labels)
        # processed_data = data.update('labels', labels)
        #
        # # processed_data = self.workers['image_resizer'].resize(data)
        # processed_data = processed_data.map(self.workers['image_resizer'].process)
        # # processed_data = self.workers['image_normalizer'].normalize(processed_data)
        # processed_data = processed_data.map(self.workers['image_normalizer'].process)
        # processed_data = processed_data.map(self.workers['image_bwhc_to_bcwh'].process)
        #
        # labels = processed_data.get('labels')
        # labels = self.workers['label_one_hot_decoder'].process(labels)
        # processed_data = processed_data.update('labels', labels)

        processed_data = \
            data \
                .update_map({'_bc': {'$regex': 'labels$'}}, self.workers['labels_background_to_color'].process) \
                .update_map({}, [
                self.workers['image_resizer'].process,
                self.workers['image_normalizer'].process,
                self.workers['image_bwhc_to_bcwh'].process
            ]) \
                .update_map({'_bc': {'$regex': 'labels$'}}, self.workers['label_one_hot_decoder'].process)

        return processed_data

    def on_val_batch_begin(self, info: Dict = None):
        """On beginning of val epoch, process the loaded val image data."""

        data = info['data']

        labels = data.labels
        labels = self.workers['labels_background_to_color'].process(labels)
        processed_data = data.update('labels', labels)

        # processed_data = self.workers['image_resizer'].resize(data)
        processed_data = processed_data.map(self.workers['image_resizer'].process)
        # processed_data = self.workers['image_normalizer'].normalize(processed_data)
        processed_data = processed_data.map(self.workers['image_normalizer'].process)
        processed_data = processed_data.map(self.workers['image_bwhc_to_bcwh'].process)

        labels = processed_data.labels
        labels = self.workers['label_one_hot_decoder'].process(labels)
        processed_data = processed_data.update('labels', labels)

        return processed_data