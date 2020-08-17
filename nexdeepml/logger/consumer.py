from .logger import LoggerManager, Logger, TQDMLogger, TheProgressBarLogger
from ..util.config import ConfigParser
from typing import Dict


class SampleLoggerManager(LoggerManager):

    def __init__(self, config: ConfigParser):

        super().__init__(config)

        self.create_workers()

    def create_workers(self):

        self.workers['train_tqdm']: TQDMLogger

    def on_train_begin(self, info: Dict = None):

        self.workers['train_tqdm']: TQDMLogger = TQDMLogger(self._config.tqdm)

    def on_train_epoch_begin(self, info: Dict = None):

        number_of_iterations = info['number_of_iterations']
        epochs = info['epochs']
        epoch = info['epoch']

        self.workers['train_tqdm'].set_number_epochs(epochs)
        self.workers['train_tqdm'].reset(number_of_iterations)
        self.workers['train_tqdm'].update(0, {'epoch': epoch})

    def on_batch_end(self, info: Dict = None):

        batch_size = info.pop('batch_size')
        self.workers['train_tqdm'].update(batch_size, info)

    def on_end(self, info: Dict = None):

        self.workers['train_tqdm'].close()


class SampleTheProgressBarLoggerManager(LoggerManager):

    def __init__(self, config: ConfigParser):

        super().__init__(config)

        self.create_workers()

    def create_workers(self):

        self.workers['train_tpb']: TheProgressBarLogger

    def on_train_begin(self, info: Dict = None):

        self.workers['train_tpb']: TheProgressBarLogger = \
            TheProgressBarLogger(self._config.TheProgressBar.update('console_handler', self.console_file)).activate()

    def on_train_epoch_begin(self, info: Dict = None):

        number_of_iterations = info['number_of_iterations']
        epochs = info['epochs']
        epoch = info['epoch']

        self.workers['train_tpb'].set_number_epochs(epochs)
        self.workers['train_tpb'].reset(number_of_iterations)
        self.workers['train_tpb'].update(0, {'epoch': epoch})

    def on_batch_end(self, info: Dict = None):

        batch_size = info.pop('batch_size')
        self.workers['train_tpb'].update(batch_size, info)

    def on_end(self, info: Dict = None):

        self.workers['train_tpb'].close()
