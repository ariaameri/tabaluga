from nexdeepml.logger.logger import LoggerManager, Logger, TheProgressBarLogger
from nexdeepml.util.config import ConfigParser
from typing import Dict
import signal


class SampleLoggerManager(LoggerManager):

    def __init__(self, config: ConfigParser):
        super().__init__(config)

        self.create_workers()

    def create_workers(self):
        self.workers['train_tpb']: TheProgressBarLogger

    def on_train_begin(self, info: Dict = None):
        self.workers['train_tpb']: TheProgressBarLogger = \
            TheProgressBarLogger(
                self._config
                    .get_or_else('TheProgressBar', ConfigParser())
                    .update({}, {'$set': {'console_handler': self.console_file}})
            ).activate()

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

    def on_val_epoch_begin(self, info: Dict = None):

        # info should contain number_of_iterations, the epoch and train loss
        number_of_iterations = info.pop('number_of_iterations')

        self.workers['train_tpb'].reset_to_next_line(number_of_iterations)
        self.workers['train_tpb'].update(0, info)

    def on_val_batch_end(self, info: Dict = None):

        batch_size = info.pop('batch_size')
        self.workers['train_tpb'].update(batch_size, info)

    def on_end(self, info: Dict = None):
        self.workers['train_tpb'].close()

    def on_os_signal(self, info: Dict = None):
        os_signal = info['signal']

        if os_signal == signal.SIGINT or os_signal == signal.SIGTERM:
            self.workers['train_tpb'].close()

        if os_signal == signal.SIGTSTP:
            self.workers['train_tpb'].pause()

        if os_signal == signal.SIGCONT:
            self.workers['train_tpb'].resume()

        super().on_os_signal(info)


# class SampleTheProgressBarLoggerManager(LoggerManager):
#
#     def __init__(self, config: ConfigParser):
#
#         super().__init__(config)
#
#         self.create_workers()
#
#     def create_workers(self):
#
#         self.workers['train_tpb']: TheProgressBarLogger
#
#     def on_train_begin(self, info: Dict = None):
#
#         self.workers['train_tpb']: TheProgressBarLogger = \
#             TheProgressBarLogger(
#                 self._config
#                 .get_or_else('TheProgressBar', ConfigParser())
#                 .update('console_handler', self.console_file)
#             ).activate()
#
#     def on_train_epoch_begin(self, info: Dict = None):
#
#         number_of_iterations = info['number_of_iterations']
#         epochs = info['epochs']
#         epoch = info['epoch']
#
#         self.workers['train_tpb'].set_number_epochs(epochs)
#         self.workers['train_tpb'].reset(number_of_iterations)
#         self.workers['train_tpb'].update(0, {'epoch': epoch})
#
#     def on_batch_end(self, info: Dict = None):
#
#         batch_size = info.pop('batch_size')
#         self.workers['train_tpb'].update(batch_size, info)
#
#     def on_end(self, info: Dict = None):
#
#         self.workers['train_tpb'].close()
#
#     def on_os_signal(self, info: Dict = None):
#
#         os_signal = info['signal']
#
#         if os_signal == signal.SIGINT:
#             self.workers['train_tpb'].close()
#
#         super().on_os_signal(info)
