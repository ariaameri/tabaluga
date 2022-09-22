from typing import List
from jointstemplant.util.util import OnAndEnabled
from tabaluga.framework.util.config import ConfigParser
from tabaluga.framework.util.data_muncher import DataMuncher


class BatchSizeChanger(OnAndEnabled):

    def __init__(self, config: ConfigParser = None):

        OnAndEnabled.__init__(self, config)

        if not self._is_enabled():
            return

        # multiply the batch size by gamma at each step
        steps_info: List = self._config.get_value_option('steps').expect('steps not provided')
        self._steps_info: DataMuncher = DataMuncher({
            str(
                ConfigParser(step_info).get_value_option('at')
                .expect('at not provided, should be epoch number')
            ): step_info
            for step_info
            in steps_info
        })

    def _calculate_new_batch_size(self, config: DataMuncher, batch_size_current: int) -> int:

        # go over the config in order
        batch_size = config\
            .get_value_option('set').filter(lambda x: x is not None).map(lambda x: int(x))\
            .or_else(
            config\
                .get_value_option('mult').filter(lambda x: x is not None).map(lambda x: int(batch_size_current * x))
            )\
            .or_else(
            config \
                .get_value_option('add').filter(lambda x: x is not None).map(lambda x: int(batch_size_current + x))
            ).get()

        return batch_size

    def calculate(self, info: DataMuncher = DataMuncher()) -> int:
        """

        Parameters
        ----------
        info : DataMuncher
            of the form
                {
                    `epoch`: epoch number
                    `batch_size`: current batch size
                }

        Returns
        -------
        int
            new batch size

        """

        epoch = info.get('epoch')
        batch_size = batch_size_current = info.get('batch_size_current')

        if not self._is_enabled():
            return batch_size

        if self._steps_info.contains_key(str(epoch)):
            batch_size = self._calculate_new_batch_size(self._steps_info.get(str(epoch)), batch_size_current)

        return batch_size
