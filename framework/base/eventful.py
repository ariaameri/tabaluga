from typing import Any, Dict, Union, Callable
from .base import ConfigReader, Logger
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..communicator.zmq.internal_async import zmq_internal_async_communicator as zmq, ZMQInternalAsyncSub
from ..util.result import Result, Ok


class EventfulInternal(Logger, ConfigReader):
    """Eventful class. Listens to events published internally and react upon them."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # mapping from topic prefix to listener
        self._listen_info = DataMuncher()

    def _add_callback(
            self,
            topic_prefix: str,
            callback: Callable[[str, Union[bytes, Dict]], Any],
    ) -> Result[None, Exception]:
        """
        Adds callback for the event.

        Parameters
        ----------
        topic_prefix : str
            the prefix of the topic to listen for

        callback : Callable[[str, Union[bytes, Dict]], Any]
            callback to call upon reception of a message

        Returns
        -------
        Result[None, Exception]

        """

        # get the receiver
        if (res := self._listen_info.get_value_option(topic_prefix)).is_empty():
            # make a receiver
            res = zmq.zmq.get_receiver(topic_prefix, self.__class__.__name__)
            if res.is_err():
                return res
            else:
                receiver: ZMQInternalAsyncSub = res.get()
                # listen
                res = receiver.receive()
                if res.is_err():
                    return res
                # store
                self._listen_info = self._listen_info.update({}, {topic_prefix: receiver})
        else:
            receiver: ZMQInternalAsyncSub = res.get()

        receiver.register_callback(callback)

        return Ok(None)
