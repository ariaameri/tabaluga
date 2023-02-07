from typing import Any, Dict, Union, Callable
from .base import ConfigReader, Logger
from ..asyncer.asyncer import asyncer
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..communicator.zmq.internal_async import zmq_internal_async_communicator as zmq, ZMQInternalAsyncSub
from ..util.result import Result, Ok


EVENTFUL_INTERNAL_CONFIG_PREFIX = "_eventful_internal"


class EventfulInternal(Logger, ConfigReader):
    """Eventful class. Listens to events published internally and react upon them."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # mapping from topic prefix to listener
        self.__event_loop_name = \
            self._config.get_or_else(f"{EVENTFUL_INTERNAL_CONFIG_PREFIX}.event_loop_name", self.__class__.__name__)
        self.__event_listen_info = DataMuncher()
        self.__event_pub = zmq.zmq.get_pub()

        # make sure the event loop exists
        if asyncer.asyncer.get_event_loop_option(self.__event_loop_name).is_empty():
            if (res := asyncer.asyncer.create_event_loop(self.__event_loop_name)).is_err():
                self._log.error(
                    f"error making the event loop of name {self.__event_loop_name} with error of '{res.get_err()}'"
                )
                raise res.get_err()

    def _add_event_callback(
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
        if (res := self.__event_listen_info.get_value_option(topic_prefix)).is_empty():
            # make a receiver
            res = zmq.zmq.get_receiver(topic_prefix, self.__event_loop_name)
            if res.is_err():
                return res
            else:
                receiver: ZMQInternalAsyncSub = res.get()
                # listen
                res = receiver.receive()
                if res.is_err():
                    return res
                # store
                self.__event_listen_info = self.__event_listen_info.update({}, {topic_prefix: receiver})
        else:
            receiver: ZMQInternalAsyncSub = res.get()

        receiver.register_callback(callback)

        return Ok(None)

    def _publish_event(self, topic: str, data: Any) -> Result:
        """
        Publishes an event.

        Parameters
        ----------
        topic : str
            topic
        data : Any
            data to publish

        Returns
        -------
        Result
        """

        return zmq.zmq.pub_with_pub(topic, data, self.__event_pub, self.__event_loop_name)
