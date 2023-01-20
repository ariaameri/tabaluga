import json
from typing import Union, Any, Optional, List, Callable, Dict
from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.data_muncher import UPDATE_MODIFIERS as UM, UPDATE_OPERATIONS as UO, UPDATE_CONDITIONALS as UC
from ..util.data_muncher import FILTER_OPERATIONS as FO, FILTER_MODIFIERS as FM
import zmq

from ..util.result import Result, Err


class ZMQInternalPubSub(BaseWorker):
    """Class to hold and represent in-process ZeroMQ pub-sub communication."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # make the pub sub communicator
        self.zmq_context, self.xpub, self.xsub = self._start_zmq_broker()

        # bookkeeping for universal variables
        self._pub_universal = self.get_pub()

    def _start_zmq_broker(self) -> (zmq.Context, zmq.Socket, zmq.Socket):
        """
        Starts the zmq pub sub broker and returns the results

        Returns
        -------
        zmq.Context, zmq.Socket, zmq.Socket

        """

        context = zmq.Context()
        xpub = context.socket(zmq.XPUB)
        xsub = context.socket(zmq.XSUB)
        xpub.bind("inproc://xpub")
        xsub.bind("inproc://xsub")
        import threading
        thread = threading.Thread(
            name='tabaluga-zmq-proxy',
            target=zmq.proxy,
            args=(xsub, xpub),
            daemon=True,
        )
        thread.start()
        self._log.info("zmq internal server started")

        return context, xpub, xsub

    def get_pub(self) -> zmq.Socket:
        """
        Makes a pub handle and returns it.

        Returns
        -------
        zmq.Socket
            pub

        """

        pub = self.zmq_context.socket(zmq.PUB)
        pub.connect("inproc://xsub")

        return pub

    def get_sub(self, topic: str) -> zmq.Socket:
        """
        Makes a sub handle and returns it.

        Parameters
        ----------
        topic : str
            topic to use for subscribing

        Returns
        -------
        zmq.Socket
            sub

        """

        sub = self.zmq_context.socket(zmq.SUB)
        sub.connect("inproc://xpub")
        sub.setsockopt(zmq.SUBSCRIBE, topic.encode())

        return sub

    def get_pubsub(self, topic: str) -> (zmq.Socket, zmq.Socket):
        """
        Makes a pub sub handle and returns it.

        Parameters
        ----------
        topic : str
            topic to use for subscribing

        Returns
        -------
        zmq.Socket, zmq.Socket
            pub, sub

        """

        pub = self.get_pub()
        sub = self.get_sub(topic=topic)

        return pub, sub

    def pub(self, topic: str, data: Union[bytes, str, Dict]) -> Result[Optional[zmq.MessageTracker], BaseException]:
        """
        Publishes the data to the topic.

        Parameters
        ----------
        topic : str
            topic to publish to
        data : Union[bytes, str, Dict]
            data in form of str, dict, or bytes

        Returns
        -------
        Result

        """

        if isinstance(data, dict):
            res = Result.from_func(json.dumps, data)
            if res.is_err():
                return res
            else:
                data = res.get()
        if isinstance(data, str):
            res = Result.from_func(data.encode, 'utf8')
            if res.is_err():
                return res
            else:
                data = res.get()
        elif not isinstance(data, bytes):
            return Err(ValueError("input value type not accepted"))

        return Result.from_func(self._pub_universal.send, f"{topic} ".encode('utf-8') + data)

    def sub(self, topic: str) -> (str, bytes):
        """
        Receives a single message from the specified topic and returns it.

        Parameters
        ----------
        topic : str
            topic to subscribe to

        Returns
        -------
        (str, bytes)
            topic, data

        """

        # received the data
        sub = self.get_sub(topic)
        data = sub.recv()

        # get index of the delimiter
        idx = data.index(b' ')

        # find the topic
        recv_topic: str = (data[:idx]).decode('utf-8')

        # extract the data
        recv_data: bytes = data[(idx+1):]

        return recv_topic, recv_data

    def get_receiver(self, topic: str) -> 'ZMQInternalSub':
        """
        Returns a populated instance of ZMQInternalSub to be used to continuously receive data.

        Parameters
        ----------
        topic : str
            topic to receive data on

        Returns
        -------
        ZMQInternalSub

        """

        return ZMQInternalSub(self.get_sub(topic))


class ZMQInternalSub(BaseWorker):

    def __init__(self, sub: zmq.Socket, config: ConfigParser = None):

        super().__init__(config)

        # store the subscriber socket
        self.sub: zmq.Socket = sub

        # list of callbacks to be called upon receiving a message
        self.callbacks: List[Callable[[str, bytes], None]] = []

        # flag to keep receiving
        self.receive_on = True

    def register_callback(self, callback: Callable[[str, bytes], None]) -> 'ZMQInternalSub':
        """
        Registers callback to be called upon receiving a message

        Parameters
        ----------
        callback : Callable[[str, bytes], None]
            callback that should input the topic and the data

        Returns
        -------
        ZMQInternalSub
            self
        """

        self.callbacks.append(callback)

        return self

    def stop_receive(self):
        """Stops receiving from the sub."""

        self.receive_on = False

    def receive(self) -> None:
        """Starts receiving."""

        poller = zmq.Poller()
        poller.register(self.sub)

        while self.receive_on:

            # probe for the data
            socks = dict(poller.poll(timeout=1000))

            if self.sub not in socks:
                continue

            # received the data
            data = self.sub.recv()

            # get index of the delimiter
            idx = data.index(b' ')

            # find the topic
            recv_topic: str = (data[:idx]).decode('utf-8')

            # extract the data
            recv_data: bytes = data[(idx + 1):]

            for callback in self.callbacks:
                callback(recv_topic, recv_data)


def init_with_config(config: ConfigParser) -> ZMQInternalPubSub:
    return ZMQInternalPubSub(config)


class _ZMQInternalPubSubGlobal:
    """
    Wrapper class around a zmq global variable.

    This class helps with zmq connector initialization on the first demand.
    """

    def __init__(self):

        # a placeholder for the global zmq instance
        self._zmq_global: Optional[ZMQInternalPubSub] = None

    def _create_instance(self) -> None:
        """Creates the zmq instance."""

        from . import config

        self._zmq_global = init_with_config(config.zmq_config or ConfigParser({}))

    @property
    def zmq(self) -> ZMQInternalPubSub:
        """
        Returns the zmq instance.

        If the instance is not yet made, this will make it.

        Returns
        -------
        ZMQInternalPubSub
        """

        # if not made, make it
        if self._zmq_global is None:
            self._create_instance()

        return self._zmq_global


# this is an instance that everyone can use
zmq_communicator: _ZMQInternalPubSubGlobal = _ZMQInternalPubSubGlobal()
