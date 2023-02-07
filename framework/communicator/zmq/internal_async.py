import asyncio
import json
from typing import Union, Awaitable, Tuple
from typing import Union, Any, Optional, List, Callable, Dict
import zmq
import zmq.asyncio
from tabaluga.framework.base.base import BaseWorker
from tabaluga.framework.util.config import ConfigParser
from tabaluga.framework.util.data_muncher import DataMuncher
from tabaluga.framework.util.data_muncher import UPDATE_MODIFIERS as UM, UPDATE_OPERATIONS as UO, UPDATE_CONDITIONALS as UC
from tabaluga.framework.util.data_muncher import FILTER_OPERATIONS as FO, FILTER_MODIFIERS as FM
from tabaluga.framework.asyncer.asyncer import asyncer
from tabaluga.framework.util.result import Result, Err, Ok
import functools

_ASYNCER_NAME = "_zmq_internal"
_XPUB_ADDR = "inproc://xpub"
_XSUB_ADDR = "inproc://xsub"


async def _sub_async(sub):
    """Coroutine to receive message"""

    msg = await sub.recv_multipart()
    msg = functools.reduce(lambda x, y: x + y, msg)

    # get index of the delimiter
    idx = msg.index(b' ')

    # find the topic
    recv_topic: str = (msg[:idx]).decode('utf-8')

    # extract the data
    recv_data: bytes = msg[(idx + 1):]

    return recv_topic, recv_data


class ZMQInternalAsyncPubSub(BaseWorker):
    """Class to hold and represent in-process Async ZeroMQ pub-sub communication."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # create the event loop
        res = asyncer.asyncer.create_event_loop(_ASYNCER_NAME)
        if res.is_err():
            raise RuntimeError(f"cannot make async loop of name {_ASYNCER_NAME}")

        # make the pub sub communicator
        self.zmq_context, self.xpub, self.xsub = self._start_zmq_broker()

        # bookkeeping for universal variables
        self._pub_universal = self.get_pub()

    def _start_zmq_broker(self) -> (zmq.asyncio.Context, zmq.asyncio.Socket, zmq.asyncio.Socket):
        """
        Starts the zmq pub sub broker and returns the results

        Returns
        -------
        zmq.asyncio.Context, zmq.asyncio.Socket, zmq.asyncio.Socket

        """

        context = zmq.asyncio.Context()
        xpub = context.socket(zmq.XPUB)
        xsub = context.socket(zmq.XSUB)
        xpub.bind(_XPUB_ADDR)
        xsub.bind(_XSUB_ADDR)
        import threading
        thread = threading.Thread(
            name='tabaluga-zmq-async-proxy',
            target=zmq.proxy,
            args=(xsub, xpub),
            daemon=True,
        )
        thread.start()

        self._log.info("zmq internal async server started")

        return context, xpub, xsub

    def get_pub(self) -> zmq.asyncio.Socket:
        """
        Makes a pub handle and returns it.

        Returns
        -------
        zmq.asyncio.Socket
            pub

        """

        pub = self.zmq_context.socket(zmq.PUB)
        pub.connect(_XSUB_ADDR)

        return pub

    def get_sub(self, topic: str) -> zmq.asyncio.Socket:
        """
        Makes a sub handle and returns it.

        Parameters
        ----------
        topic : str
            topic to use for subscribing

        Returns
        -------
        zmq.asyncio.Socket
            sub

        """

        sub = self.zmq_context.socket(zmq.SUB)
        sub.connect(_XPUB_ADDR)
        sub.setsockopt(zmq.SUBSCRIBE, topic.encode())

        return sub

    def get_pubsub(self, topic: str) -> (zmq.asyncio.Socket, zmq.asyncio.Socket):
        """
        Makes a pub sub handle and returns it.

        Parameters
        ----------
        topic : str
            topic to use for subscribing

        Returns
        -------
        zmq.asyncio.Socket, zmq.asyncio.Socket
            pub, sub

        """

        pub = self.get_pub()
        sub = self.get_sub(topic=topic)

        return pub, sub

    async def _pub_async(self, msg: bytes):
        """Coroutine to send message"""

        await self._pub_universal.send_multipart(msg)

    def pub(
            self,
            topic: str,
            data: Union[bytes, str, Dict]
    ) -> Result[asyncio.events.Handle, BaseException]:
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

        async def x():
            await self._pub_universal.send_multipart([f"{topic} ".encode('utf-8') + data])
            print("sent")

        res = asyncer.asyncer.add_coroutine(
            x(),
            event_loop_name=_ASYNCER_NAME,
        )

        # res = asyncer.asyncer.add_callback(
        #     callback=functools.partial(self._pub_universal.send_multipart, f"{topic} ".encode('utf-8') + data),
        #     event_loop_name=_ASYNCER_NAME,
        # )

        return Ok(res)

    async def sub(self, topic: str) -> Result[Tuple[str, bytes], Exception]:
        """
        Receives a single message from the specified topic and returns it.

        Parameters
        ----------
        topic : str
            topic to subscribe to

        Returns
        -------
        Result[Tuple[str, bytes], Exception]
            topic, data

        """

        # received the data
        sub = self.get_sub(topic)

        recv_topic, recv_data = await _sub_async(sub)

        return Ok((recv_topic, recv_data))

    def get_receiver(
            self,
            topic: str,
            event_loop_name: str,
            ensure_event_loop_uniqueness: bool = False
    ) -> Result['ZMQInternalAsyncSub', Exception]:
        """
        Returns a populated instance of ZMQInternalSub to be used to continuously receive data.

        Parameters
        ----------
        topic : str
            topic to receive data on
        event_loop_name : str
            event loop name to use for async
        ensure_event_loop_uniqueness : bool
            whether to make sure that the event loop is not created before

        Returns
        -------
        Result['ZMQInternalAsyncSub', Exception]

        """

        config = ConfigParser({
            "event_loop_name": event_loop_name,
            "ensure_event_loop_uniqueness": ensure_event_loop_uniqueness,
        })

        res = Result.from_func(
            ZMQInternalAsyncSub,
            sub=self.get_sub(topic),
            config=config,
        )
        return res


class ZMQInternalAsyncSub(BaseWorker):

    def __init__(
            self,
            sub: zmq.asyncio.Socket,
            config: ConfigParser = None
    ):

        super().__init__(config)

        # store the subscriber socket
        self.sub: zmq.asyncio.Socket = sub

        # make the event loop
        self._event_loop_name = self._config.get_value_option("event_loop_name").expect("event_loop_name not provided")
        elo = asyncer.asyncer.get_event_loop_option(self._event_loop_name)
        if elo.is_defined() and self._config.get_or_else("ensure_event_loop_uniqueness", False):
            raise RuntimeError(f"event loop {self._event_loop_name} exists")
        if elo.is_empty():
            asyncer.asyncer.create_event_loop(self._event_loop_name)

        # list of callbacks to be called upon receiving a message
        self.callbacks: List[Callable[[str, bytes], None]] = []

        # flag to keep receiving
        self.receive_on = True

    def register_callback(self, callback: Callable[[str, bytes], None]) -> 'ZMQInternalAsyncSub':
        """
        Registers callback to be called upon receiving a message

        Parameters
        ----------
        callback : Callable[[str, bytes], None]
            callback that should input the topic and the data

        Returns
        -------
        ZMQInternalAsyncSub
            self
        """

        self.callbacks.append(callback)

        return self

    def register_callbacks(self, callbacks: List[Callable[[str, bytes], None]]) -> 'ZMQInternalAsyncSub':
        """
        Registers callback to be called upon receiving a message

        Parameters
        ----------
        callbacks : List[Callable[[str, bytes], None]]
            callback that should input the topic and the data

        Returns
        -------
        ZMQInternalAsyncSub
            self
        """

        for callback in callbacks:
            self.register_callback(callback)

        return self

    def stop_receive(self) -> 'ZMQInternalAsyncSub':
        """Stops receiving from the sub."""

        self.receive_on = False

        return self

    async def _receive(self):

        poller = zmq.asyncio.Poller()
        poller.register(self.sub)

        while self.receive_on:

            # probe for the data
            socks = dict(await poller.poll(timeout=1000))
            if self.sub not in socks:
                continue

            recv_topic, recv_data = await _sub_async(self.sub)

            for callback in self.callbacks:
                callback(recv_topic, recv_data)

    def receive(self) -> Result[asyncio.Task, ValueError]:
        """Starts receiving."""

        return asyncer.asyncer.add_coroutine(
            coroutine=self._receive(),
            event_loop_name=self._event_loop_name,
        )


def init_with_config(config: ConfigParser) -> ZMQInternalAsyncPubSub:
    return ZMQInternalAsyncPubSub(config)


class _ZMQInternalAsyncPubSubGlobal:
    """
    Wrapper class around a zmq global variable.

    This class helps with zmq connector initialization on the first demand.
    """

    def __init__(self):

        # a placeholder for the global zmq instance
        self._zmq_global: Optional[ZMQInternalAsyncPubSub] = None

    def _create_instance(self) -> None:
        """Creates the zmq instance."""

        from .. import config

        self._zmq_global = init_with_config(config.zmq_config or ConfigParser({}))

    @property
    def zmq(self) -> ZMQInternalAsyncPubSub:
        """
        Returns the zmq instance.

        If the instance is not yet made, this will make it.

        Returns
        -------
        ZMQInternalAsyncPubSub
        """

        # if not made, make it
        if self._zmq_global is None:
            self._create_instance()

        return self._zmq_global


# this is an instance that everyone can use
zmq_internal_async_communicator: _ZMQInternalAsyncPubSubGlobal = _ZMQInternalAsyncPubSubGlobal()
