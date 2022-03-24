import time
from dataclasses import dataclass
from ..asyncer.asyncer import asyncer
from .base import ConfigReader, Logger
from ..util.config import ConfigParser
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable, Any
import janus
from ..util.result import Result

# type of message we accept
MessageType = TypeVar("MessageType")
MessagePasserBaseSubclass = TypeVar("MessagePasserBaseSubclass", bound="MessagePasserBase")

# constants
MESSAGE_PASSER_CONFIG_PREFIX = "_message_passer"


class ReceiveResults:
    """class to define the next behavior of the messaging system."""

    class BaseReceiveResult:
        pass

    class SameReceive(BaseReceiveResult):
        """Indicating the same behavior for receiving"""

        pass
    same_receive = SameReceive()

    class EndReceive(BaseReceiveResult):
        """Indicating end of receiving"""

        pass
    end_receive = EndReceive()

    class DestructReceive(BaseReceiveResult):
        """Indicating end of everything, the destructor"""

        pass
    destruct_receive = DestructReceive()


@dataclass
class _PanicReceive(ReceiveResults.BaseReceiveResult):
    """Used when the calling message receiver panics."""
    error: BaseException


class MessagePasserBase(Generic[MessageType], Logger, ConfigReader, ABC):
    """Abstract parent trait implementing message internal passing interface."""

    def __init__(self, config: ConfigParser = None):
        """Initializer."""

        ConfigReader.__init__(self, config)
        Logger.__init__(self, self._config)

        self._event_loop_name = \
            self._config.get_or_else(f"{MESSAGE_PASSER_CONFIG_PREFIX}.event_loop_name", "message_passer")

        # ensure we have the async event loop
        if asyncer.asyncer.get_event_loop_option(
                self._event_loop_name
        ).is_empty() and \
            asyncer.asyncer.create_event_loop(
            self._event_loop_name
        ).is_err():
            self._log.error("could not create event loop for message passing.")
            raise RuntimeError("could not create event loop for message passing.")

        async def make_janus_queue():
            """Helper function for creating a janus queue in the desired event loop."""
            return janus.Queue()

        # this is because of the stupid janus that only accepts current running loop as its event loop
        # block until we have the queue available
        res = asyncer.asyncer.add_coroutine(
            coroutine=make_janus_queue(),
            event_loop_name=
            self._event_loop_name
        )
        if res.is_err():
            self._log.error("could not create queue for event loop for message passing.")
            raise RuntimeError("could not create queue for event loop for message passing.")
        i = 0
        while not res.get().done() and i < 5000:
            i += 1
            time.sleep(.001)
        if i == 5000:
            self._log.error("waited too log to obtain queue for message passing")
            raise RuntimeError("waited too log to obtain queue for message passing")
        self._message_pass_queue: janus.Queue[MessageType] = res.get().result()
        self._message_pass_queue_sync: janus.SyncQueue = self._message_pass_queue.sync_q
        self._message_pass_queue_async: janus.AsyncQueue = self._message_pass_queue.async_q

    @abstractmethod
    async def _receive(self) -> None:
        """Main method. Receives the messages sent and acts upon them."""

        raise NotImplementedError

    def tell(self, message: MessageType) -> Result:
        """
        Sends the message to self for later processing.

        Parameters
        ----------
        message : MessageType
            the message to be sent for later processing

        Returns
        -------
        Result
            whether the operation was successful

        """

        return Result.from_func(self._message_pass_queue_sync.put, message)


class MessagePasser(MessagePasserBase[MessageType], ABC):
    """Trait implementing message internal passing interface."""

    def __init__(self, config: ConfigParser = None):
        """Initializer."""

        super().__init__(config)

    def __del__(self):

        self.tell(ReceiveResults.destruct_receive)

    def close(self):
        """Destruct everything and close."""

        self.__del__()

    def _start_message_passing(self) -> None:
        """Starts the whole message passing listening in an async way."""

        asyncer.asyncer.add_coroutine(
            coroutine=self._receive(),
            event_loop_name=
            self._event_loop_name,
        )

    async def _receive(self) -> None:
        """Main method. Receives the messages sent and acts upon them."""

        while True:

            # get the command
            try:
                message: MessageType = await self._message_pass_queue_async.get()
            except BaseException as e:
                self._log.debug(f"failed getting message from queue with error of '{e}'")
                break

            try:
                # process the message
                result: ReceiveResults.BaseReceiveResult = await self._receive_message(message)
            except BaseException as e:
                result: ReceiveResults.BaseReceiveResult = _PanicReceive(error=e)

            if isinstance(result, ReceiveResults.SameReceive):
                # do nothing and go to the next fetching
                pass
            elif isinstance(result, ReceiveResults.EndReceive):
                break
            elif isinstance(result, ReceiveResults.DestructReceive):
                break
            elif isinstance(result, _PanicReceive):
                self._log.warning(f"receiver panicked with error of '{result.error}'")
                break
            else:
                raise RuntimeError(f"received unknown receive behavior of '{result}' of type '{type(result)}'")

    @abstractmethod
    async def _receive_message(self, message: MessageType) -> ReceiveResults.BaseReceiveResult:
        """
        Processes the given message and acts upon it.

        Parameters
        ----------
        message : MessageType
            received message

        Returns
        -------
        ReceiveResults.BaseReceiveResult
            subclass of BaseReceiveResult indicating the next action

        """

        raise NotImplementedError

    def ask(self, message_factory: Callable[[MessagePasserBaseSubclass], MessageType]) -> Result[Any, BaseException]:
        """
        Sends the message to self and wait for its reply.

        This method should NOT be called from the thread that is listening to messages.

        Parameters
        ----------
        message_factory : Callable[['MessagePasserBase'], MessageType]
            the message factory. it should be a function that takes a reference to a MessagePasser and returns the message

        Returns
        -------
        Result
            the answer wrapped in Result

        """

        # check we are not the same method as the listener
        # if self._message_pass_thread_ident is not None and threading.get_ident() == self._message_pass_thread_ident:
        #     raise RuntimeError("ask is called from the same thread as listener.")

        return _Asker(self).ask(message_factory)


def ask(
        message_passer: MessagePasser[MessageType],
        message_factory: Callable[[MessagePasserBaseSubclass], MessageType]
) -> Any:
    """
    Asks the message passer a message and blocks until gets back the result.

    Parameters
    ----------
    message_passer : MessagePasser
        the message passer to ask
    message_factory :  Callable[['MessagePasserBase'], MessageType]
        the message factory. it should be a function that takes a reference to a MessagePasser and returns the message

    Returns
    -------
    Any
        the result

    """

    return _Asker(message_passer).ask(message_factory)


MessageTypeAsk = TypeVar("MessageTypeAsk")


class _Asker(MessagePasserBase[Any]):

    def __init__(self, reference: MessagePasser[MessageTypeAsk], config: ConfigParser = None):
        """
        Initializer.

        Parameters
        ----------
        reference : MessagePasser[MessageTypeAsk]
            the message passer to ask from
        """

        super().__init__(config)

        # keep the message passer for asking
        self.ask_reference: MessagePasser[MessageTypeAsk] = reference

    def _receive(self) -> Any:
        """Main method. Receives the messages and returns it."""

        # get only a single message and return it

        # get the message
        # notice that we get an Any message because we do not know the type of the sender's message
        message: Any = self._message_pass_queue_sync.get()

        return message

    def ask(self, message_factory: Callable[['_Asker'], MessageType]) -> Result[Any, BaseException]:
        """
        Asks for a value, waits for it, and returns it.

        Parameters
        ----------
        message_factory : Callable[['MessagePasserBase'], MessageType]
        the message factory. it should be a function that takes a reference to a MessagePasser and returns the message

        Returns
        -------
        Result
           the answer wrapped in Result

        """

        # first tell the message and make it send to self
        self.ask_reference.tell(message_factory(self))

        # now, wait for the result
        result = Result.from_func(self._receive)

        return result
