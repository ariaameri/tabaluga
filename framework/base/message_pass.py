import queue
import random
import threading
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable, Any

# type of message we accept
MessageType = TypeVar("MessageType")
MessagePasserBaseSubclass = TypeVar("MessagePasserBaseSubclass", bound="MessagePasserBase")


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


class MessagePasserBase(Generic[MessageType], ABC):
    """Abstract parent trait implementing message internal passing interface."""

    def __init__(self):
        """Initializer."""

        # make a queue for message passing
        self._message_pass_queue: queue.Queue[MessageType] = queue.Queue()

    @abstractmethod
    def _receive(self) -> None:
        """Main method. Receives the messages sent and acts upon them."""

        raise NotImplementedError


class MessagePasser(MessagePasserBase[MessageType], ABC):
    """Trait implementing message internal passing interface."""

    def __init__(self):
        """Initializer."""

        super().__init__()

        # make a thread to run it
        self._message_pass_thread: Optional[threading.Thread] = None
        self._message_pass_thread_ident: Optional[int] = None

    def _start_message_passing(self, blocking: bool = False) -> None:
        """
        Starts the whole message passing listening but spawning a new thread.

        Parameters
        ----------
        blocking : bool, optional
            whether to block or not by starting a new thread for listening

        """

        if blocking is False:
            self._message_pass_thread = \
                threading.Thread(
                    # come up with random name
                    name=f'{self.__class__.__name__}_message_processor_{random.randint(1, 1000)}',
                    target=self._receive,
                    args=(),
                    daemon=True
                )
            self._message_pass_thread.start()
            self._message_pass_thread_ident = self._message_pass_thread.ident
        else:
            self._receive()

    def _receive(self) -> None:
        """Main method. Receives the messages sent and acts upon them."""

        while True:

            # get the command
            message: MessageType = self._message_pass_queue.get()

            # process the message
            result: ReceiveResults.BaseReceiveResult = self._receive_message(message)

            if isinstance(result, ReceiveResults.SameReceive):
                # do nothing and go to the next fetching
                pass
            elif isinstance(result, ReceiveResults.EndReceive):
                break
            else:
                raise RuntimeError(f"received unknown receive behavior of '{result}' of type '{type(result)}'")

    @abstractmethod
    def _receive_message(self, message: MessageType) -> ReceiveResults.BaseReceiveResult:
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

    def tell(self, message: MessageType) -> None:
        """
        Sends the message to self for later processing.

        Parameters
        ----------
        message : MessageType
            the message to be sent for later processing

        """

        self._message_pass_queue.put(message)

    def ask(self, message_factory: Callable[[MessagePasserBaseSubclass], MessageType]) -> Any:
        """
        Sends the message to self and wait for its reply.

        This method should NOT be called from the thread that is listening to messages.

        Parameters
        ----------
        message_factory : Callable[['MessagePasserBase'], MessageType]
            the message factory. it should be a function that takes a reference to a MessagePasser and returns the message

        Returns
        -------
        Any
            the result

        """

        # check we are not the same method as the listener
        if self._message_pass_thread_ident is not None and threading.get_ident() == self._message_pass_thread_ident:
            raise RuntimeError("ask is called from the same thread as listener.")

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

    def __init__(self, reference: MessagePasser[MessageTypeAsk]):
        """
        Initializer.

        Parameters
        ----------
        reference : MessagePasser[MessageTypeAsk]
            the message passer to ask from
        """

        super().__init__()

        # keep the message passer for asking
        self.ask_reference: MessagePasser[MessageTypeAsk] = reference

    def _receive(self) -> Any:
        """Main method. Receives the messages and returns it."""

        # get only a single message and return it

        # get the message
        # notice that we get an Any message because we do not know the type of the sender's message
        message: Any = self._message_pass_queue.get()

        return message

    def ask(self, message_factory: Callable[['_Asker'], MessageType]) -> Any:
        """
        Asks for a value, waits for it, and returns it.

        Parameters
        ----------
        message_factory : Callable[['MessagePasserBase'], MessageType]
        the message factory. it should be a function that takes a reference to a MessagePasser and returns the message

        Returns
        -------
        Any
            the result

        """

        # first tell the message and make it send to self
        self.ask_reference.tell(message_factory(self))

        # now, wait for the result
        result = self._receive()

        return result
