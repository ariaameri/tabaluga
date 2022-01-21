import multiprocessing.connection
import queue
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional

# type of message we accept
MessageType = TypeVar("MessageType")


class ReceiveResults:
    """class to define the next behavior of the messaging system."""

    class BaseReceiveResult:
        pass

    class SameReceive(BaseReceiveResult):
        """Indicating the same behavior for receiving"""

        pass
    sameReceive = SameReceive()

    class EndReceive(BaseReceiveResult):
        """Indicating end of receiving"""

        pass
    endReceive = EndReceive()


class MessagePasser(Generic[MessageType], ABC):
    """Trait implementing message internal passing interface."""

    def __init__(self):
        """Initializer."""

        # make a queue for the internal messages and a thread to run it
        self._message_pass_queue: queue.Queue = queue.Queue()
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

    def ask(self, message: MessageType) -> MessageType:
        """
        Sends the message to self and wait for its reply.

        This method should NOT be called from the thread that is listening to messages.

        Parameters
        ----------
        message : MessageType
            the message to be sent for later processing

        Returns
        -------
        MessageType
            the result

        """

        # check we are not the same method as the listener
        if self._message_pass_thread_ident is not None and threading.get_ident() == self._message_pass_thread_ident:
            raise RuntimeError("ask is called from the same thread as listener.")

        return _Asker(self).ask(message)


class _Asker(Generic[MessageType]):

    def __init__(self, reference: MessagePasser[MessageType]):
        """
        Initializer.

        Parameters
        ----------
        reference : MessagePasser
            the message passer to ask from
        """

        # make a queue for the internal messages and a thread to run it
        self._message_pass_queue: queue.Queue = queue.Queue()
        # keep the message passer for asking
        self.ask_reference: MessagePasser[MessageType] = reference

    def _receive(self) -> MessageType:
        """Main method. Receives the messages and returns it."""

        # get only a single message and return it

        # get the message
        message: MessageType = self._message_pass_queue.get()

        return message

    def ask(self, message: MessageType) -> MessageType:

        # first tell the message
        self.ask_reference.tell(message)

        # now, wait for the result
        result = self._receive()

        return result
