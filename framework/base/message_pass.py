import multiprocessing.connection
import queue
import random
import threading
from dataclasses import dataclass


class MessagePasser:
    """Trait implementing message internal passing interface."""

    class Error:
        """Messages representing an error."""

        class BaseError(BaseException):
            pass

    class Message:
        """Messages."""

        class BaseMessage:
            pass

        @dataclass
        class AskMessage:
            """Message wrapper for the ask pattern."""
            pipe_write: multiprocessing.connection.Connection
            message: 'MessagePasser.Message.BaseMessage'

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

    def __init__(self):
        """Initializer."""

        # make a queue for the internal messages and a thread to run it
        self._message_pass_queue: queue.Queue = queue.Queue()
        self._message_pass_thread: threading.Thread = \
            threading.Thread(
                name=f'{self.__class__.__name__}_message_processor_{random.randint(1, 1000)}',  # come up with rand name
                target=self._receive,
                args=(),
                daemon=True
            )

    def _start_message_passing(self) -> None:
        """Starts the whole message passing listening but spawning a new thread."""

        self._message_pass_thread.start()

    def _receive(self) -> None:
        """Main method. Receives the messages sent and acts upon them."""

        while True:

            # get the command
            message: MessagePasser.Message.BaseMessage = self._message_pass_queue.get()

            # process the message
            result: MessagePasser.ReceiveResults.BaseReceiveResult = self._receive_message(message)

            if isinstance(result, self.ReceiveResults.SameReceive):
                pass
            elif isinstance(result, self.ReceiveResults.EndReceive):
                break
            else:
                raise RuntimeError(f"received unknown receive behavior of '{result}' of type '{type(result)}'")

    def _receive_message(self, message: Message.BaseMessage) -> ReceiveResults.BaseReceiveResult:
        """
        Processes the given message and acts upon it.

        Parameters
        ----------
        message : Message.BaseMessage
            received message

        Returns
        -------
        ReceiveResults.BaseReceiveResult
            subclass of BaseReceiveResult indicating the next action

        """

        raise NotImplementedError

    def _send_message_to_self(self, message: Message.BaseMessage) -> None:
        """
        Sends the message to self for later processing.

        Parameters
        ----------
        message : Message.BaseMessage
            the message to be sent for later processing

        """

        self._message_pass_queue.put(message)

    def _ask_message_from_self(self, message: Message.BaseMessage) -> Message.BaseMessage:
        """
        Sends the message to self and wait for its reply.

        Parameters
        ----------
        message : Message.BaseMessage
            the message to be sent for later processing

        Returns
        -------
        Message.BaseMessage
            the result

        """

        # make a pipe for receiving the result
        r, w = multiprocessing.Pipe(duplex=False)

        # wrap the message in an ask class and send it
        self._message_pass_queue.put(
            self.Message.AskMessage(
                message=message,
                pipe_write=w,
            )
        )

        return r.recv()

