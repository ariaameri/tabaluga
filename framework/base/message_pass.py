import asyncio
from numbers import Number
from threading import Lock
import time
from dataclasses import dataclass
import numpy as np
from ..asyncer.asyncer import asyncer
from .base import ConfigReader, Logger, BaseWorker
from ..util.config import ConfigParser
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable, Any, Set, Tuple, List
import janus
from ..util.data_muncher import DataMuncher
from ..util.data_muncher import FILTER_OPERATIONS as FO, FILTER_MODIFIERS as FM
from ..util.data_muncher import UPDATE_OPERATIONS as UO, UPDATE_MODIFIERS as UM
from ..util.result import Result, Err, Ok

# type of message we accept
MessageType = TypeVar("MessageType")
MessagePasserBaseSubclass = TypeVar("MessagePasserBaseSubclass", bound="MessagePasserBase")

# constants
MESSAGE_PASSER_CONFIG_PREFIX = "_message_passer"

# global variables
# keep a list of all message passers so that we can in the end send a message to all to terminate
_ALL_MESSAGE_PASSERS_LOCK = Lock()
_ALL_MESSAGE_PASSERS: Set['MessagePasserBase'] = set()


class BaseMessage:
    """base class pf the message classes. All message types should extend this class."""


class TerminateMessage(BaseMessage):
    """message to send to tell the message passer to terminate"""


terminate_message = TerminateMessage


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


class MessagePasserError:
    """Struct to hold message passer errors"""

    class BaseMessagePasserError(Exception):
        """Base class for message passer errors."""
        pass

    class QueueError(BaseMessagePasserError):
        """Errors related to the queue."""
        pass

    class QueueNotExist(QueueError):
        """Error when the queue does not exist."""
        pass

    class OperationError(BaseMessagePasserError):
        """Error for when we are not operational."""
        pass


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

        # we can have multiple queues
        self._queues_info = self._config.get_or_empty(f"{MESSAGE_PASSER_CONFIG_PREFIX}.queues")
        # for convenience have the info as array of data as well
        self._queues = DataMuncher({
            "_names": [],
            "_priorities": [],
            "_queues": [],
            "_queues_sync": [],
            "_queues_async": [],
            "queues": {},
        })
        for name in self._queues_info.get_keys():
            priority = self._queues_info.get_or_else(f"{name}.priority", 1)
            if not isinstance(priority, Number):
                self._log.error(f"for the queue {name} got priority of {priority} that is not a number")
                raise ValueError(f"wrong queue priority")
            if (message_pass_queue_res := self._make_async_queue()).is_err():
                raise message_pass_queue_res.get_err()
            queue: janus.Queue[MessageType] = message_pass_queue_res.get()
            queue_sync: janus.SyncQueue = queue.sync_q
            queue_async: janus.AsyncQueue = queue.async_q
            self._queues = self._queues.update({}, {
                f"queues.{name}.name": name,
                f"queues.{name}.priority": priority,
                f"queues.{name}.queue": queue,
                f"queues.{name}.q_sync": queue_sync,
                f"queues.{name}.q_async": queue_async,
            })
            self._queues.get_value_option("_names").for_each(lambda x: x.append(name))
            self._queues.get_value_option("_priorities").for_each(lambda x: x.append(priority))
            self._queues.get_value_option("_queues").for_each(lambda x: x.append(queue))
            self._queues.get_value_option("_queues_sync").for_each(lambda x: x.append(queue_sync))
            self._queues.get_value_option("_queues_async").for_each(lambda x: x.append(queue_async))
        # sort the arrays based on the priority
        if len(ps := self._queues.get("_priorities")) > 1:
            self._pending_polls = {}
            sorted_idxs = np.argsort(ps)[::-1]
            self._queues = \
                self._queues\
                .update_map(
                    {FM.KEYNAME: {FO.REGEX: r'^_\w+'}},
                    lambda x: [x[idx] for idx in sorted_idxs],
                )

            # take the one with the lowest priority and make that the default queue
            self._default_queue_sync = self._queues.get("_queues_sync")[-1]

        # if we have only one queue, optimize a bit
        self._single_queue = False  # to know if we have a single queue
        if 0 <= len(self._queues_info.get_keys()) <= 1:
            self._single_queue = True
            if (message_pass_queue_res := self._make_async_queue()).is_err():
                raise message_pass_queue_res.get_err()
            self._message_pass_queue: janus.Queue[MessageType] = message_pass_queue_res.get()
            self._message_pass_queue_sync: janus.SyncQueue = self._message_pass_queue.sync_q
            self._message_pass_queue_async: janus.AsyncQueue = self._message_pass_queue.async_q
            self._default_queue_sync = self._message_pass_queue_sync

            # make a queue for storing the messages that will be processed later
        from collections import deque
        self._inbox_storage: deque[MessageType] = deque()

        # bookkeeping to know we are operational
        self._message_pass_working = True

        # add us to the list of all message passers
        with _ALL_MESSAGE_PASSERS_LOCK:
            _ALL_MESSAGE_PASSERS.add(self)

    def _make_async_queue(self) -> Result[janus.Queue[MessageType], Exception]:
        """Makes and returns an async queue, wrapped in Result."""

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
            return Err(RuntimeError("could not create queue for event loop for message passing."))
        i = 0
        while not res.get().done() and i < 5000:
            i += 1
            time.sleep(.001)
        if i == 5000:
            self._log.error("waited too log to obtain queue for message passing")
            return Err(RuntimeError("waited too log to obtain queue for message passing"))

        message_pass_queue: janus.Queue[MessageType] = res.get().result()

        return Ok(message_pass_queue)

    def _message_store(self, message: MessageType) -> None:
        """
        Stores the message provided.

        Parameters
        ----------
        message : MessageType
            the message

        """

        self._inbox_storage.append(message)

    def _resend_first_stored_message(self) -> None:
        """Resends the first stored message to self. If no message is saved, does nothing."""

        if len(self._inbox_storage) > 0:
            self.tell(self._inbox_storage.popleft())

    def _resend_stored_messages(self) -> None:
        """Resends all the stored message to self. If no messages are saved, does nothing."""

        while self._inbox_storage:
            self.tell(self._inbox_storage.popleft())

    def get_approximate_message_passer_queue_size(self) -> int:
        """Returns the approximate size of the message passing queue."""

        if self._message_pass_working is False:
            return 0

        if self._single_queue:
            size = self._message_pass_queue_async.qsize()
        else:
            size = sum(_.qsize() for _ in self._queues.get("_queues_async"))

        return size

    @abstractmethod
    async def _receive(self) -> None:
        """Main method. Receives the messages sent and acts upon them."""

        raise NotImplementedError

    def tell(self, message: MessageType, queue_name: str = None) -> Result:
        """
        Sends the message to self for later processing.

        Parameters
        ----------
        message : MessageType
            the message to be sent for later processing
        queue_name : str, optional
            whether to put this message on a special queue, when not passed, default queue is used

        Returns
        -------
        Result
            whether the operation was successful

        """

        # make sure if we are working and initialized
        if not hasattr(self, "_message_pass_working") or self._message_pass_working is False:
            return Err(MessagePasserError.OperationError("not operational"))

        if queue_name is None:
            msg_queue = self._default_queue_sync
        else:
            res = self._queues.get_value_option(f"queues.{queue_name}.q_sync")
            if res.is_empty():
                return Err(MessagePasserError.QueueNotExist(f"queue with name {queue_name} does not exist"))
            msg_queue = res.get()

        res = Result.from_func(msg_queue.put, message)
        if res.is_err():
            res = Err(MessagePasserError.QueueError(res.get_err()))

        return res

    def __terminate_message_passing__(self) -> Result:
        """sends a message to self to be terminated."""

        return self.tell(terminate_message)


class MessagePasser(MessagePasserBase[MessageType], ABC):
    """Trait implementing message internal passing interface."""

    def __init__(self, config: ConfigParser = None):
        """Initializer."""

        super().__init__(config)

        # function that will be used for getting messages
        asyncer_loop = asyncer.asyncer.get_event_loop_option(self._event_loop_name).get()
        self._message_getter = self._get_message(asyncer_loop)

    def __del__(self):

        self.tell(terminate_message)

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

    def _get_message(self, asyncer_loop: asyncio.AbstractEventLoop):
        """Helper method to create a function for getting the messages in order of priority."""

        if self._single_queue:
            # if we have a single queue, optimize a bit
            async def receiver_helper() -> List[Tuple[Number, str, MessageType]]:
                message: MessageType = await self._message_pass_queue_async.get()
                return [(1, "default", message)]

        else:
            # if we have multiple queues, sort based on priority
            async def receiver_helper() -> List[Tuple[Number, str, MessageType]]:

                # this is very stupid
                # we have to make this functionality manually
                # because there is no functionality like `select.select` for python's asyncio
                # also janus queues do not have polling method :/
                # so, here we go...

                # construct the polls that we have not processed yet and the ones that are new
                poll = [
                    asyncer_loop.create_task(q.get())
                    if idx not in self._pending_polls
                    else self._pending_polls[idx]
                    for idx, q
                    in enumerate(self._queues.get("_queues_async"))
                ]
                done, pending = await asyncio.wait(poll, return_when=asyncio.FIRST_COMPLETED)
                self._pending_polls = \
                    {
                        idx: poll[idx]
                        for idx
                        in range(len(self._queues.get("_queues_async")))
                        if poll[idx] in pending
                    }

                # perform the actions in round-robin fashion
                # note that the queues and priorities arrays are already sorted based on priorities
                # thus, the following array is sorted based on priorities
                p_res = [
                    (p, n, poll[idx].result())
                    for idx, (p, n)
                    in enumerate(zip(self._queues.get("_priorities"), self._queues.get("_names")))
                    if poll[idx] in done
                ]

                return p_res

        return receiver_helper

    async def _receive(self) -> None:
        """Main method. Receives the messages sent and acts upon them."""

        while True:

            # get the command
            try:
                priority_messages: List[Tuple[Number, str, MessageType]] = await self._message_getter()
            except BaseException as e:
                self._log.error(f"failed getting message from queue with error of '{e}'. stopping message receiving")
                break

            for priority, q_name, msg in priority_messages:

                # if it is a termination message, terminate
                if msg == terminate_message:
                    break

                try:
                    # process the message
                    result: ReceiveResults.BaseReceiveResult = await self._receive_message(msg)
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
                    self._message_pass_working = False
                    raise RuntimeError(f"received unknown receive behavior of '{result}' of type '{type(result)}'")

        # no longer operational
        self._message_pass_working = False

        # remove ourselves from the set of all message passers
        with _ALL_MESSAGE_PASSERS_LOCK:
            _ALL_MESSAGE_PASSERS.remove(self)

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

        if self._message_pass_working is False:
            return Err(MessagePasserError.OperationError("not operational"))

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

        # remove ourselves from the set of all message passers
        with _ALL_MESSAGE_PASSERS_LOCK:
            _ALL_MESSAGE_PASSERS.remove(self)

        # if it is a termination message, terminate
        if message == terminate_message:
            raise RuntimeError("received a terminate message before getting the result")

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


class _MessagePasserTerminator(BaseWorker):

    def __init__(self, config: ConfigParser):

        super().__init__(config)

        self.number_of_retries: int = int(self._config.get_or_else('number_of_retries', 1000))
        self.retry_interval: int = int(self._config.get_or_else('retry_interval', 15))

    @staticmethod
    def _check_all_message_passers_terminated() -> (bool, dict):
        """
        Helper function to check if all message passers are terminated

        Returns
        -------
        bool, dict
            result, dict of the names of the ones remaining and their appx queue size

        """

        result = _ALL_MESSAGE_PASSERS_LOCK.acquire(blocking=False)
        if result is False:
            return False, None

        check = (len(_ALL_MESSAGE_PASSERS) == 0)
        names_queue = {
            item.__class__.__name__: item.get_approximate_message_passer_queue_size()
            for item
            in _ALL_MESSAGE_PASSERS
        }
        _ALL_MESSAGE_PASSERS_LOCK.release()
        return check, names_queue

    def terminate_all_message_passers(self, block: bool = True):
        """
        terminates all message passers.

        Parameters
        ----------
        block : bool, optional
            whether to block for all message passers to terminate

        """

        # send termination message
        with _ALL_MESSAGE_PASSERS_LOCK:
            for message_passer in _ALL_MESSAGE_PASSERS:
                message_passer.__terminate_message_passing__()

        if block:
            import time

            check, names_queue = self._check_all_message_passers_terminated()
            if check is False:
                i = 0
                while i != self.number_of_retries:
                    check, names_queue = self._check_all_message_passers_terminated()
                    if check is True:
                        break
                    # construct the wait time in good format
                    wait_time = i * self.retry_interval
                    wait_time_list = [wait_time // 3600, (wait_time // 60) % 60, wait_time % 60]
                    # find the first non-zero time and remove the previous ones
                    first_nonzero_idx = next((i for i, x in enumerate(wait_time_list) if x), len(wait_time_list) - 1)
                    wait_str = " and ".join([
                        f"{wait_time_list[0]} hours",
                        f"{wait_time_list[1]} minutes",
                        f"{wait_time_list[2]} seconds",
                    ][first_nonzero_idx:])

                    log_str = f'waiting on all message passers to terminate... ' \
                              f'this is retry number {i}/{self.number_of_retries} and ' \
                              f'it has been {wait_str}. '
                    if names_queue is not None:
                        log_str += \
                            f"the followings are still remaining with approximate queue sizes:\n\n\t - " + \
                            '\n\t - '.join([f"{k}: {v}" for k, v in names_queue.items()]) + '\n\n'
                    self._log.info(log_str)
                    time.sleep(self.retry_interval if i != 0 else 5)
                    i += 1
                else:
                    self._log.warning('giving up on waiting for the message passers to terminate.')
                    return

            self._log.info('all message passers are terminated successfully')


def terminate_all_message_passers(config: ConfigParser, block: bool = True):
    """
    function to terminate all message passers.

    Parameters
    ----------
    config : ConfigParser
        config for the instance
    block : bool, optional
        whether to block for all message passers to terminate

    """

    _MessagePasserTerminator(config).terminate_all_message_passers(block)
