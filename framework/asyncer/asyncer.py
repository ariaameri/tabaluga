import time
from ..base.base import Logger, ConfigReader
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from tabaluga.framework.util.data_muncher import UPDATE_MODIFIERS as UM, UPDATE_OPERATIONS as UO, UPDATE_CONDITIONALS as UC
from tabaluga.framework.util.data_muncher import FILTER_OPERATIONS as FO, FILTER_MODIFIERS as FM
from ..util.option import Option, Some, nothing
from ..util.result import Result, Ok, Err
import asyncio
from threading import Thread
import functools
import typing


_EVENT_LOOP_NAMES = DataMuncher({
    "main": "__main__",
})


class Asyncer(Logger, ConfigReader):
    """Spawns, maintain, and destroy threads containing asyncer loops."""

    def __init__(self, config: ConfigParser = None):

        Logger.__init__(self, config)

        # variable to flag if we should block on destruction to wait for the work to finish
        self.block_on_destruction = self._config.get_or_else('block_on_destruction', False)

        # bookkeeping for the asyncer info
        initial_async_info = {}
        self.async_info = DataMuncher(initial_async_info)

        # add the main event loop
        self.create_event_loop(_EVENT_LOOP_NAMES.get('main'))

    def __del__(self):
        """To be called upon closing."""

        # check if we should block for destruction
        if self.block_on_destruction is False:
            return

        # the amount of time between each probes to check if coroutines are done
        coroutine_finish_probe_period = self._config.get_or_else('coroutine_finish_probe_period', 1)
        event_loop_finish_probe_period = self._config.get_or_else('event_loop_finish_probe_period', .5)

        async def event_loop_closer(event_loop: asyncio.AbstractEventLoop) -> None:
            """
            Function to close a single event loop.

            Parameters
            ----------
            event_loop : asyncio.AbstractEventLoop
                the event loop to close

            """

            # if not running, no problemo
            if not event_loop.is_running():
                return

            # keep probing until no coroutine is working
            while len(asyncio.all_tasks(loop=event_loop)) > 0:
                await asyncio.sleep(coroutine_finish_probe_period)

            # tell the event to close itself and probe for it, then close it
            event_loop.call_soon_threadsafe(event_loop.stop)
            while event_loop.is_running():
                await asyncio.sleep(event_loop_finish_probe_period)
            event_loop.close()

        async def close_all(coroutine_list: typing.List[typing.Coroutine]) -> None:
            """
            Function to close a single event loop.

            Parameters
            ----------
            coroutine_list : typing.List[typing.Coroutine]
                list containing the coroutines to run

            """

            await asyncio.gather(*coroutine_list)

        # make coroutines for closing all the event loops
        coroutine_list = [
            event_loop_closer(event_loop)
            for event_loop
            in self.async_info.find_all({FM.BC: {FO.REGEX: r'^\.([^\.]+)\.event_loop'}})
        ]

        # run the closing of the event loops until completion
        closer_loop = asyncio.new_event_loop()
        closer_loop.run_until_complete(close_all(coroutine_list))

    def close(self):
        """Destruct everything and close."""

        self.__del__()

    def get_event_loop_option(self, name: str) -> Option[asyncio.AbstractEventLoop]:
        """
        Returns the event loop wrapped in an Option.

        Parameters
        ----------
        name : str
            name of the event loop

        Returns
        -------
        Option[asyncio.AbstractEventLoop]
            option-wrapped event loop

        """

        return self.async_info.get_value_option(f"{name}.event_loop")

    def create_event_loop(self, name: str) -> Result[asyncio.AbstractEventLoop, ValueError]:
        """
        Creates an event loop and returns it wrapped in an Result. The result will be None if the event loop already
        exists.

        Parameters
        ----------
        name : str
            name of the event loop

        Returns
        -------
        Result[asyncio.AbstractEventLoop]
            result-wrapped event loop

        """

        # if event loop exists, return None
        if self.async_info.contains_key(name):
            return Err(ValueError("event loop already exists"))

        # check name does not contain .s
        if '.' in name:
            return Err(ValueError(f"asyncer event loop name '{name}' should not contain '.'"))

        # create the event loop
        event_loop = asyncio.new_event_loop()

        def thread_run():
            """Runs the event loop. To be used in a new thread."""
            asyncio.set_event_loop(event_loop)
            event_loop.run_forever()

        thread = Thread(
            name=f'tabaluga-asyncer-{name}',
            target=thread_run,
            args=(),
            daemon=False if self.block_on_destruction else True,
        )
        thread.start()

        self.async_info = self.async_info.update(
            {},
            {
                UO.SET: {
                    name: {
                        'thread': thread,
                        'event_loop': event_loop,
                    }
                }
            }
        )

        return Ok(event_loop)

    def add_coroutine(
            self,
            coroutine: typing.Coroutine,
            event_loop_name: str = None,
            task_name: str = None
    ) -> Result[asyncio.Task, ValueError]:
        """
        Adds a coroutine to the event loop.

        Parameters
        ----------
        coroutine : typing.Coroutine
            the coroutine to add
        event_loop_name : str, optional
            name of the event loop
        task_name : str, optional
            name of the task to assign

        Returns
        -------
        Result[asyncio.Task, ValueError]
            the result of adding the coroutine

        """

        event_loop = self.get_event_loop_option(event_loop_name or _EVENT_LOOP_NAMES.get('main'))
        if event_loop.is_empty():
            return Err(ValueError(f"no such event loop with the name '{event_loop_name}'"))

        res = Result.from_func(event_loop.get().create_task, coroutine, name=task_name)
        # set a callback to be run by the loop
        # this will force the new coroutine to be run
        if res.is_ok():
            self.add_callback(
                callback=functools.partial(time.sleep, .01),
                event_loop_name=event_loop_name
            )

        return res

    def add_callback(
            self,
            callback: functools.partial,
            event_loop_name: str = None,
    ) -> Result[asyncio.events.Handle, ValueError]:
        """
        Calls a callback using the event loop.

        Parameters
        ----------
        callback : functools.partial,
            the callback to call soon
        event_loop_name : str, optional
            name of the event loop

        Returns
        -------
        Result[asyncio.events.Handle, ValueError]
            the result of adding the callback

        """

        event_loop = self.get_event_loop_option(event_loop_name or _EVENT_LOOP_NAMES.get('main'))
        if event_loop.is_empty():
            return Err(ValueError(f"no such event loop with the name '{event_loop_name}'"))

        handle = event_loop.get().call_soon_threadsafe(callback)

        return Ok(handle)


def init_with_config(config: ConfigParser) -> Asyncer:
    return Asyncer(config)


class _AsyncerGlobal:
    """
    Wrapper class around an asyncer global variable.

    This class helps with rabbitmq connector initialization on the first demand.
    """

    def __init__(self):

        # a placeholder for the global asyncer instance
        self._asyncer_global: typing.Optional[Asyncer] = None

    def __del__(self):

        del self._asyncer_global

    def _create_instance(self) -> None:
        """Creates the asyncer instance."""

        from . import config

        self._asyncer_global = init_with_config(config.asyner_config or ConfigParser({}))

    @property
    def asyncer(self) -> Asyncer:
        """
        Returns the asyncer instance.

        If the instance is not yet made, this will make it.

        Returns
        -------
        Asyncer
        """

        # if not made, make it
        if self._asyncer_global is None:
            self._create_instance()

        return self._asyncer_global


# this is an instance that everyone can use
asyncer: _AsyncerGlobal = _AsyncerGlobal()
