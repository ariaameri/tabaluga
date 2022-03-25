from time import sleep
from typing import Callable, List
import colored
import numpy as np
from .process import Process
from ..util.config import ConfigParser
from ..communicator import mpi
import pynvml
from ..util.result import Result, Ok, Err


class CUDAInformation(Process):
    """Class to get and print cuda information."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # skip if not enabled
        if self._config.get_or_else('enabled', True) is False:
            return

        try:
            # initialize the cuda nvml library
            pynvml.nvmlInit()
        except:
            self._log.warning("could not initialize the pynvml library.")

    def process(self, data=None):
        """
        Retrieves information and prints them.

        Parameters
        ----------
        data
            Should be left empty

        Returns
        -------
        * Nothing *

        """

        # skip if not enabled
        if self._config.get_or_else('enabled', True) is False:
            return

        # skip if not the main local rank
        if not mpi.mpi_communicator.is_main_local_rank():
            return

        try:
            # initialize the cuda nvml library
            pynvml.nvmlInit()
        except:
            self._log.warning("could not initialize the pynvml library to report CUDA information.")
            return

        import re

        device_info: str = ""

        for i in range(pynvml.nvmlDeviceGetCount()):
            device_info += f"* device {i}:\n"
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # get memory info
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                device_info += \
                    f'\t \u2577 {colored.fg("cornsilk_1")}memory{colored.attr("reset")}: \n' \
                    f'\t \u251C\u2500\u2500\u2500 {colored.fg("dark_orange_3a")}used: {info.used/(1024**2)}MB{colored.attr("reset")} \n' \
                    f'\t \u251C\u2500\u2500\u2500 {colored.fg("green_3a")}free: {info.free/(1024**2)}MB{colored.attr("reset")} \n' \
                    f'\t \u2514\u2500\u2500\u2500 {colored.fg("deep_sky_blue_4c")}total: {info.total/(1024**2)}MB{colored.attr("reset")}\n\n'
            except:
                pass

            # get temperature info
            try:
                info = pynvml.nvmlDeviceGetTemperature(handle, 0)
                device_info += f'\t - {colored.fg("cornsilk_1")}temperature{colored.attr("reset")}: {info}\u00B0C\n'
            except:
                pass

        # indent the device info
        device_info = re.sub(r'(^|\n)', '\n\t\t\t', device_info)

        msg = f'\n\n'\
              f'\t{colored.fg("turquoise_2")}\u25A0 {colored.fg("blue")}CUDA information{colored.attr("reset")} is as follows:\n'\
              f'\t\t \u00B7 {colored.fg("cornflower_blue")}driver version{colored.attr("reset")}: {pynvml.nvmlSystemGetDriverVersion().decode("utf-8")}\n' \
              f'\t\t \u00B7 {colored.fg("cornflower_blue")}cuda version{colored.attr("reset")}: {pynvml.nvmlSystemGetCudaDriverVersion()}\n' \
              f'\t\t \u00B7 {colored.fg("cornflower_blue")}device count{colored.attr("reset")}: {pynvml.nvmlDeviceGetCount()}'\
              f'\t\t {device_info}\n'

        self._log.info(msg)

        # shutdown the pynvml so it does not use resources
        pynvml.nvmlShutdown()


class CUDACooler(Process):
    """Class to stall everything to let cuda gpu cool down!"""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # skip if not enabled
        if self._config.get_or_else('enabled', True) is False or mpi.mpi_communicator.is_main_local_rank() is False:
            return

        if (decision_function_res := self._get_decision_function(self._config.get_or_else('decision_method', 'max')))\
                .is_err():
            self._log.error(f'{decision_function_res.get_err()}')
            raise decision_function_res.get_err()
        self.decision_function = decision_function_res.get()

        self.threshold_high_temperature = float(self._config.get_or_else('threshold_high_temperature', 60))
        self.threshold_cold_temperature = float(self._config.get_or_else('threshold_cold_temperature', 50))
        if self.threshold_cold_temperature > self.threshold_high_temperature:
            self._log.error('cold temperature for cuda must be smaller than or equal to its high temperature')
            raise ValueError('cold temperature for cuda must be smaller than or equal to its high temperature')

        self.wait_interval_seconds = int(self._config.get_or_else('wait_interval_seconds', 60))

        # keep a state to make sure we can initialize nvml
        self._initialized = True
        try:
            # initialize the cuda nvml library
            pynvml.nvmlInit()
        except:
            self._log.warning("could not initialize the pynvml library to wait for it to cool down.")
            self._initialized = False

        # keep state to know whether we are working or not
        self._working_state = True

    def _get_decision_function(self, decision_method: str) -> Result[Callable[[List[float]], float], Exception]:
        """
        Gets a string as the decision method and returns its corresponding function.

        Parameters
        ----------
        decision_method : str
            the decision method, look at the implementation to know the accepted strings

        Returns
        -------
        Result[Callable[[List[float]], float], Exception]

        """

        if decision_method == 'mean':
            return Ok(np.mean)
        elif decision_method == 'max':
            return Ok(np.max)
        elif decision_method == 'min':
            return Ok(np.min)
        else:
            return Err(Exception(f"provided decision method '{decision_method}' is not recognized."))

    def _decide_for_threshold(self) -> (float, bool):
        """
        Decides whether the temperature is ok or not.

        Returns
        -------
        (float, bool)
        the temperature, whether it is ok or not

        """

        # get all the temperatures
        try:
            temperature = \
                self.decision_function([
                    pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(i), 0)
                    for i
                    in range(pynvml.nvmlDeviceGetCount())
                ])
        except:
            self._log.warning("could not read cuda temperatures, assuming its bad...")
            return -1, False

        # if in working state, check for high temperature
        good = True
        if \
                (self._working_state is True and temperature > self.threshold_high_temperature) \
                or \
                (self._working_state is False and temperature > self.threshold_cold_temperature):
            good = False

        return temperature, good

    def _wait(self) -> None:
        """Waits until the temperature threshold is reached."""

        while True:
            # get the decision
            temp, good = self._decide_for_threshold()
            if good is True:
                self._log.info(
                    f"temperature of {colored.fg('green')}{temp}\u00B0C{colored.attr('reset')} "
                    f"for cuda is ok, continuing..."
                )
                self._working_state = True
                return
            if good is False:
                self._log.warning(
                    f"temperature of {colored.fg('red')}{temp}\u00B0C{colored.attr('reset')} "
                    f"for cuda is too high, waiting for {self.wait_interval_seconds} seconds"
                )
                self._working_state = False
            sleep(self.wait_interval_seconds)

    def process(self, data=None):
        """
        Retrieves information and prints them.

        Parameters
        ----------
        data
            Should be left empty

        Returns
        -------
        * Nothing *

        """

        if self._config.get_or_else('enabled', True) is False:
            return

        if mpi.mpi_communicator.is_main_local_rank() is True:
            self._wait()

        # make every node wait here
        mpi.mpi_communicator.barrier()
