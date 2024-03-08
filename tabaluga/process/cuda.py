import colored
from .process import Process
from ..util.config import ConfigParser
from ..communicator import mpi


class CUDAInformation(Process):
    """Class to get and print cuda information."""

    def __init__(self, config: ConfigParser = None):

        super().__init__(config)

        # skip if not enabled
        if self._config.get_or_else('enabled', True) is False:
            return

        try:
            import pynvml
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
            import pynvml
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
                device_info += f'\t - {colored.fg("cornsilk_1")}temperature{colored.attr("reset")}: {info}\u00B0C\n\n'
            except:
                pass

        # indent the device info
        device_info = re.sub(r'(^|\n)', '\n', device_info)

        msg = f''\
              f'{colored.fg("turquoise_2")}\u25A0 {colored.fg("blue")}CUDA information{colored.attr("reset")} is as follows:\n\n'\
              f'\u00B7 {colored.fg("cornflower_blue")}driver version{colored.attr("reset")}: {pynvml.nvmlSystemGetDriverVersion()}\n' \
              f'\u00B7 {colored.fg("cornflower_blue")}cuda version{colored.attr("reset")}: {pynvml.nvmlSystemGetCudaDriverVersion()}\n' \
              f'\u00B7 {colored.fg("cornflower_blue")}device count{colored.attr("reset")}: {pynvml.nvmlDeviceGetCount()}\n'\
              f'{device_info}\n'

        self._log.info(msg)

        # shutdown the pynvml so it does not use resources
        pynvml.nvmlShutdown()

