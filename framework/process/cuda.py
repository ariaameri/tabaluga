from .process import Process
from framework.util.config import ConfigParser
import pynvml


class CUDAInformation(Process):
    """Class to get and print cuda information."""

    def __init__(self, config: ConfigParser):

        super().__init__(config)

        # initialize the cuda nvml library
        pynvml.nvmlInit()

    def process(self, data=None):
        """Retrieves information and prints them.

        This function can only be run once as it shuts down the pynvml at the end.

        Parameters
        ----------
        data
            Should be left empty

        Returns
        -------
        * Nothing *

        """

        import re

        device_info: str = ""

        for i in range(pynvml.nvmlDeviceGetCount()):
            device_info += f"* device {i}:\n"
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # get memory info
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                device_info += f"\t - memory | " \
                               f"used: {info.used/(1024**2)}MB, " \
                               f"free: {info.free/(1024**2)}MB, " \
                               f"total: {info.total/(1024**2)}MB\n"
            except:
                pass

            # get temperature info
            try:
                info = pynvml.nvmlDeviceGetTemperature(handle, 0)
                device_info += f"\t - temperature: {info}\u00B0C\n"
            except:
                pass

        # indent the device info
        device_info = re.sub(r'(^|\n)', '\n\t\t\t', device_info)

        msg = f"\n\n" \
              f"\t\u25A0 CUDA information is as follows:\n" \
              f"\t\t \u00B7 driver version: {pynvml.nvmlSystemGetDriverVersion().decode('utf-8')}\n" \
              f"\t\t \u00B7 cuda version: {pynvml.nvmlSystemGetCudaDriverVersion()}\n" \
              f"\t\t \u00B7 device count: {pynvml.nvmlDeviceGetCount()}" \
              f"\t\t {device_info}\n"

        self._universal_log(msg, level='info')

        # shutdown the pynvml so it does not use resources
        pynvml.nvmlShutdown()

