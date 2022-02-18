from typing import Optional
from ..util.config import ConfigParser


# store configs
mpi_config: Optional[ConfigParser] = None
influx_config: Optional[ConfigParser] = None
rabbit_config: Optional[ConfigParser] = None
mongo_config: Optional[ConfigParser] = None
zmq_config: Optional[ConfigParser] = None
