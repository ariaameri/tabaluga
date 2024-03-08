import multiprocessing
from typing import Dict, Optional
from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.result import Result
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError


class InfluxConnector(BaseWorker):
    """Class to handle InfluxDB connections."""

    def __init__(self, config: ConfigParser = None):
        """
        Initializer.

        Parameters
        ----------
        config : ConfigParser, optional
            the config

        """

        super().__init__(config)

        # parameters
        self._influx_initial_connect_blocking = self._config.get_or_else("initial_connect_blocking", False)
        self._influx_connection_timeout_ms = int(self._config.get_or_else("connection_timeout_ms", 5000))
        self._influx_user = self._config.get_or_else("influx_user", None)
        self._influx_pass = self._config.get_or_else("influx_pass", None)
        self._influx_host = self._config.get_or_else("influx_host", "localhost")
        self._influx_token = self._config.get_or_else("influx_token", "")
        self._influx_org = self._config.get_or_else("influx_org", "")
        self._influx_port = int(self._config.get_or_else("influx_port", 8086))
        self._influx_uri = self._config.get_or_else("influx_uri", None)
        self._influx_final_uri = self._influx_uri or f'http://{self._influx_host}:{self._influx_port}'

        # the influx client
        self.influx_client: InfluxDBClient = InfluxDBClient(
            url=self._influx_final_uri,
            token=self._influx_token,
            org=self._influx_org,
            timeout=self._influx_connection_timeout_ms,
        )
        if self._influx_initial_connect_blocking:
            # The ping command is cheap and does not require auth.
            if self.influx_client.ping():
                self._log.info(
                    f"successfully connected to the influxdb server at {self._influx_host}:{self._influx_port}"
                )
            else:
                self._log.error(f"failed connecting to the influx server at {self._influx_final_uri}")

        # get the write-api's
        self._write_api_sync = self.influx_client.write_api(SYNCHRONOUS)
        self._write_api_async = self.influx_client.write_api(ASYNCHRONOUS)

    def _make_record(
            self,
            measurement: str,
            fields: Dict,
            tags: Dict = None,
    ) -> Dict:
        """
        Makes an influx record from the given data.

        Parameters
        ----------
        measurement: str
            name of the measurement
        fields: Dict
            field dictionary
        tags: Dict, optional
            tags dictionary

        Returns
        -------
        Dict
            constructed record in dictionary form
        """

        if tags is not None:
            record = {
                'measurement': measurement,
                'fields': fields,
                'tags': tags,
            }
        else:
            record = {
                'measurement': measurement,
                'fields': fields,
            }

        return record

    def insert_one_sync(
            self,
            bucket: str,
            measurement: str,
            fields: Dict,
            tags: Dict = None,
            org: str = None,
    ) -> Result[None, InfluxDBError]:
        """
        Insert one record into the given bucket.

        Parameters
        ----------
        bucket : str
            name of the bucket
        measurement: str
            name of the measurement
        fields: Dict
            field dictionary
        tags: Dict, optional
            tags dictionary
        org: str, optional
            organization, must be provided if not provided initially

        Returns
        -------
        Result[None, InfluxDBError]
            result of the insertion

        """

        record = self._make_record(
            measurement=measurement,
            fields=fields,
            tags=tags,
        )

        result = Result.from_func(
            self._write_api_sync.write,
            bucket=bucket,
            record=record,
            org=org,
        )

        return result

    def insert_one_async(
            self,
            bucket: str,
            measurement: str,
            fields: Dict,
            tags: Dict = None,
            org: str = None,
    ) -> multiprocessing.pool.ApplyResult:
        """
        Insert one record into the given bucket in an async manner.

        Parameters
        ----------
        bucket : str
            name of the bucket
        measurement: str
            name of the measurement
        fields: Dict
            field dictionary
        tags: Dict, optional
            tags dictionary
        org: str, optional
            organization, must be provided if not provided initially

        Returns
        -------
        Result[None, InfluxDBError]
            result of the insertion

        """

        record = self._make_record(
            measurement=measurement,
            fields=fields,
            tags=tags,
        )

        return self._write_api_async.write(
            bucket=bucket,
            record=record,
            org=org,
        )


def init_with_config(config: ConfigParser) -> InfluxConnector:
    return InfluxConnector(config)


class _InfluxGlobal:
    """
    Wrapper class around a influx global variable.

    This class helps with influxdb connector initialization on the first demand.
    """

    def __init__(self):

        # a placeholder for the global instance
        self._influx_global: Optional[InfluxConnector] = None

    def _create_instance(self) -> None:
        """Creates the influx instance."""

        from . import config

        self._influx_global = init_with_config(config.influx_config or ConfigParser({}))

    @property
    def influx(self) -> InfluxConnector:
        """
        Returns the influx instance.

        If the instance is not yet made, this will make it.

        Returns
        -------
        InfluxConnector
        """

        # if not made, make it
        if self._influx_global is None:
            self._create_instance()

        return self._influx_global


# this is an instance that everyone can use
influx_communicator: _InfluxGlobal = _InfluxGlobal()
