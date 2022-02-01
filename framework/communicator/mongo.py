from typing import Dict, Iterable, Optional
import pymongo.database
import pymongo.collection
import pymongo.results
from pymongo import MongoClient
from pymongo import errors as pyerrors
from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.result import Result
from dataclasses import dataclass


class MongoContainer:
    """base class for all mongo containers"""
    pass


@dataclass
class MongoDB(MongoContainer):
    db: pymongo.database.Database


@dataclass
class MongoCollection(MongoContainer):
    db: pymongo.database.Database
    collection: pymongo.collection.Collection


class MongoConnector(BaseWorker):
    """Class to handle MongoDB connections."""

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
        self._mongo_initial_connect = self._config.get_or_else("initial_connect", True)
        self._mongo_initial_connect_blocking = self._config.get_or_else("initial_connect_blocking", False)
        self._mongo_connection_timeout_ms = int(self._config.get_or_else("connection_timeout_ms", 5000))
        self._mongo_socket_timeout_ms = int(self._config.get_or_else("socket_timeout_ms", 5000))
        self._mongo_server_selection_timeout_ms = int(self._config.get_or_else("server_selection_timeout_ms", 5000))
        self._mongo_wait_queue_timeout_ms = int(self._config.get_or_else("wait_queue_timeout_ms", 5000))
        self._mongo_user = self._config.get_or_else("mongo_user", None)
        self._mongo_pass = self._config.get_or_else("mongo_pass", None)
        self._mongo_host = self._config.get_or_else("mongo_host", "localhost")
        self._mongo_port = int(self._config.get_or_else("mongo_port", 27017))
        self._mongo_uri = self._config.get_or_else("mongo_uri", None)
        self._mongo_final_uri = self._mongo_uri or f'mongodb://{self._mongo_host}:{self._mongo_port}'

        # the mongo client
        self.mongo_client: MongoClient = MongoClient(
            host=self._mongo_final_uri,
            port=self._mongo_port,
            connect=self._mongo_initial_connect,
            socketTimeoutMS=self._mongo_socket_timeout_ms,
            serverSelectionTimeoutMS=self._mongo_server_selection_timeout_ms,
            waitQueueTimeoutMS=self._mongo_wait_queue_timeout_ms,
            connectTimeoutMS=self._mongo_connection_timeout_ms,
            username=self._mongo_user,
            password=self._mongo_pass,
        )
        if self._mongo_initial_connect_blocking:
            try:
                # The ping command is cheap and does not require auth.
                self._log.info(f"attempting to connect to the mongo server at {self._mongo_final_uri}")
                self.mongo_client.admin.command('ping')
                self._log.info(f"successfully connected to the mongo server at {self._mongo_final_uri}")
            except pyerrors.ConnectionFailure:
                self._log.error(f"failed connecting to the mongo server at {self._mongo_final_uri}")

    def get_database(self, db_name: str) -> MongoDB:
        """
        Gets a database from the mongo server.

        Parameters
        ----------
        db_name : str
            name of the database

        Returns
        -------
        MongoDB
            the mongo database

        """

        return MongoDB(self.mongo_client[db_name])

    def get_collection(self, db_name: str, collection_name: str) -> MongoCollection:
        """
        Gets a collection from the mongo server.

        Parameters
        ----------
        db_name : str
            name of the database
        collection_name : str
            name of the collection

        Returns
        -------
        MongoCollection
            the mongo collection

        """

        db = self.mongo_client[db_name]

        return MongoCollection(db, db[collection_name])

    def insert_one(
            self,
            collection: MongoCollection,
            document: Dict
    ) -> Result[pymongo.results.InsertOneResult, pyerrors.PyMongoError]:
        """
        Insert one document into the given collection.

        Parameters
        ----------
        collection : MongoCollection
            the collection to insert the document to
        document : Dict
            the document

        Returns
        -------
        Result[pymongo.results.InsertOneResult, pyerrors.PyMongoError]
            result of the insertion

        """

        return Result.from_func(collection.collection.insert_one, document=document)

    def insert_many(
            self,
            collection: MongoCollection,
            documents: Iterable[Dict],
    ) -> Result[pymongo.results.InsertManyResult, pyerrors.PyMongoError]:
        """
        Insert one document into the given collection.

        Parameters
        ----------
        collection : MongoCollection
            the collection to insert the document to
        documents: Iterable[Dict]
            the documents

        Returns
        -------
        Result[pymongo.results.InsertManyResult, pyerrors.PyMongoError]
            result of the insertion

        """

        return Result.from_func(collection.collection.insert_many, documents=documents)


def init_with_config(config: ConfigParser) -> MongoConnector:
    return MongoConnector(config)


class _MongoGlobal:
    """
    Wrapper class around a mongo global variable.

    This class helps with mongodb connector initialization on the first demand.
    """

    def __init__(self):

        # a placeholder for the global instance
        self._mongo_global: Optional[MongoConnector] = None

    def _create_instance(self) -> None:
        """Creates the mongo instance."""

        from . import config

        self._mongo_global = init_with_config(config.mongo_config or ConfigParser({}))

    @property
    def mongo(self) -> MongoConnector:
        """
        Returns the mongo instance.

        If the instance is not yet made, this will make it.

        Returns
        -------
        MongoConnector
        """

        # if not made, make it
        if self._mongo_global is None:
            self._create_instance()

        return self._mongo_global


# this is an instance that everyone can use
mongo_communicator: _MongoGlobal = _MongoGlobal()
