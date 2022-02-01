from ..base.base import BaseWorker
from ..util.config import ConfigParser
from ..util.data_muncher import DataMuncher
from ..util.data_muncher import UPDATE_MODIFIERS as UM, UPDATE_OPERATIONS as UO, UPDATE_CONDITIONALS as UC
from ..util.data_muncher import FILTER_OPERATIONS as FO, FILTER_MODIFIERS as FM
from ..util.option import Option
from typing import Optional, Callable, Union, List
import kombu
import kombu.pools
from kombu import mixins
from enum import Enum


class RabbitMQExchangeType(Enum):

    DIRECT = "direct"
    FANOUT = "fanout"
    TOPIC = "topic"
    HEADERS = "headers"


class RabbitMQConsumer(mixins.ConsumerMixin, BaseWorker):

    def __init__(
            self,
            connection: kombu.Connection,
            name: str,
            no_ack: bool = True,
            auto_declare: bool = True,
            on_decode_error=None,
            prefetch_count: int = None,
    ):
        """
        Initializer

        Parameters
        ----------
        connection : kombu.Connection
            connection to use
        name : str
            name of this instance
        for the rest, look at kombu doc
        """

        mixins.ConsumerMixin.__init__(connection)
        BaseWorker.__init__(self, ConfigParser())

        self.connection = connection

        # name for this instance
        self.name = name

        # callbacks for when a message is received
        self.callbacks: List[Callable] = []

        # queues to listen to
        self.queues: List[kombu.Queue] = []

        # book keeping
        self.no_ack = no_ack
        self.auto_declare = auto_declare
        self.on_decode_error = on_decode_error
        self.prefetch_count = prefetch_count

    def add_callback(self, funcs: List[Callable]):
        """
        Adds callbacks to the list of callbacks to be called upon new message

        Parameters
        ----------
        funcs
            function with the parameters of (body, message)

        Returns
        -------
        self

        """

        self.callbacks.extend(funcs)

        return self

    def add_queue(self, queues: List[kombu.Queue]):
        """
        Adds queues to the list of queues to be listened to

        Parameters
        ----------
        queues : list[kombu.Queue]
            list of queues

        Returns
        -------
        self

        """

        self.queues.extend(queues)

        return self

    def get_consumers(self, Consumer, channel):
        """
        Returns the consumers for this instance

        Parameters
        ----------
        Consumer
        channel

        """

        return [
            Consumer(
                # channel=self.connection,
                queues=self.queues,
                callbacks=self.callbacks,
                no_ack=self.no_ack,
                auto_declare=self.auto_declare,
                on_decode_error=self.on_decode_error,
                prefetch_count=self.prefetch_count,
            )
        ]

    def on_connection_error(self, exc, interval):
        self._log.debug(f"consumer of name '{self.name}' disconnected.")

    def on_connection_revived(self):
        self._log.debug(f"consumer of name '{self.name}' reconnected.")

    def on_consume_ready(self, connection, channel, consumers, **kwargs):
        self._log.debug(f"consumer of name '{self.name}' ready.")

    def on_consume_end(self, connection, channel):
        self._log.debug(f"consumer of name '{self.name}' ended consumption.")


class RabbitMQCommunicator(BaseWorker):
    """
    Class to handle RabbitMQ-related tasks.

    This class is a singleton. This has to be created only once and then reused.
    """

    def __init__(self, config: ConfigParser):
        """
        Initializer.

        Parameters
        ----------
        config : ConfigParser
            config to contain all the necessary information
        """

        super().__init__(config)

        # set logger name
        self._log.set_name("RabbitMQ")

        # parameters
        self._broker_user = self._config.get_or_else("broker_user", None)
        self._broker_pass = self._config.get_or_else("broker_pass", None)
        self._broker_host = self._config.get_or_else("broker_host", "localhost")
        self._broker_port = self._config.get_or_else("broker_port", 5672)

        # book keeping the connections
        connection = self._make_connection()
        # make it connect
        self.connect(connection)
        # set the maximum number of connections
        # we will use the pool only for producers
        # this is only because a single kombu producer does not work well with multiple thread as I tested
        # for consumers, we will have another connection with different channels
        self._config.get_option("pool_limit").map(lambda x: kombu.pools.set_limit(x))
        # save the result
        initial_broker_info = {
            # we will have one connection for publishing
            'publish': {
                "connection": connection,
                "channels": {
                    "main": connection.channel(),  # for the main stuff!
                    "declaration": connection.channel(),  # only for declaring stuff
                },
                "producer_pool": kombu.pools.producers[connection],
            },
            # we will have one separate connection for consuming
            'consumer': {
                "connection": connection.clone(),
                "channels": [],
            },
            # list of exchanges made
            # all exchanges should be unbound
            'exchange': {},
            # list of queues made
            # all queues should be unbound
            'queue': {},
        }
        self._broker_info = DataMuncher(initial_broker_info)

    def _make_connection(self) -> kombu.Connection:
        """
        Makes and returns a kombu connection

        Returns
        -------
        kombu.Connection
            connection
        """

        connection = \
            kombu.Connection(
                hostname=self._broker_host,
                port=self._broker_port,
                userid=self._broker_user,
                password=self._broker_pass,
                ssl=self._config.get_or_else('broker_ssl', False),
                connect_timeout=self._config.get_or_else('broker_connect_timeout', 5),

            )

        return connection

    def connect(self, connection: kombu.Connection) -> None:
        """Connects to the broker."""

        # log
        self._log.info(f'connecting to the broker at {self._broker_user}@{self._broker_host}:{self._broker_port}')

        # force the connection to connect
        # we want the error to propagate
        try:
            connection.connect()
            self._log.info(f'successfully connected to the broker at '
                           f'{self._broker_user}@{self._broker_host}:{self._broker_port}'
                           )
        except BaseException as e:
            self._log.error(f'error in connecting to the broker at '
                            f'{self._broker_user}@{self._broker_host}:{self._broker_port}'
                            f'with error of "{e}"'
                            )
            raise e

    # exchange methods

    def get_exchange_option(self, name: str) -> Option[kombu.Exchange]:
        """Returns an Option value of the exchange"""

        return self._broker_info.get_value_option(f'exchange.{name}')

    def get_exchange(self, name: str) -> kombu.Exchange:
        """Returns an exchange, if does not exist, throws an error."""

        exchange = self.get_exchange_option(name)
        if exchange.is_empty():
            raise ValueError(f"Exchange '{name}' does not exist.")

        return exchange.get()

    def make_and_get_exchange(
            self,
            name: str,
            type: RabbitMQExchangeType = RabbitMQExchangeType.DIRECT,
            return_on_exist: bool = False,
            declare: bool = True,
            channel: kombu = None,
            durable: bool = True,
            auto_delete: bool = False,
            delivery_mode: int = 2,
            arguments: dict = None,
    ) -> kombu.Exchange:
        """
        Makes an Exchange instance, saves it, and return it.

        Parameters
        ----------
        name : str
            name of the exchange
        type : RabbitMQExchangeType, optional
            routing type of the exchange
        return_on_exist : bool, optional
            return the exchange if it does exist without creating it. Note that if the exchange 'name' exists, it will
            return it even though it might be different from the one that would be created from the given info.
        For more information, see kombu doc.

        Returns
        -------
        kombu.Exchange
            instance of the exchange
        """

        # check if the exchange does not exist
        if self.get_exchange_option(name).is_defined():
            if return_on_exist is True:
                return self.get_exchange(name=name)
            else:
                raise ValueError(f"Exchange '{name}' already exists.")

        exchange = \
            kombu.Exchange(
                name=name,
                type=type.value,
                channel=channel,
                durable=durable,
                auto_delete=auto_delete,
                delivery_mode=delivery_mode,
                arguments=arguments,
            )

        # if we have to declare it
        # bind it to a channel, declare, then unbind it
        # if declare is True:
        #     exchange.bind(self._broker_info.get('publish.channels.declaration'))
        #     exchange.declare()
        #     exchange.unbind_from()

        # save it
        self._broker_info = self._broker_info.update({FM.BC: {FO.REGEX: 'exchange$'}}, {UO.SET: {name: exchange}})

        return exchange

    # queue methods

    def get_queue_option(self, name: str) -> Option[kombu.Queue]:
        """Returns an Option value of the queue"""

        return self._broker_info.get_value_option(f'queue.{name}')

    def get_queue(self, name: str) -> kombu.Queue:
        """Returns an queue, if does not exist, throws an error."""

        queue = self.get_queue_option(name)
        if queue.is_empty():
            raise ValueError(f"Exchange '{name}' does not exist.")

        return queue.get()

    def make_and_get_queue(
            self,
            name: str,
            exchange_name: str = None,
            routing_key: str = '',
            return_on_exist: bool = False,
            durable: bool = True,
            exclusive: bool = False,
            auto_delete: bool = False,
            queue_arguments: dict = None,
            binding_arguments: dict = None,
            consumer_arguments: dict = None,
            on_declared=None,
            expires: float = None,
            message_ttl: float = None,
            max_length: int = None,
            max_length_bytes: int = None,
            max_priority: int = None,
    ) -> kombu.Queue:
        """
        Makes a Queue instance, saves it, and return it.

        Parameters
        ----------
        name : str
            name of the queue
        exchange_name : str, optional
            the name of the exchange in case we want to bind this queue to
            note that this exchange has to be defined before and will look for it in the internal db
        routing_key : str, optional
            routing key to be used in case of binding to exchange
        return_on_exist : bool, optional
            return the exchange if it does exist without creating it. Note that if the exchange 'name' exists, it will
            return it even though it might be different from the one that would be created from the given info.
        For more information, see kombu doc.

        Returns
        -------
        kombu.Queue
            instance of Queue

        """

        # check if the queue does not exist
        if self.get_queue_option(name).is_defined():
            if return_on_exist is True:
                return self.get_queue(name=name)
            else:
                raise ValueError(f"Queue '{name}' already exists.")

        # get the exchange
        if self.get_exchange_option(exchange_name).is_empty():
            raise ValueError(f"Exchange {exchange_name} does not exist to use for queue {name}")

        queue = \
            kombu.Queue(
                name=name,
                exchange=self.get_exchange(exchange_name),
                routing_key=routing_key,
                durable=durable,
                exclusive=exclusive,
                auto_delete=auto_delete,
                queue_arguments=queue_arguments,
                binding_arguments=binding_arguments,
                consumer_arguments=consumer_arguments,
                on_declared=on_declared,
                expires=expires,
                message_ttl=message_ttl,
                max_length=max_length,
                max_length_bytes=max_length_bytes,
                max_priority=max_priority,
            )

        # save it
        self._broker_info = self._broker_info.update({FM.BC: {FO.REGEX: 'queue$'}}, {UO.SET: {name: queue}})

        return queue

    # publish methods

    def publish(
            self,
            body,
            routing_key: str,
            exchange: [kombu.Exchange, str] = None,
            extra_declare: List[Union[kombu.Exchange, kombu.Queue]] = None,
            block: bool = True,
            delivery_mode=None,
            mandatory: bool = False,
            immediate: bool = False,
            priority: int = 0,
            content_type: str = None,
            content_encoding: str = None,
            serializer: str = None,
            compression: str = None,
            headers: dict = None,
            retry: bool = True,
            retry_policy: dict = None,
            expiration: float = None,
            timeout: float = None,
    ) -> bool:
        """
        Publishes a message to the broker.

        Parameters
        ----------
        body : Any
            the message body
        routing_key : str
            the routing key
        exchange : Union[kombu.Exchange, str]
            the exchange to publish at
        extra_declare : list[Union[kombu.Exchange, kombu.Queue]]
            a list of the rest of the items to be declared
        for the rest of the items, look at kombu doc.

        Returns
        -------
        bool
            whether or not the operation succeeded

        """

        # get the pool
        producer_pool: kombu.pools.ProducerPool = self._broker_info.get('publish.producer_pool')

        # get a list of declaration
        # first make sure extra_declares are of correct type
        if extra_declare is not None:
            if any([
                not (isinstance(item, kombu.Queue) or isinstance(item, kombu.Exchange))
                for item
                in extra_declare
            ]) is True:
                self._log.error(
                    "given values for declaring while publishing has to be either kombu.Exchange or kombu.Queue"
                )
                raise ValueError(
                    "given values for declaring while publishing has to be either kombu.Exchange or kombu.Queue"
                )
        else:
            extra_declare = []
        # we will always declare the exchange
        declaration = [exchange] if exchange is not None and isinstance(exchange, kombu.Exchange) else []
        declaration = declaration + extra_declare
        declaration = declaration or None

        # get a producer and publish
        with producer_pool.acquire(block=block) as producer:

            return producer.publish(
                body=body,
                routing_key=routing_key,
                declare=declaration,
                delivery_mode=delivery_mode,
                mandatory=mandatory,
                immediate=immediate,
                priority=priority,
                content_type=content_type,
                content_encoding=content_encoding,
                serializer=serializer,
                compression=compression,
                headers=headers,
                exchange=exchange,
                retry=retry,
                retry_policy=retry_policy,
                expiration=expiration,
                timeout=timeout,
            )

    # consume methods

    def consume(
            self,
            name: str,
            queue_names: List[str] = None,
            no_ack: bool = True,
            auto_declare: bool = True,
            on_decode_error=None,
            prefetch_count: int = None,
    ) -> RabbitMQConsumer:
        """
        blockingly consumes a thread

        Parameters
        ----------
        name : str
            name to use for the consumer
        queue_names : list[str]
            list of name of queues that were previously declared
        for the rest, look at kombu doc

        Returns
        -------
        RabbitMQConsumer
            instance of the RabbitMQConsumer
        """

        # get the queues
        queues_list = [self.get_queue(queue_name) for queue_name in (queue_names if queue_names is not None else [])]

        return RabbitMQConsumer(
            connection=self._broker_info.get('consumer.connection'),
            name=name,
            no_ack=no_ack,
            auto_declare=auto_declare,
            on_decode_error=on_decode_error,
            prefetch_count=prefetch_count,
        ).add_queue(queues_list)


def init_with_config(config: ConfigParser) -> RabbitMQCommunicator:
    return RabbitMQCommunicator(config)


class _RabbitMQGlobal:
    """
    Wrapper class around a rabbitmq global variable.

    This class helps with rabbitmq connector initialization on the first demand.
    """

    def __init__(self):

        # a placeholder for the global rabbitmq instance
        self._rabbitmq_global: Optional[RabbitMQCommunicator] = None

    def _create_instance(self) -> None:
        """Creates the rabbitmq instance."""

        from . import config

        self._rabbitmq_global = init_with_config(config.rabbit_config or ConfigParser({}))

    @property
    def rabbitmq(self) -> RabbitMQCommunicator:
        """
        Returns the rabbitmq instance.

        If the instance is not yet made, this will make it.

        Returns
        -------
        RabbitMQCommunicator
        """

        # if not made, make it
        if self._rabbitmq_global is None:
            self._create_instance()

        return self._rabbitmq_global


# this is an instance that everyone can use
rabbitmq_communicator: _RabbitMQGlobal = _RabbitMQGlobal()
