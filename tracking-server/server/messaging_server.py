import pika


class MessagingServer(object):
    """The wrapper used to connect to a RabbitMQ server and send message"""

    def __init__(self, server_name):
        """
        Instantiates a new MessagingServer
        :param server_name: The DNS or IP of a running RabbitMQ server
        """
        self._server_name = server_name
        self._exchange = "snooker"
        self._connection = None
        self._channel = None

    def connect(self):
        """Sets up the connection to the RabbitMQ server"""
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=self._server_name))
        self._channel = self._connection.channel()

        self._channel.exchange_declare(
            exchange=self._exchange,
            type='direct'
        )

    def disconnect(self):
        """Closes the connection to the RabbitMQ server"""
        self._connection.close()

    def send(self, message, route):
        """
        Sends the specified message to the RabbitMQ server through a route specified
        :param message: The message to send to the server
        :param route: The route that filters the receiver of the message
        """
        self._channel.basic_publish(
            exchange=self._exchange,
            routing_key=route,
            body=message
        )

