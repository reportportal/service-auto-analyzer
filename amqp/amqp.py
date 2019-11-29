import logging
import os

logger = logging.getLogger("analyzerApp.amqp")

class AmqpClient:

    def __init__(self, connection):
        self.connection = connection

    def bind_queue(self, channel, name, exchange_name):
        try:
            result = channel.queue_declare(queue = name, durable=False,
             exclusive=False, auto_delete=True, arguments=None)
        except Exception as err:
            logger.error("Failed to open a channel pid(%d)"%os.getpid())
            logger.error(err)
            raise
        logger.info("Queue '%s' has been declared pid(%d)"%(result.method.queue, os.getpid()))
        try:
            channel.queue_bind(exchange=exchange_name, queue=result.method.queue, routing_key = name)
        except Exception as err:
            logger.error("Failed to open a channel pid(%d)"%os.getpid())
            logger.error(err)
            raise
        return True

    def receive(self, exchange_name, queue, auto_ack, exclusive, no_local, no_wait, msg_callback):
        try:
            channel = self.connection.channel()
            self.bind_queue(channel, queue, exchange_name)
            self.consume_queue(channel, queue, auto_ack, exclusive, no_local, no_wait, msg_callback)
            logger.info("started consuming pid(%d) on the queue %s"%(os.getpid(), queue))
            channel.start_consuming()
        except Exception as err:
            logger.error("Failed to consume messages pid(%d) in queue %s"%(os.getpid(), queue))
            logger.error(err)

    def consume_queue(self, channel, queue, auto_ack, exclusive, no_local, no_wait, msg_callback):
        try:
            channel.basic_qos(prefetch_count=1, prefetch_size = 0)
        except Exception as err:
            logger.error("Failed to configure Qos pid(%d)"%os.getpid())
            logger.error(err)
            raise
        try:
            channel.basic_consume(queue=queue, auto_ack = auto_ack, exclusive = exclusive,
                on_message_callback=msg_callback) #don't know how to set noWait and noLocal
        except Exception as err:
            logger.error("Failed to register a consumer pid(%d)"%os.getpid())
            logger.error(err)
            raise
