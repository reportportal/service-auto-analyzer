#!flask/bin/python
from flask import Flask
from flask_cors import CORS
import logging
import logging.config
import pika
import json
import os
import sys
import threading
import time
import amqp.amqp_handler as amqp_handler
import commons.launch_objects as launch_objects
from amqp.amqp import AmqpClient
from commons.esclient import EsClient


APP_CONFIG = {
    "esHost":            os.getenv("ES_HOST", "http://localhost:9200"),
    "logLevel":          os.getenv("LOGGING_LEVEL", "DEBUG"),
    "amqpUrl":           os.getenv("AMQP_URL", "amqp://rabbitmq:rabbitmq@localhost:5672"),
    "exchangeName":      os.getenv("AMQP_EXCHANGE_NAME", "analyzer"),
    "analyzerPriority":  os.getenv("ANALYZER_PRIORITY", "1"),
    "analyzerIndex":     os.getenv("ANALYZER_INDEX", "true"),
    "analyzerLogSearch": os.getenv("ANALYZER_LOG_SEARCH", "true"),
}

SEARCH_CONFIG = {
    "MinShouldMatch":           os.getenv("ES_MIN_SHOULD_MATCH", "80%"),
    "MinTermFreq":              os.getenv("ES_MIN_TERM_FREQ", 1),
    "MinDocFreq":               os.getenv("ES_MIN_DOC_FREQ", 1),
    "BoostAA":                  os.getenv("ES_BOOST_AA", 2),
    "BoostLaunch":              os.getenv("ES_BOOST_LAUNCH", 2),
    "BoostUniqueID":            os.getenv("ES_BOOST_UNIQUE_ID", 2),
    "MaxQueryTerms":            os.getenv("ES_MAX_QUERY_TERMS", 50),
    "SearchLogsMinShouldMatch": os.getenv("ES_LOGS_MIN_SHOULD_MATCH", "98%"),
}

def create_application():
    application = Flask(__name__)
    return application

def create_thread(func, args):
    thread = threading.Thread(target= func, args = args)
    thread.daemon = True
    thread.start()
    return thread

class ThreadConnectionAwaiter(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.num_of_retries = 0
        self.client = None
        self.daemon = True

    def run(self):
        logger.info("Starting waiting for AMQP connection")

        self.num_of_retries = 0
        while True:
            try:
                self.client = create_ampq_client()
                break
            except:
                if self.num_of_retries >= 10:
                    break
                time.sleep(10)
                self.num_of_retries += 1

def create_ampq_client():
    amqp_full_url = "{}/{}?heartbeat=600".format(APP_CONFIG["amqpUrl"], APP_CONFIG["exchangeName"])\
        if "heartbeat" not in APP_CONFIG["amqpUrl"] else APP_CONFIG["amqpUrl"]
    return AmqpClient(pika.BlockingConnection(\
        pika.connection.URLParameters(amqp_full_url)))

def create_es_client():
    return EsClient(APP_CONFIG["esHost"], SEARCH_CONFIG)

def declare_exchange(channel, queues, config):
    logger.info("ExchangeName: %s" % config["exchangeName"])
    try:
        channel.exchange_declare(exchange=config["exchangeName"], exchange_type='direct',\
            durable = False, auto_delete = True, internal = False, arguments = {
                "analyzer":            config["exchangeName"],
                "analyzer_index":      config["analyzerIndex"],
                "analyzer_priority":   config["analyzerPriority"],
                "analyzer_log_search": config["analyzerLogSearch"],
                "version":             1 #to do create versioning,
            })
    except Exception as err:
        logger.error("Failed to declare exchange")
        logger.error(err)
        return         
    logger.info("Exchange '%s' has been declared" % config["exchangeName"])
    return True

def init_amqp(amqp_client, request_handler):
    index_queue = "index"
    analyze_queue = "analyze"
    delete_queue = "delete"
    clean_queue = "clean"
    search_queue = "search"

    queues = [index_queue, analyze_queue, delete_queue, clean_queue, search_queue]
    with amqp_client.connection.channel() as channel:
        try:
            declare_exchange(channel, queues, APP_CONFIG)
        except Exception as err:
            logger.error("Failed to declare amqp objects")
            logger.error(err)
            return
    
    create_thread(create_ampq_client().receive,
        (APP_CONFIG["exchangeName"], index_queue, True, False, False, False,
     lambda channel, method, props, body: amqp_handler.handle_amqp_request(channel,
        method, props, body, request_handler.index_logs,
        prepare_response_data = amqp_handler.prepare_index_response_data)))
    create_thread(create_ampq_client().receive,
        (APP_CONFIG["exchangeName"], analyze_queue, True, False, False, False,
     lambda channel, method, props, body: amqp_handler.handle_amqp_request(channel,
        method, props, body, request_handler.analyze_logs,
        prepare_response_data = amqp_handler.prepare_analyze_response_data)))
    create_thread(create_ampq_client().receive,
        (APP_CONFIG["exchangeName"], delete_queue, True, False, False, False,
     lambda channel, method, props, body: amqp_handler.handle_delete_request(channel,
        method, props, body, request_handler.delete_logs)))
    create_thread(create_ampq_client().receive,
        (APP_CONFIG["exchangeName"], clean_queue, True, False, False, False,
     lambda channel, method, props, body: amqp_handler.handle_clean_request(channel,
        method, props, body, request_handler.delete_index)))
    create_thread(create_ampq_client().receive,
        (APP_CONFIG["exchangeName"], search_queue, True, False, False, False,
     lambda channel, method, props, body: amqp_handler.handle_amqp_request(channel,
        method, props, body, request_handler.search_logs,
        prepare_data_func = amqp_handler.prepare_search_logs,
        prepare_response_data = amqp_handler.prepare_search_response_data
        )))

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger("analyzerApp")

application = create_application()
CORS(application)

program_initialized = False
amqp_connection_awaiter = ThreadConnectionAwaiter()
amqp_connection_awaiter.start()
amqp_connection_awaiter.join()
if amqp_connection_awaiter.client is not None:
    amqp_client = amqp_connection_awaiter.client
    init_amqp(amqp_client, create_es_client())
    program_initialized = True
else:
    logger.error("Amqp connection was not established")

@application.route('/', methods=['GET'])
def get_analyzer_info():
    return "Analyzer is running"

if __name__ == '__main__':
    logger.info("Program started")

    if program_initialized:
        application.run(host='0.0.0.0', port=5000)
