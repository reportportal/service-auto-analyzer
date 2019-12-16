"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import logging
import logging.config
import os
import threading
import time
import json
import pika
from flask import Flask
from flask_cors import CORS
import amqp.amqp_handler as amqp_handler
from amqp.amqp import AmqpClient
from commons.esclient import EsClient


APP_CONFIG = {
    "esHost":            os.getenv("ES_HOST", "http://elasticsearch:9200"),
    # "esHost":            os.getenv("ES_HOST", "http://localhost:9200"),
    "logLevel":          os.getenv("LOGGING_LEVEL", "DEBUG"),
    "amqpUrl":           os.getenv("AMQP_URL", "amqp://rabbitmq:rabbitmq@rabbitmq:5672"),
    # "amqpUrl":           os.getenv("AMQP_URL", "amqp://rabbitmq:rabbitmq@localhost:5672"),
    "exchangeName":      os.getenv("AMQP_EXCHANGE_NAME", "analyzer"),
    "analyzerPriority":  int(os.getenv("ANALYZER_PRIORITY", "1")),
    "analyzerIndex":     json.loads(os.getenv("ANALYZER_INDEX", "true").lower()),
    "analyzerLogSearch": json.loads(os.getenv("ANALYZER_LOG_SEARCH", "true").lower()),
}

SEARCH_CONFIG = {
    "MinShouldMatch":           os.getenv("ES_MIN_SHOULD_MATCH", "80%"),
    "MinTermFreq":              int(os.getenv("ES_MIN_TERM_FREQ", "1")),
    "MinDocFreq":               int(os.getenv("ES_MIN_DOC_FREQ", "1")),
    "BoostAA":                  float(os.getenv("ES_BOOST_AA", "2.0")),
    "BoostLaunch":              float(os.getenv("ES_BOOST_LAUNCH", "2.0")),
    "BoostUniqueID":            float(os.getenv("ES_BOOST_UNIQUE_ID", "2.0")),
    "MaxQueryTerms":            int(os.getenv("ES_MAX_QUERY_TERMS", "50")),
    "SearchLogsMinShouldMatch": os.getenv("ES_LOGS_MIN_SHOULD_MATCH", "98%"),
    "SearchLogsMinSimilarity":  float(os.getenv("ES_LOGS_MIN_SHOULD_MATCH", "0.9")),
    "MinWordLength":            int(os.getenv("ES_MIN_WORD_LENGTH", "0")),
}


def create_application():
    """Creates a Flask application"""
    _application = Flask(__name__)
    return _application


def create_thread(func, args):
    """Creates a thread with specified function and arguments"""
    thread = threading.Thread(target=func, args=args)
    thread.daemon = True
    thread.start()
    return thread


class ThreadConnectionAwaiter(threading.Thread):
    """ThreadConnectionAwaiter waits for amqp connection establishment"""
    def __init__(self):
        threading.Thread.__init__(self)
        self.num_of_retries = 10
        self.client = None
        self.daemon = True

    def run(self):
        """ThreadConnectionAwaiter starts to wait for amqp connection establishment"""
        self.num_of_retries = 10
        while True:
            try:
                self.client = create_ampq_client()
                break
            except Exception as err:
                self.num_of_retries -= 1
                logger.error("Amqp connection was not established. %d tries are left",
                             self.num_of_retries)
                logger.error(err)
                if self.num_of_retries <= 0:
                    break
                time.sleep(10)


def create_ampq_client():
    """Creates AMQP client"""
    amqp_full_url = "{}/{}?heartbeat=600".format(APP_CONFIG["amqpUrl"], APP_CONFIG["exchangeName"])\
        if "heartbeat" not in APP_CONFIG["amqpUrl"] else APP_CONFIG["amqpUrl"]
    return AmqpClient(pika.BlockingConnection(
        pika.connection.URLParameters(amqp_full_url)))


def create_es_client():
    """Creates Elasticsearch client"""
    return EsClient(APP_CONFIG["esHost"], SEARCH_CONFIG)


def declare_exchange(channel, config):
    """Declares exchange for rabbitmq"""
    logger.info("ExchangeName: %s", config["exchangeName"])
    try:
        channel.exchange_declare(exchange=config["exchangeName"], exchange_type='direct',
                                 durable=False, auto_delete=True, internal=False,
                                 arguments={
                                     "analyzer":            config["exchangeName"],
                                     "analyzer_index":      config["analyzerIndex"],
                                     "analyzer_priority":   config["analyzerPriority"],
                                     "analyzer_log_search": config["analyzerLogSearch"],
                                     "version":             version, })
    except Exception as err:
        logger.error("Failed to declare exchange")
        logger.error(err)
        return False
    logger.info("Exchange '%s' has been declared", config["exchangeName"])
    return True


def init_amqp(_amqp_client, request_handler):
    """Initialize rabbitmq queues, exchange and stars threads for queue messages processing"""
    index_queue = "index"
    analyze_queue = "analyze"
    delete_queue = "delete"
    clean_queue = "clean"
    search_queue = "search"

    with _amqp_client.connection.channel() as channel:
        try:
            declare_exchange(channel, APP_CONFIG)
        except Exception as err:
            logger.error("Failed to declare amqp objects")
            logger.error(err)
            return

    create_thread(create_ampq_client().receive,
                  (APP_CONFIG["exchangeName"], index_queue, True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    request_handler.index_logs,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_index_response_data)))
    create_thread(create_ampq_client().receive,
                  (APP_CONFIG["exchangeName"], analyze_queue, True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    request_handler.analyze_logs,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_analyze_response_data)))
    create_thread(create_ampq_client().receive,
                  (APP_CONFIG["exchangeName"], delete_queue, True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_delete_request(method, props, body,
                                                      request_handler.delete_index)))
    create_thread(create_ampq_client().receive,
                  (APP_CONFIG["exchangeName"], clean_queue, True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_clean_request(method, props, body,
                                                     request_handler.delete_logs)))
    create_thread(create_ampq_client().receive,
                  (APP_CONFIG["exchangeName"], search_queue, True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    request_handler.search_logs,
                                                    prepare_data_func=amqp_handler.
                                                    prepare_search_logs,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_search_response_data)))


def read_version():
    """Reads the application build version"""
    version_filename = "VERSION"
    if os.path.exists(version_filename):
        with open(version_filename, "r") as file:
            return file.read().strip()
    return ""


log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger("analyzerApp")
version = read_version()

application = create_application()
CORS(application)

program_initialized = False

logger.info("Starting waiting for AMQP connection")
amqp_connection_awaiter = ThreadConnectionAwaiter()
amqp_connection_awaiter.start()
amqp_connection_awaiter.join()
if amqp_connection_awaiter.client is not None:
    amqp_client = amqp_connection_awaiter.client
    init_amqp(amqp_client, create_es_client())
    program_initialized = True
else:
    logger.error("Amqp connection was not established")

if __name__ == '__main__':
    if program_initialized:
        logger.info("Analyzer has started")
        application.run(host="0.0.0.0", port=5002)
