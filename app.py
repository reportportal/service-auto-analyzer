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
from signal import signal, SIGINT
from sys import exit
import os
import threading
import time
import json
from flask import Flask, Response, jsonify
from flask_cors import CORS
import amqp.amqp_handler as amqp_handler
from amqp.amqp import AmqpClient
from commons.esclient import EsClient
from utils import utils
from service.cluster_service import ClusterService
from service.auto_analyzer_service import AutoAnalyzerService
from service.suggest_service import SuggestService
from service.search_service import SearchService
from service.namespace_finder_service import NamespaceFinderService
from service.delete_index_service import DeleteIndexService
from service.retraining_service import RetrainingService


APP_CONFIG = {
    "esHost":            os.getenv("ES_HOSTS", "http://elasticsearch:9200").strip("/").strip("\\"),
    "logLevel":          os.getenv("LOGGING_LEVEL", "DEBUG").strip(),
    "amqpUrl":           os.getenv("AMQP_URL", "").strip("/").strip("\\"),
    "exchangeName":      os.getenv("AMQP_EXCHANGE_NAME", "analyzer"),
    "analyzerPriority":  int(os.getenv("ANALYZER_PRIORITY", "1")),
    "analyzerIndex":     json.loads(os.getenv("ANALYZER_INDEX", "true").lower()),
    "analyzerLogSearch": json.loads(os.getenv("ANALYZER_LOG_SEARCH", "true").lower()),
    "turnOffSslVerification": json.loads(os.getenv("ES_TURN_OFF_SSL_VERIFICATION", "false").lower()),
    "esVerifyCerts":     json.loads(os.getenv("ES_VERIFY_CERTS", "false").lower()),
    "esUseSsl":          json.loads(os.getenv("ES_USE_SSL", "false").lower()),
    "esSslShowWarn":     json.loads(os.getenv("ES_SSL_SHOW_WARN", "false").lower()),
    "esCAcert":          os.getenv("ES_CA_CERT", ""),
    "esClientCert":      os.getenv("ES_CLIENT_CERT", ""),
    "esClientKey":       os.getenv("ES_CLIENT_KEY", ""),
    "minioHost":         os.getenv("MINIO_SHORT_HOST", "minio:9000"),
    "minioAccessKey":    os.getenv("MINIO_ACCESS_KEY", "minio"),
    "minioSecretKey":    os.getenv("MINIO_SECRET_KEY", "minio123"),
    "appVersion":        ""
}

SEARCH_CONFIG = {
    "MinShouldMatch":              os.getenv("ES_MIN_SHOULD_MATCH", "80%"),
    "BoostAA":                     float(os.getenv("ES_BOOST_AA", "-8.0")),
    "BoostLaunch":                 float(os.getenv("ES_BOOST_LAUNCH", "4.0")),
    "BoostUniqueID":               float(os.getenv("ES_BOOST_UNIQUE_ID", "8.0")),
    "MaxQueryTerms":               int(os.getenv("ES_MAX_QUERY_TERMS", "50")),
    "SearchLogsMinSimilarity":     float(os.getenv("ES_LOGS_MIN_SHOULD_MATCH", "0.98")),
    "MinWordLength":               int(os.getenv("ES_MIN_WORD_LENGTH", "2")),
    "BoostModelFolder":            "",
    "SuggestBoostModelFolder":     "",
    "SimilarityWeightsFolder":     "",
    "GlobalDefectTypeFolder":      ""
}


def create_application():
    """Creates a Flask application"""
    _application = Flask(__name__)
    return _application


def create_thread(func, args):
    """Creates a thread with specified function and arguments"""
    thread = threading.Thread(target=func, args=args)
    thread.start()
    return thread


def create_es_client():
    """Creates Elasticsearch client"""
    return EsClient(APP_CONFIG, SEARCH_CONFIG)


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
                                     "version":             APP_CONFIG["appVersion"], })
    except Exception as err:
        logger.error("Failed to declare exchange")
        logger.error(err)
        return False
    logger.info("Exchange '%s' has been declared", config["exchangeName"])
    return True


def init_amqp(_amqp_client):
    """Initialize rabbitmq queues, exchange and stars threads for queue messages processing"""
    with _amqp_client.connection.channel() as channel:
        try:
            declare_exchange(channel, APP_CONFIG)
        except Exception as err:
            logger.error("Failed to declare amqp objects")
            logger.error(err)
            return
    threads = []
    es_client = create_es_client()
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "index", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    es_client.index_logs,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_index_response_data))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "analyze", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    AutoAnalyzerService(
                                                        APP_CONFIG,
                                                        SEARCH_CONFIG).analyze_logs,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_analyze_response_data))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "delete", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    DeleteIndexService(
                                                        APP_CONFIG, SEARCH_CONFIG).delete_index,
                                                    prepare_data_func=amqp_handler.
                                                    prepare_delete_index,
                                                    prepare_response_data=amqp_handler.
                                                    output_result))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "clean", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    es_client.delete_logs,
                                                    prepare_data_func=amqp_handler.
                                                    prepare_clean_index,
                                                    prepare_response_data=amqp_handler.
                                                    output_result))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "search", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    SearchService(APP_CONFIG, SEARCH_CONFIG).search_logs,
                                                    prepare_data_func=amqp_handler.
                                                    prepare_search_logs,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_analyze_response_data))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "suggest", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    SuggestService(APP_CONFIG, SEARCH_CONFIG).suggest_items,
                                                    prepare_data_func=amqp_handler.
                                                    prepare_test_item_info,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_analyze_response_data))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "cluster", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    ClusterService(APP_CONFIG, SEARCH_CONFIG).find_clusters,
                                                    prepare_data_func=amqp_handler.
                                                    prepare_launch_info,
                                                    prepare_response_data=amqp_handler.
                                                    prepare_analyze_response_data))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "stats_info", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_inner_amqp_request(channel, method, props, body,
                                                          es_client.send_stats_info))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "namespace_finder", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_amqp_request(channel, method, props, body,
                                                    NamespaceFinderService(
                                                        APP_CONFIG,
                                                        SEARCH_CONFIG).update_chosen_namespaces))))
    threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                   (APP_CONFIG["exchangeName"], "train_models", True, False,
                   lambda channel, method, props, body:
                   amqp_handler.handle_inner_amqp_request(channel, method, props, body,
                                                          RetrainingService(
                                                              APP_CONFIG,
                                                              SEARCH_CONFIG).train_models))))

    return threads


def read_version():
    """Reads the application build version"""
    version_filename = "VERSION"
    if os.path.exists(version_filename):
        with open(version_filename, "r") as file:
            return file.read().strip()
    return ""


def read_model_settings():
    """Reads paths to models"""
    model_settings = utils.read_json_file("", "model_settings.json", to_json=True)
    SEARCH_CONFIG["BoostModelFolder"] = model_settings["BOOST_MODEL_FOLDER"]
    SEARCH_CONFIG["SuggestBoostModelFolder"] = model_settings["SUGGEST_BOOST_MODEL_FOLDER"]
    SEARCH_CONFIG["SimilarityWeightsFolder"] = model_settings["SIMILARITY_WEIGHTS_FOLDER"]
    SEARCH_CONFIG["GlobalDefectTypeModelFolder"] = model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]


log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)
if APP_CONFIG["logLevel"].lower() == "debug":
    logging.disable(logging.NOTSET)
elif APP_CONFIG["logLevel"].lower() == "info":
    logging.disable(logging.DEBUG)
else:
    logging.disable(logging.INFO)
logger = logging.getLogger("analyzerApp")
APP_CONFIG["appVersion"] = read_version()
read_model_settings()

application = create_application()
CORS(application)
threads = []


@application.route('/', methods=['GET'])
def get_health_status():
    global threads
    status = ""
    if not utils.is_healthy(APP_CONFIG["esHost"]):
        status += "Elasticsearch is not healthy;"
    if status:
        logger.error("Analyzer health check status failed: %s", status)
        return Response(json.dumps({"status": status}), status=503, mimetype='application/json')
    return jsonify({"status": "healthy"})


def handler(signal_received, frame):
    print('The analyzer has stopped')
    exit(0)


def start_http_server():
    application.logger.setLevel(logging.INFO)
    logger.info("Started http server")
    application.run(host='0.0.0.0', port=5001, use_reloader=False)


signal(SIGINT, handler)
threads = []
logger.info("The analyzer has started")
while True:
    try:
        logger.info("Starting waiting for AMQP connection")
        try:
            amqp_client = AmqpClient(APP_CONFIG["amqpUrl"])
        except Exception as err:
            logger.error("Amqp connection was not established")
            logger.error(err)
            time.sleep(10)
            continue
        threads = init_amqp(amqp_client)
        logger.info("Analyzer has started")
        break
    except Exception as err:
        logger.error("The analyzer has failed")
        logger.error(err)


if __name__ == '__main__':
    logger.info("Program started")

    start_http_server()

    logger.info("The analyzer has finished")
    exit(0)
