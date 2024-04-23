#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json

import logging.config
import os
import threading
import time
from signal import signal, SIGINT
from sys import exit

from flask import Flask, Response, jsonify
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect

from app.amqp import amqp_handler
from app.amqp.amqp import AmqpClient
from app.commons import model_chooser, logging as my_logging
from app.commons.esclient import EsClient
from app.service import AnalyzerService
from app.service import AutoAnalyzerService
from app.service import CleanIndexService
from app.service import ClusterService
from app.service import DeleteIndexService
from app.service import NamespaceFinderService
from app.service import RetrainingService
from app.service import SearchService
from app.service import SuggestInfoService
from app.service import SuggestPatternsService
from app.service import SuggestService
from app.utils import utils

APP_CONFIG = {
    "esHost": os.getenv("ES_HOSTS", "http://elasticsearch:9200").strip("/").strip("\\"),
    "esUser": os.getenv("ES_USER", "").strip(),
    "esPassword": os.getenv("ES_PASSWORD", "").strip(),
    "logLevel": os.getenv("LOGGING_LEVEL", "DEBUG").strip(),
    "amqpUrl": os.getenv("AMQP_URL", "").strip("/").strip("\\") + "/" + os.getenv(
        "AMQP_VIRTUAL_HOST", "analyzer"),
    "exchangeName": os.getenv("AMQP_EXCHANGE_NAME", "analyzer"),
    "analyzerPriority": int(os.getenv("ANALYZER_PRIORITY", "1")),
    "analyzerIndex": json.loads(os.getenv("ANALYZER_INDEX", "true").lower()),
    "analyzerLogSearch": json.loads(os.getenv("ANALYZER_LOG_SEARCH", "true").lower()),
    "analyzerSuggest": json.loads(os.getenv("ANALYZER_SUGGEST", "true").lower()),
    "analyzerCluster": json.loads(os.getenv("ANALYZER_CLUSTER", "true").lower()),
    "turnOffSslVerification": json.loads(os.getenv("ES_TURN_OFF_SSL_VERIFICATION", "false").lower()),
    "esVerifyCerts": json.loads(os.getenv("ES_VERIFY_CERTS", "false").lower()),
    "esUseSsl": json.loads(os.getenv("ES_USE_SSL", "false").lower()),
    "esSslShowWarn": json.loads(os.getenv("ES_SSL_SHOW_WARN", "false").lower()),
    "esCAcert": os.getenv("ES_CA_CERT", ""),
    "esClientCert": os.getenv("ES_CLIENT_CERT", ""),
    "esClientKey": os.getenv("ES_CLIENT_KEY", ""),
    "minioHost": os.getenv("MINIO_SHORT_HOST", "minio:9000"),
    "minioAccessKey": os.getenv("MINIO_ACCESS_KEY", "minio"),
    "minioSecretKey": os.getenv("MINIO_SECRET_KEY", "minio123"),
    "minioUseTls": json.loads(os.getenv("MINIO_USE_TLS", "false").lower()),
    "appVersion": "",
    "binaryStoreType": os.getenv("ANALYZER_BINSTORE_TYPE",
                                 os.getenv("ANALYZER_BINARYSTORE_TYPE", "filesystem")),
    "minioBucketPrefix": os.getenv("ANALYZER_BINSTORE_BUCKETPREFIX",
                                   os.getenv("ANALYZER_BINARYSTORE_BUCKETPREFIX", "prj-")),
    "minioRegion": os.getenv("ANALYZER_BINSTORE_MINIO_REGION",
                             os.getenv("ANALYZER_BINARYSTORE_MINIO_REGION", None)),
    "instanceTaskType": os.getenv("INSTANCE_TASK_TYPE", "").strip(),
    "filesystemDefaultPath": os.getenv("FILESYSTEM_DEFAULT_PATH", "storage").strip(),
    "esChunkNumber": int(os.getenv("ES_CHUNK_NUMBER", "1000")),
    "esChunkNumberUpdateClusters": int(os.getenv("ES_CHUNK_NUMBER_UPDATE_CLUSTERS", "500")),
    "esProjectIndexPrefix": os.getenv("ES_PROJECT_INDEX_PREFIX", "").strip(),
    "analyzerHttpPort": int(os.getenv("ANALYZER_HTTP_PORT", "5001")),
    "analyzerPathToLog": os.getenv("ANALYZER_FILE_LOGGING_PATH", "/tmp/config.log")
}

SEARCH_CONFIG = {
    "SearchLogsMinSimilarity": float(os.getenv("ES_LOGS_MIN_SHOULD_MATCH", "0.95")),
    "MinShouldMatch": os.getenv("ES_MIN_SHOULD_MATCH", "80%"),
    "BoostAA": float(os.getenv("ES_BOOST_AA", "-8.0")),
    "BoostLaunch": float(os.getenv("ES_BOOST_LAUNCH", "4.0")),
    "BoostTestCaseHash": float(os.getenv("ES_BOOST_TEST_CASE_HASH", "8.0")),
    "MaxQueryTerms": int(os.getenv("ES_MAX_QUERY_TERMS", "50")),
    "MinWordLength": int(os.getenv("ES_MIN_WORD_LENGTH", "2")),
    "TimeWeightDecay": float(os.getenv("ES_TIME_WEIGHT_DECAY", "0.95")),
    "PatternLabelMinPercentToSuggest": float(os.getenv("PATTERN_LABEL_MIN_PERCENT", "0.9")),
    "PatternLabelMinCountToSuggest": int(os.getenv("PATTERN_LABEL_MIN_COUNT", "5")),
    "PatternMinCountToSuggest": int(os.getenv("PATTERN_MIN_COUNT", "10")),
    "MaxLogsForDefectTypeModel": int(os.getenv("MAX_LOGS_FOR_DEFECT_TYPE_MODEL", "10000")),
    "ProbabilityForCustomModelSuggestions": min(
        0.8, float(os.getenv("PROB_CUSTOM_MODEL_SUGGESTIONS", "0.7"))),
    "ProbabilityForCustomModelAutoAnalysis": min(
        1.0, float(os.getenv("PROB_CUSTOM_MODEL_AUTO_ANALYSIS", "0.5"))),
    "BoostModelFolder": "",
    "SuggestBoostModelFolder": "",
    "SimilarityWeightsFolder": "",
    "GlobalDefectTypeModelFolder": "",
    "RetrainSuggestBoostModelConfig": "",
    "RetrainAutoBoostModelConfig": "",
    "MaxSuggestionsNumber": int(os.getenv("MAX_SUGGESTIONS_NUMBER", "3")),
    "AutoAnalysisTimeout": int(os.getenv("AUTO_ANALYSIS_TIMEOUT", "300")),
    "MaxAutoAnalysisItemsToProcess": int(
        os.getenv("ANALYZER_MAX_ITEMS_TO_PROCESS", os.getenv("MAX_AUTO_ANALYSIS_ITEMS_TO_PROCESS", "4000")))
}


def create_application():
    """Creates a Flask application"""
    _application = Flask(__name__)
    CORS(_application)
    CSRFProtect(_application)
    return _application


def create_thread(func, args):
    """Creates a thread with specified function and arguments"""
    thread = threading.Thread(target=func, args=args, daemon=True)
    thread.start()
    return thread


def declare_exchange(channel, config):
    """Declares exchange for rabbitmq"""
    logger.info("ExchangeName: %s", config["exchangeName"])
    try:
        channel.exchange_declare(exchange=config["exchangeName"], exchange_type='direct',
                                 durable=False, auto_delete=True, internal=False,
                                 arguments={
                                     "analyzer": config["exchangeName"],
                                     "analyzer_index": config["analyzerIndex"],
                                     "analyzer_priority": config["analyzerPriority"],
                                     "analyzer_log_search": config["analyzerLogSearch"],
                                     "analyzer_suggest": config["analyzerSuggest"],
                                     "analyzer_cluster": config["analyzerCluster"],
                                     "version": config["appVersion"], })
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
    _threads = []
    _model_chooser = model_chooser.ModelChooser(APP_CONFIG, SEARCH_CONFIG)
    if APP_CONFIG["instanceTaskType"] == "train":
        _retraining_service = RetrainingService(_model_chooser, APP_CONFIG, SEARCH_CONFIG)
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "train_models", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_inner_amqp_request(
                                           current_channel, method, props, body,
                                           _retraining_service.train_models))))
    else:
        _es_client = EsClient(APP_CONFIG, SEARCH_CONFIG)
        _auto_analyzer_service = AutoAnalyzerService(_model_chooser, APP_CONFIG, SEARCH_CONFIG)
        _delete_index_service = DeleteIndexService(_model_chooser, APP_CONFIG, SEARCH_CONFIG)
        _clean_index_service = CleanIndexService(APP_CONFIG, SEARCH_CONFIG)
        _analyzer_service = AnalyzerService(_model_chooser, SEARCH_CONFIG)
        _suggest_service = SuggestService(_model_chooser, APP_CONFIG, SEARCH_CONFIG)
        _suggest_info_service = SuggestInfoService(APP_CONFIG, SEARCH_CONFIG)
        _search_service = SearchService(APP_CONFIG, SEARCH_CONFIG)
        _cluster_service = ClusterService(APP_CONFIG, SEARCH_CONFIG)
        _namespace_finder_service = NamespaceFinderService(APP_CONFIG, SEARCH_CONFIG)
        _suggest_patterns_service = SuggestPatternsService(APP_CONFIG, SEARCH_CONFIG)
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "index", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _es_client.index_logs,
                                                                        prepare_response_data=amqp_handler.
                                                                        prepare_index_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "analyze", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _auto_analyzer_service.analyze_logs,
                                                                        prepare_response_data=amqp_handler.
                                                                        prepare_analyze_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "delete", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _delete_index_service.delete_index,
                                                                        prepare_data_func=amqp_handler.
                                                                        prepare_delete_index,
                                                                        prepare_response_data=amqp_handler.
                                                                        output_result))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "clean", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _clean_index_service.delete_logs,
                                                                        prepare_data_func=amqp_handler.
                                                                        prepare_clean_index,
                                                                        prepare_response_data=amqp_handler.
                                                                        output_result))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "search", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _search_service.search_logs,
                                                                        prepare_data_func=amqp_handler.
                                                                        prepare_search_logs,
                                                                        prepare_response_data=amqp_handler.
                                                                        prepare_analyze_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "suggest", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _suggest_service.suggest_items,
                                                                        prepare_data_func=amqp_handler.
                                                                        prepare_test_item_info,
                                                                        prepare_response_data=amqp_handler.
                                                                        prepare_analyze_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "cluster", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _cluster_service.find_clusters,
                                                                        prepare_data_func=amqp_handler.
                                                                        prepare_launch_info,
                                                                        prepare_response_data=amqp_handler.
                                                                        prepare_index_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "stats_info", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_inner_amqp_request(current_channel, method, props,
                                                                              body,
                                                                              _es_client.send_stats_info))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "namespace_finder", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(
                                           current_channel, method, props, body,
                                           _namespace_finder_service.update_chosen_namespaces,
                                           publish_result=False))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "suggest_patterns", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(
                                           current_channel, method, props, body,
                                           _suggest_patterns_service.suggest_patterns,
                                           prepare_data_func=amqp_handler.prepare_delete_index,
                                           prepare_response_data=amqp_handler.prepare_index_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "index_suggest_info", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(
                                           current_channel, method, props, body,
                                           _suggest_info_service.index_suggest_info,
                                           prepare_data_func=amqp_handler.prepare_suggest_info_list,
                                           prepare_response_data=amqp_handler.prepare_index_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "remove_suggest_info", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(
                                           current_channel, method, props, body,
                                           _suggest_info_service.remove_suggest_info,
                                           prepare_data_func=amqp_handler.prepare_delete_index,
                                           prepare_response_data=amqp_handler.output_result))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "update_suggest_info", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(
                                           current_channel, method, props, body,
                                           _suggest_info_service.update_suggest_info,
                                           prepare_data_func=lambda x: x))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "remove_models", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _analyzer_service.remove_models,
                                                                        prepare_data_func=lambda x: x,
                                                                        prepare_response_data=amqp_handler.
                                                                        output_result))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "get_model_info", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _analyzer_service.get_model_info,
                                                                        prepare_data_func=lambda x: x))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "defect_update", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _es_client.defect_update,
                                                                        prepare_data_func=lambda x: x,
                                                                        prepare_response_data=amqp_handler.
                                                                        prepare_search_response_data))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "item_remove", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(
                                           current_channel, method, props, body,
                                           _clean_index_service.delete_test_items,
                                           prepare_data_func=lambda x: x,
                                           prepare_response_data=amqp_handler.output_result))))
        _threads.append(create_thread(AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                                      (APP_CONFIG["exchangeName"], "launch_remove", True, False,
                                       lambda current_channel, method, props, body:
                                       amqp_handler.handle_amqp_request(current_channel, method, props, body,
                                                                        _clean_index_service.delete_launches,
                                                                        prepare_data_func=lambda x: x,
                                                                        prepare_response_data=amqp_handler.
                                                                        output_result))))
        _threads.append(
            create_thread(
                AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                (
                    APP_CONFIG["exchangeName"],
                    "remove_by_launch_start_time",
                    True,
                    False,
                    lambda current_channel, method, props, body: amqp_handler.handle_amqp_request(
                        current_channel,
                        method,
                        props,
                        body,
                        _clean_index_service.remove_by_launch_start_time,
                        prepare_data_func=lambda x: x,
                        prepare_response_data=amqp_handler.output_result,
                    ),
                ),
            )
        )
        _threads.append(
            create_thread(
                AmqpClient(APP_CONFIG["amqpUrl"]).receive,
                (
                    APP_CONFIG["exchangeName"],
                    "remove_by_log_time",
                    True,
                    False,
                    lambda current_channel, method, props, body: amqp_handler.handle_amqp_request(
                        current_channel,
                        method,
                        props,
                        body,
                        _clean_index_service.remove_by_log_time,
                        prepare_data_func=lambda x: x,
                        prepare_response_data=amqp_handler.output_result,
                    ),
                ),
            )
        )

    return _threads


def read_version():
    """Reads the application build version"""
    version_filename = "VERSION"
    if os.path.exists(version_filename):
        with open(version_filename, "r") as file:
            return file.read().strip()
    return ""


def read_model_settings():
    """Reads paths to models"""
    model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
    SEARCH_CONFIG["BoostModelFolder"] = model_settings["BOOST_MODEL_FOLDER"]
    SEARCH_CONFIG["SuggestBoostModelFolder"] = model_settings["SUGGEST_BOOST_MODEL_FOLDER"]
    SEARCH_CONFIG["SimilarityWeightsFolder"] = model_settings["SIMILARITY_WEIGHTS_FOLDER"]
    SEARCH_CONFIG["GlobalDefectTypeModelFolder"] = model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]
    SEARCH_CONFIG["RetrainSuggestBoostModelConfig"] = model_settings["RETRAIN_SUGGEST_BOOST_MODEL_CONFIG"]
    SEARCH_CONFIG["RetrainAutoBoostModelConfig"] = model_settings["RETRAIN_AUTO_BOOST_MODEL_CONFIG"]


log_file_path = 'res/logging.conf'
logging.config.fileConfig(log_file_path, defaults={'logfilename': APP_CONFIG["analyzerPathToLog"]})
if APP_CONFIG["logLevel"].lower() == "debug":
    logging.disable(logging.NOTSET)
elif APP_CONFIG["logLevel"].lower() == "info":
    logging.disable(logging.DEBUG)
else:
    logging.disable(logging.INFO)
logger = my_logging.getLogger("analyzerApp")
APP_CONFIG["appVersion"] = read_version()
es_client = EsClient(APP_CONFIG, SEARCH_CONFIG)
read_model_settings()

application = create_application()
threads = []


@application.route('/', methods=['GET'])
def get_health_status():
    status = ""
    if not es_client.is_healthy():
        status += "Elasticsearch is not healthy;"
    if status:
        logger.error("Analyzer health check status failed: %s", status)
        return Response(json.dumps({"status": status}), status=503, mimetype='application/json')
    return jsonify({"status": "healthy"})


# noinspection PyUnusedLocal
def handler(signal_received, frame):
    print('The analyzer has stopped')
    exit(0)


def start_http_server():
    application.logger.setLevel(logging.INFO)
    logger.info("Started http server")
    application.run(host='0.0.0.0', port=APP_CONFIG["analyzerHttpPort"], use_reloader=False)


signal(SIGINT, handler)
logger.info("The analyzer has started")
while True:
    try:
        logger.info("Starting waiting for AMQP connection")
        try:
            amqp_client = AmqpClient(APP_CONFIG["amqpUrl"])
        except Exception as exc:
            logger.error("Amqp connection was not established")
            logger.exception(exc)
            time.sleep(10)
            continue
        threads = init_amqp(amqp_client)
        logger.info("Analyzer has started")
        break
    except Exception as exc:
        logger.error("The analyzer has failed")
        logger.exception(exc)

if __name__ == '__main__':
    logger.info("Program started")

    start_http_server()

    logger.info("The analyzer has finished")
    exit(0)
