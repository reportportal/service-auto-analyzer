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
from signal import SIGINT, signal
from sys import exit
from typing import Any

from commons.model.ml import ModelType
from flask import Flask, Response
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect

from app.amqp.amqp import AmqpClient
from app.amqp.amqp_handler import AmqpRequestHandler, DirectAmqpRequestHandler, ProcessAmqpRequestHandler
from app.commons import logging as my_logging
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.utils import utils

APP_CONFIG = ApplicationConfig(
    # Mute Sonar about hardcoded HTTP URL, since this is a hostname inside a docker-compose file
    esHost=os.getenv("ES_HOSTS", "http://elasticsearch:9200").strip("/").strip("\\"),  # NOSONAR
    esUser=os.getenv("ES_USER", "").strip(),
    esPassword=os.getenv("ES_PASSWORD", "").strip(),
    logLevel=os.getenv("LOGGING_LEVEL", "DEBUG").strip(),
    debugMode=json.loads(os.getenv("DEBUG_MODE", "false").lower()),
    amqpUrl=os.getenv("AMQP_URL", "").strip("/").strip("\\") + "/" + os.getenv("AMQP_VIRTUAL_HOST", "analyzer"),
    amqpExchangeName=os.getenv("AMQP_EXCHANGE_NAME", "analyzer"),
    amqpInitialRetryInterval=int(os.getenv("AMQP_INITIAL_RETRY_INTERVAL", "1")),
    amqpMaxRetryTime=int(os.getenv("AMQP_MAX_RETRY_TIME", "300")),
    amqpHeartbeatInterval=int(os.getenv("AMQP_HEARTBEAT_INTERVAL", "30")),
    amqpBackoffFactor=int(os.getenv("AMQP_BACKOFF_FACTOR", "2")),
    amqpHandlerMaxRetries=int(os.getenv("AMQP_HANDLER_MAX_RETRIES", "3")),
    amqpHandlerTaskTimeout=int(os.getenv("AMQP_HANDLER_TASK_TIMEOUT", "600")),
    analyzerPriority=int(os.getenv("ANALYZER_PRIORITY", "1")),
    analyzerIndex=json.loads(os.getenv("ANALYZER_INDEX", "true").lower()),
    analyzerLogSearch=json.loads(os.getenv("ANALYZER_LOG_SEARCH", "true").lower()),
    analyzerSuggest=json.loads(os.getenv("ANALYZER_SUGGEST", "true").lower()),
    analyzerCluster=json.loads(os.getenv("ANALYZER_CLUSTER", "true").lower()),
    turnOffSslVerification=json.loads(os.getenv("ES_TURN_OFF_SSL_VERIFICATION", "false").lower()),
    esVerifyCerts=json.loads(os.getenv("ES_VERIFY_CERTS", "false").lower()),
    esUseSsl=json.loads(os.getenv("ES_USE_SSL", "false").lower()),
    esSslShowWarn=json.loads(os.getenv("ES_SSL_SHOW_WARN", "false").lower()),
    esCAcert=os.getenv("ES_CA_CERT", ""),
    esClientCert=os.getenv("ES_CLIENT_CERT", ""),
    esClientKey=os.getenv("ES_CLIENT_KEY", ""),
    minioHost=os.getenv("MINIO_SHORT_HOST", "minio:9000"),
    minioAccessKey=os.getenv("MINIO_ACCESS_KEY", "minio"),
    minioSecretKey=os.getenv("MINIO_SECRET_KEY", "minio123"),
    minioUseTls=json.loads(os.getenv("MINIO_USE_TLS", "false").lower()),
    appVersion="",
    binaryStoreType=os.getenv("ANALYZER_BINSTORE_TYPE", os.getenv("ANALYZER_BINARYSTORE_TYPE", "filesystem")),
    bucketPrefix=os.getenv("ANALYZER_BINSTORE_BUCKETPREFIX", os.getenv("ANALYZER_BINARYSTORE_BUCKETPREFIX", "prj-")),
    minioRegion=os.getenv("ANALYZER_BINSTORE_MINIO_REGION", os.getenv("ANALYZER_BINARYSTORE_MINIO_REGION", None)),
    instanceTaskType=os.getenv("INSTANCE_TASK_TYPE", "").strip(),
    filesystemDefaultPath=os.getenv("FILESYSTEM_DEFAULT_PATH", "storage").strip(),
    esChunkNumber=int(os.getenv("ES_CHUNK_NUMBER", "1000")),
    esChunkNumberUpdateClusters=int(os.getenv("ES_CHUNK_NUMBER_UPDATE_CLUSTERS", "500")),
    esProjectIndexPrefix=os.getenv("ES_PROJECT_INDEX_PREFIX", "").strip(),
    analyzerHttpPort=int(os.getenv("ANALYZER_HTTP_PORT", "5001")),
    analyzerPathToLog=os.getenv("ANALYZER_FILE_LOGGING_PATH", "/tmp/config.log"),
)

SEARCH_CONFIG = SearchConfig(
    SearchLogsMinSimilarity=float(os.getenv("ES_LOGS_MIN_SHOULD_MATCH", "0.95")),
    MinShouldMatch=os.getenv("ES_MIN_SHOULD_MATCH", "80%"),
    BoostAA=float(os.getenv("ES_BOOST_AA", "0.0")),
    BoostMA=float(os.getenv("ES_BOOST_MA", "10.0")),
    BoostLaunch=float(os.getenv("ES_BOOST_LAUNCH", "4.0")),
    BoostTestCaseHash=float(os.getenv("ES_BOOST_TEST_CASE_HASH", "8.0")),
    MaxQueryTerms=int(os.getenv("ES_MAX_QUERY_TERMS", "50")),
    MinWordLength=int(os.getenv("ES_MIN_WORD_LENGTH", "2")),
    TimeWeightDecay=float(os.getenv("ES_TIME_WEIGHT_DECAY", "0.999")),
    PatternLabelMinPercentToSuggest=float(os.getenv("PATTERN_LABEL_MIN_PERCENT", "0.9")),
    PatternLabelMinCountToSuggest=int(os.getenv("PATTERN_LABEL_MIN_COUNT", "5")),
    PatternMinCountToSuggest=int(os.getenv("PATTERN_MIN_COUNT", "10")),
    MaxLogsForDefectTypeModel=int(os.getenv("MAX_LOGS_FOR_DEFECT_TYPE_MODEL", "10000")),
    ProbabilityForCustomModelSuggestions=min(0.8, float(os.getenv("PROB_CUSTOM_MODEL_SUGGESTIONS", "0.7"))),
    ProbabilityForCustomModelAutoAnalysis=min(1.0, float(os.getenv("PROB_CUSTOM_MODEL_AUTO_ANALYSIS", "0.5"))),
    BoostModelFolder="",
    SuggestBoostModelFolder="",
    SimilarityWeightsFolder="",
    GlobalDefectTypeModelFolder="",
    MaxSuggestionsNumber=int(os.getenv("MAX_SUGGESTIONS_NUMBER", "3")),
    AutoAnalysisTimeout=int(os.getenv("AUTO_ANALYSIS_TIMEOUT", "300")),
    MaxAutoAnalysisItemsToProcess=int(
        os.getenv("ANALYZER_MAX_ITEMS_TO_PROCESS", os.getenv("MAX_AUTO_ANALYSIS_ITEMS_TO_PROCESS", "4000"))
    ),
    MlModelForSuggestions=ModelType[os.getenv("ML_MODEL_FOR_SUGGESTIONS", "suggestion").strip().lower()].value,
)


def create_application():
    """Creates a Flask application"""
    _application = Flask(__name__)
    CORS(_application)
    CSRFProtect(_application)
    return _application


def create_thread(func, args, name: str):
    """Creates a thread with specified function and arguments"""
    thread = threading.Thread(target=func, args=args, name=name, daemon=True)
    thread.start()
    return thread


def except_train(request: str) -> bool:
    """Filters out requests that are for training models"""
    return request != "train_models"


def only_train(request: str) -> bool:
    """Filters out requests that are not related to training models"""
    return request == "train_models"


def init_amqp_queues():
    """Initialize rabbitmq queues, exchange and starts threads for queue messages processing"""
    _threads = []

    if APP_CONFIG.debugMode:
        handler_class = DirectAmqpRequestHandler
    else:
        handler_class = ProcessAmqpRequestHandler

    _main_amqp_handler = handler_class(
        APP_CONFIG,
        SEARCH_CONFIG,
        routing_key_predicate=except_train,
        name="main_handler",
        init_services=[
            "index",
            "analyze",
            "delete",
            "clean",
            "search",
            "suggest",
            "cluster",
            "stats_info",
            "namespace_finder",
            "suggest_patterns",
            "index_suggest_info",
            "remove_suggest_info",
            "update_suggest_info",
            "remove_models",
            "get_model_info",
            "defect_update",
            "item_remove",
            "launch_remove",
            "remove_by_launch_start_time",
            "remove_by_log_time",
        ],
    )
    _train_amqp_handler = handler_class(
        APP_CONFIG,
        SEARCH_CONFIG,
        routing_key_predicate=only_train,
        name="train_handler",
        init_services=["train_models"],
    )

    _threads.append(
        (
            "all",
            create_thread(
                AmqpClient(APP_CONFIG).receive,
                (
                    "all",
                    _main_amqp_handler.handle_amqp_request,
                    None,
                ),
                "all",
            ),
            _main_amqp_handler,
        )
    )
    _threads.append(
        (
            "train",
            create_thread(
                AmqpClient(APP_CONFIG).receive,
                (
                    "train",
                    _train_amqp_handler.handle_amqp_request,
                    None,
                ),
                "train",
            ),
            _train_amqp_handler,
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
    if not model_settings or not isinstance(model_settings, dict):
        raise RuntimeError("Failed to read model settings")

    SEARCH_CONFIG.BoostModelFolder = utils.strip_path(model_settings["BOOST_MODEL_FOLDER"])
    SEARCH_CONFIG.SuggestBoostModelFolder = utils.strip_path(model_settings["SUGGEST_BOOST_MODEL_FOLDER"])
    SEARCH_CONFIG.SimilarityWeightsFolder = utils.strip_path(model_settings["SIMILARITY_WEIGHTS_FOLDER"])
    SEARCH_CONFIG.GlobalDefectTypeModelFolder = utils.strip_path(model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"])


my_logging.setup(APP_CONFIG)
logger = my_logging.getLogger("analyzerApp")
APP_CONFIG.appVersion = read_version()
es_client = EsClient(APP_CONFIG)
read_model_settings()

application = create_application()
THREADS: list[tuple[str, threading.Thread, AmqpRequestHandler]] = []


@application.route("/", methods=["GET"])
def get_health_status():
    status: dict[str, Any] = {"status": "healthy"}
    status_code = 200
    if not es_client.is_healthy():
        logger.error("Analyzer health check status failed: %s", status)
        status["status"] = "Elasticsearch is not healthy"
        status_code = 503
    if THREADS:
        status["threads"] = []
        for thread_name, thread, handler in THREADS:
            if not thread.is_alive():
                status["threads"].append({"name": thread_name, "status": "not alive"})
            else:
                thread_status: dict[str, Any] = {
                    "name": thread_name,
                    "status": "alive",
                    "pid": handler.processor.pid,
                }
                status["threads"].append(thread_status)
                tasks = [
                    {
                        "routing_key": task.routing_key,
                        "correlation_id": task.msg_correlation_id,
                        "send_time": task.send_time,
                    }
                    for task in handler.running_tasks
                ]
                thread_status["running_tasks"] = {
                    "number": len(tasks),
                    "tasks": tasks,
                }

    return Response(json.dumps(status), status=status_code, mimetype="application/json")


# noinspection PyUnusedLocal
def signal_handler(signal_received, frame):
    print("The analyzer has stopped")
    exit(0)


def start_http_server():
    application.logger.setLevel(logging.INFO)
    logger.info("Started http server")
    application.run(host="0.0.0.0", port=APP_CONFIG.analyzerHttpPort, use_reloader=False)


signal(SIGINT, signal_handler)

# Run the application directly
if __name__ == "__main__":
    if APP_CONFIG.debugMode:
        logger.warning("The application is running in DEBUG mode, processing will run in threads.")
    else:
        logger.info("The application is running in PRODUCTION mode, processing will run in processes.")
    logger.info("Program started. Creating AMQP connections.")
    THREADS.extend(init_amqp_queues())
    logger.info("The analyzer has started.")

    start_http_server()

    logger.info("The analyzer has finished.")
    exit(0)
