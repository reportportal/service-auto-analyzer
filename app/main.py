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
import warnings
from signal import SIGINT, signal
from sys import exit
from typing import Any, Optional

from flask import Flask, Response
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect

from app.amqp.amqp import AmqpClient
from app.amqp.amqp_handler import AmqpRequestHandler, DirectAmqpRequestHandler, ProcessAmqpRequestHandler
from app.commons import logging as my_logging
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.ml import ModelType
from app.utils import utils


def to_bool(value: Optional[Any]) -> Optional[bool]:
    """Convert value of any type to boolean or raise ValueError.

    :param value: value to convert
    :return: boolean value
    :raises ValueError: if value is not boolean
    """
    if value is None or value == "":
        return None
    if value in {"TRUE", "True", "true", "1", "Y", "y", 1, True}:
        return True
    if value in {"FALSE", "False", "false", "0", "N", "n", 0, False}:
        return False
    raise ValueError(f"Invalid boolean value {value}.")


# Handle all datastore type settings, old and new
datastore_type = os.getenv("DATASTORE_TYPE")
old_datastore_type = os.getenv("ANALYZER_BINSTORE_TYPE")
if old_datastore_type:
    warnings.warn(
        "'ANALYZER_BINSTORE_TYPE' configuration variable is deprecated, use 'DATASTORE_TYPE' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_type:
        datastore_type = os.getenv("DATASTORE_TYPE")
    old_datastore_type = None
old_datastore_type = os.getenv("ANALYZER_BINARYSTORE_TYPE")
if old_datastore_type:
    warnings.warn(
        "'ANALYZER_BINARYSTORE_TYPE' configuration variable is deprecated, use 'DATASTORE_TYPE' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_type:
        datastore_type = old_datastore_type
if not datastore_type:
    datastore_type = "filesystem"


# Handle all datastore endpoint settings, old and new
datastore_endpoint = os.getenv("DATASTORE_ENDPOINT")
minio_short_host = os.getenv("MINIO_SHORT_HOST")
if minio_short_host:
    warnings.warn(
        "'MINIO_SHORT_HOST' configuration variable is deprecated, use 'DATASTORE_ENDPOINT' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
minio_use_tls = os.getenv("MINIO_USE_TLS")
if minio_use_tls:
    warnings.warn(
        "'MINIO_USE_TLS' configuration variable is deprecated, use 'DATASTORE_ENDPOINT' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
if not datastore_endpoint:
    if minio_short_host:
        use_tls = to_bool(minio_use_tls)
        datastore_endpoint = f"http{'s' if use_tls else ''}://{minio_short_host}"

# Handle all datastore region settings, old and new
datastore_region = os.getenv("DATASTORE_REGION")
old_datastore_region = os.getenv("ANALYZER_BINSTORE_MINIO_REGION")
if old_datastore_region:
    warnings.warn(
        "'ANALYZER_BINSTORE_MINIO_REGION' configuration variable is deprecated, use 'DATASTORE_REGION' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_region:
        datastore_region = old_datastore_region
    old_datastore_region = None
old_datastore_region = os.getenv("ANALYZER_BINARYSTORE_MINIO_REGION")
if old_datastore_region:
    warnings.warn(
        "'ANALYZER_BINARYSTORE_MINIO_REGION' configuration variable is deprecated, use 'DATASTORE_REGION' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_region:
        datastore_region = old_datastore_region

# Handle all datastore bucket prefix settings, old and new
datastore_default_bucket_name = os.getenv("DATASTORE_DEFAULTBUCKETNAME")
datastore_bucket_prefix = os.getenv("DATASTORE_BUCKETPREFIX")
old_datastore_bucketprefix = os.getenv("ANALYZER_BINSTORE_BUCKETPREFIX")
if old_datastore_bucketprefix:
    warnings.warn(
        "'ANALYZER_BINSTORE_BUCKETPREFIX' configuration variable is deprecated, use 'DATASTORE_BUCKETPREFIX' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_bucket_prefix:
        datastore_bucket_prefix = old_datastore_bucketprefix
    old_datastore_bucketprefix = None
old_datastore_bucketprefix = os.getenv("ANALYZER_BINARYSTORE_BUCKETPREFIX")
if old_datastore_bucketprefix:
    warnings.warn(
        "'ANALYZER_BINARYSTORE_BUCKETPREFIX' configuration variable is deprecated, use 'DATASTORE_BUCKETPREFIX' "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_bucket_prefix:
        datastore_bucket_prefix = old_datastore_bucketprefix
if datastore_default_bucket_name:
    if datastore_bucket_prefix:
        datastore_bucket_prefix = f"{datastore_default_bucket_name}/{datastore_bucket_prefix}"
    else:
        datastore_bucket_prefix = datastore_default_bucket_name
else:
    if not datastore_bucket_prefix:
        datastore_bucket_prefix = "prj-"

datastore_bucket_postfix = os.getenv("DATASTORE_BUCKETPOSTFIX", "")

# Handle all datastore access key settings, old and new
datastore_access_key = os.getenv("DATASTORE_ACCESSKEY")
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
if minio_access_key:
    warnings.warn(
        "'MINIO_ACCESS_KEY' configuration variable is deprecated, use 'DATASTORE_ACCESSKEY' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_access_key:
        datastore_access_key = minio_access_key

# Handle all datastore secret key settings, old and new
datastore_secret_key = os.getenv("DATASTORE_SECRETKEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
if minio_secret_key:
    warnings.warn(
        "'MINIO_SECRET_KEY' configuration variable is deprecated, use 'DATASTORE_SECRETKEY' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not datastore_secret_key:
        datastore_secret_key = minio_secret_key


APP_CONFIG = ApplicationConfig(
    # ES/OS settings
    # Mute Sonar about hardcoded HTTP URL, since this is a hostname inside a docker-compose file
    esHost=os.getenv("ES_HOSTS", "http://opensearch:9200").strip("/").strip("\\"),  # NOSONAR
    esUser=os.getenv("ES_USER", "").strip(),
    esPassword=os.getenv("ES_PASSWORD", "").strip(),
    esUseSsl=to_bool(os.getenv("ES_USE_SSL", "false")),
    esVerifyCerts=to_bool(os.getenv("ES_VERIFY_CERTS", "false")),
    esSslShowWarn=to_bool(os.getenv("ES_SSL_SHOW_WARN", "false")),
    esCAcert=os.getenv("ES_CA_CERT", ""),
    esClientCert=os.getenv("ES_CLIENT_CERT", ""),
    esClientKey=os.getenv("ES_CLIENT_KEY", ""),
    turnOffSslVerification=to_bool(os.getenv("ES_TURN_OFF_SSL_VERIFICATION", "false")),
    esChunkNumber=int(os.getenv("ES_CHUNK_NUMBER", "1000")),
    esChunkNumberUpdateClusters=int(os.getenv("ES_CHUNK_NUMBER_UPDATE_CLUSTERS", "500")),
    esProjectIndexPrefix=os.getenv("ES_PROJECT_INDEX_PREFIX", "").strip(),
    # AMQP settings
    amqpUrl=os.getenv("AMQP_URL", "").strip("/").strip("\\") + "/" + os.getenv("AMQP_VIRTUAL_HOST", "analyzer"),
    amqpExchangeName=os.getenv("AMQP_EXCHANGE_NAME", "analyzer"),
    amqpInitialRetryInterval=int(os.getenv("AMQP_INITIAL_RETRY_INTERVAL", "1")),
    amqpMaxRetryTime=int(os.getenv("AMQP_MAX_RETRY_TIME", "300")),
    amqpHeartbeatInterval=int(os.getenv("AMQP_HEARTBEAT_INTERVAL", "30")),
    amqpBackoffFactor=int(os.getenv("AMQP_BACKOFF_FACTOR", "2")),
    amqpHandlerMaxRetries=int(os.getenv("AMQP_HANDLER_MAX_RETRIES", "3")),
    amqpHandlerTaskTimeout=int(os.getenv("AMQP_HANDLER_TASK_TIMEOUT", "600")),
    analyzerPriority=int(os.getenv("ANALYZER_PRIORITY", "1")),
    analyzerIndex=to_bool(os.getenv("ANALYZER_INDEX", "true")),
    analyzerLogSearch=to_bool(os.getenv("ANALYZER_LOG_SEARCH", "true")),
    analyzerSuggest=to_bool(os.getenv("ANALYZER_SUGGEST", "true")),
    analyzerCluster=to_bool(os.getenv("ANALYZER_CLUSTER", "true")),
    appVersion="",
    # Storage settings
    datastoreType=datastore_type,
    datastoreEndpoint=datastore_endpoint,
    datastoreRegion=datastore_region,
    datastoreBucketPrefix=datastore_bucket_prefix,
    datastoreBucketPostfix=datastore_bucket_postfix,
    datastoreAccessKey=datastore_access_key,
    datastoreSecretKey=datastore_secret_key,
    filesystemDefaultPath=os.getenv("FILESYSTEM_DEFAULT_PATH", "storage").strip(),
    # HTTP endpoint settings
    analyzerHttpPort=int(os.getenv("ANALYZER_HTTP_PORT", "5001")),
    # Log settings
    analyzerPathToLog=os.getenv("ANALYZER_FILE_LOGGING_PATH", "/tmp/config.log"),
    logLevel=os.getenv("LOGGING_LEVEL", "DEBUG").strip(),
    # Debug settings, controls if AMQP handler runs in threaded mode to ease debugging
    debugMode=to_bool(os.getenv("DEBUG_MODE", "false")),
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
    GlobalDefectTypeModelFolder="",
    MaxSuggestionsNumber=int(os.getenv("MAX_SUGGESTIONS_NUMBER", "3")),
    MaxAutoAnalysisItemsToProcess=int(
        os.getenv("ANALYZER_MAX_ITEMS_TO_PROCESS", os.getenv("MAX_AUTO_ANALYSIS_ITEMS_TO_PROCESS", "4000"))
    ),
    MlModelForSuggestions=os.getenv("ML_MODEL_FOR_SUGGESTIONS", ModelType.suggestion.name).strip(),
)


def create_application():
    """Creates a Flask application"""
    _application = Flask(__name__)
    CORS(_application, resources={r"/*": {"origins": "*", "send_wildcard": "False"}})
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
        status["status"] = "OpenSearch is not healthy"
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
