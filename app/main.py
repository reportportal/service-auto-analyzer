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

"""Entry point of the Analyzer: runs the coordinators of the "all" and "train" queues and the health endpoint.

This module only defines functions at import time: worker processes start from a fork server, which imports the main
module again.
"""

import asyncio
import json
import os
import signal
import warnings
from sys import exit
from typing import Any, Optional

from app.amqp.async_amqp import AsyncAmqpClient
from app.commons import logging as my_logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.ml import ModelType
from app.coordinator.budget import IntakeBudget
from app.coordinator.coordinator import Coordinator, DiscardingCoordinator, LaneCoordinator
from app.coordinator.health import HealthServer, OpenSearchHealth
from app.coordinator.pool import InlinePool, TaskPool, WorkerPool
from app.utils import file_utils

ALL_SERVICES: list[str] = [
    "index",
    "analyze",
    "delete",
    "clean",
    "search",
    "suggest",
    "cluster",
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
]
TRAIN_SERVICES: list[str] = ["train_models"]

logger = my_logging.getLogger("analyzerApp")


def to_bool(value: Any) -> bool:
    """Convert value of any type to boolean or raise ValueError.

    :param value: value to convert
    :return: boolean value
    :raises ValueError: if value is not boolean
    """
    if value in {"TRUE", "True", "true", "1", "Y", "y", True}:
        return True
    if value in {"FALSE", "False", "false", "0", "N", "n", False}:
        return False
    raise ValueError(f"Invalid boolean value {value}.")


def parse_string_map(value: Optional[str]) -> dict[str, str]:
    """Parse a JSON object into a dictionary with string keys and values.

    :param value: JSON object as a string, empty or `None` values give an empty dictionary
    :return: The dictionary where all non-string values are converted to their JSON representation
    :raises ValueError: If the value is not a valid JSON object
    """
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON object expected, got: {type(parsed).__name__}")
    return {str(key): val if isinstance(val, str) else json.dumps(val) for key, val in parsed.items()}


def read_version() -> str:
    """Read the application build version.

    :return: The version from the `VERSION` file, or an empty string if there is no such file
    """
    version_filename = "VERSION"
    if os.path.exists(version_filename):
        with open(version_filename, "r") as file:
            return file.read().strip()
    return ""


def read_model_settings() -> dict[str, Any]:
    """Read the settings of the global models.

    :return: The content of `res/model_settings.json`
    """
    model_settings = file_utils.read_resource_file("model_settings.json", to_json=True)
    if not model_settings or not isinstance(model_settings, dict):
        raise RuntimeError("Failed to read model settings")
    return model_settings


def deprecation_message(name: str, replacement: str) -> str:
    """Message of the warning about a deprecated configuration variable.

    :param name: The deprecated variable
    :param replacement: The variable to use instead
    :return: The message
    """
    return f"'{name}' configuration variable is deprecated, use '{replacement}' instead."


def getenv_with_deprecated(name: str, *deprecated_names: str) -> Optional[str]:
    """Read a configuration variable which has deprecated names: the current name wins, then the deprecated names in
    the given order. Each deprecated name which is set gives a warning.

    :param name: The current variable name
    :param deprecated_names: The deprecated variable names
    :return: The value, or None if no variable is set
    """
    value = os.getenv(name)
    for deprecated_name in deprecated_names:
        deprecated_value = os.getenv(deprecated_name)
        if not deprecated_value:
            continue
        warnings.warn(deprecation_message(deprecated_name, name), DeprecationWarning, stacklevel=2)
        if not value:
            value = deprecated_value
    return value


def read_datastore_endpoint(datastore_type: str) -> Optional[str]:
    """Read the datastore endpoint, or build it from the deprecated MinIO host and TLS settings.

    :param datastore_type: The datastore type, `s3-compatible` type has a default endpoint
    :return: The endpoint, or None if it is not set
    """
    for deprecated_name in ("MINIO_SHORT_HOST", "MINIO_USE_TLS"):
        if os.getenv(deprecated_name):
            warnings.warn(deprecation_message(deprecated_name, "DATASTORE_ENDPOINT"), DeprecationWarning, stacklevel=2)
    datastore_endpoint = os.getenv("DATASTORE_ENDPOINT")
    minio_short_host = os.getenv("MINIO_SHORT_HOST")
    if not datastore_endpoint and minio_short_host:
        minio_use_tls = os.getenv("MINIO_USE_TLS")
        use_tls = to_bool(minio_use_tls) if minio_use_tls else False
        datastore_endpoint = f"http{'s' if use_tls else ''}://{minio_short_host}"
    if not datastore_endpoint and datastore_type == "s3-compatible":
        # Mute Sonar about hardcoded HTTP URL, since this is a hostname inside a docker-compose file
        datastore_endpoint = "http://minio:9000"  # NOSONAR
    return datastore_endpoint


def read_datastore_region(datastore_type: str) -> Optional[str]:
    """Read the datastore region.

    :param datastore_type: The datastore type, OpenDAL S3 types have a default region
    :return: The region, or None if it is not set
    """
    datastore_region = getenv_with_deprecated(
        "DATASTORE_REGION", "ANALYZER_BINSTORE_MINIO_REGION", "ANALYZER_BINARYSTORE_MINIO_REGION"
    )
    if not datastore_region and datastore_type in {"s3-compatible", "aws-s3"}:
        datastore_region = "us-east-1"
    return datastore_region


def read_datastore_bucket_prefix() -> str:
    """Read the bucket prefix, under the default bucket if one is set."""
    bucket_prefix = getenv_with_deprecated(
        "DATASTORE_BUCKETPREFIX", "ANALYZER_BINSTORE_BUCKETPREFIX", "ANALYZER_BINARYSTORE_BUCKETPREFIX"
    )
    default_bucket_name = os.getenv("DATASTORE_DEFAULTBUCKETNAME")
    if not default_bucket_name:
        return bucket_prefix or "prj-"
    if bucket_prefix:
        return f"{default_bucket_name}/{bucket_prefix}"
    return default_bucket_name


def read_datastore_default_bucket_name(bucket_prefix: str) -> str:
    """Read the bucket which OpenDAL datastore types are bound to.

    :param bucket_prefix: The bucket prefix, which can hold the bucket as the first part of its path
    :return: The default bucket, if it is not set: the first part of the prefix path if it has one, otherwise the same
        default bucket as in ReportPortal API service
    """
    default_bucket_name = os.getenv("DATASTORE_DEFAULTBUCKETNAME")
    if default_bucket_name:
        return default_bucket_name
    prefix_path = bucket_prefix.replace("\\", "/").strip("/")
    if "/" in prefix_path:
        return prefix_path.split("/")[0]
    return "rp-bucket"


def build_app_config() -> ApplicationConfig:
    """Read the application configuration from environment variables, old and new names.

    :return: Application configuration
    """
    # Datastore settings are read in this order, to give deprecation warnings in the order of the variables
    datastore_type = (
        getenv_with_deprecated("DATASTORE_TYPE", "ANALYZER_BINSTORE_TYPE", "ANALYZER_BINARYSTORE_TYPE") or "filesystem"
    )
    datastore_endpoint = read_datastore_endpoint(datastore_type)
    datastore_region = read_datastore_region(datastore_type)
    datastore_bucket_prefix = read_datastore_bucket_prefix()
    datastore_default_bucket_name = read_datastore_default_bucket_name(datastore_bucket_prefix)
    return ApplicationConfig(
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
        analyzerResponseExchange=os.getenv("AMQP_RESPONSE_EXCHANGE", "analyzer-reply"),
        analyzerResponseQueue=os.getenv("ANALYZER_RESPONSE_QUEUE", "analysis.matches"),
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
        appVersion=read_version(),
        # Storage settings
        datastoreType=datastore_type,
        datastoreEndpoint=datastore_endpoint,
        datastoreRegion=datastore_region,
        datastoreBucketPrefix=datastore_bucket_prefix,
        datastoreBucketPostfix=os.getenv("DATASTORE_BUCKETPOSTFIX", ""),
        datastoreDefaultBucketName=datastore_default_bucket_name,
        datastoreAccessKey=getenv_with_deprecated("DATASTORE_ACCESSKEY", "MINIO_ACCESS_KEY"),
        datastoreSecretKey=getenv_with_deprecated("DATASTORE_SECRETKEY", "MINIO_SECRET_KEY"),
        datastoreAzureAccountName=os.getenv("DATASTORE_AZURE_ACCOUNTNAME"),
        datastoreAzureAccountKey=os.getenv("DATASTORE_AZURE_ACCOUNTKEY"),
        datastoreAzureEndpoint=os.getenv("DATASTORE_AZURE_ENDPOINT"),
        datastoreAzureContainer=os.getenv("DATASTORE_AZURE_CONTAINER") or datastore_default_bucket_name,
        datastoreGcsBucket=os.getenv("DATASTORE_GCS_BUCKET") or datastore_default_bucket_name,
        datastoreGcsCredentialsPath=os.getenv("DATASTORE_GCS_CREDENTIALSPATH"),
        datastoreGcsEndpoint=os.getenv("DATASTORE_GCS_ENDPOINT"),
        datastoreOpendalOptions=parse_string_map(os.getenv("DATASTORE_OPENDAL_OPTIONS")),
        filesystemDefaultPath=os.getenv("FILESYSTEM_DEFAULT_PATH", "storage").strip(),
        # HTTP endpoint settings
        analyzerHttpPort=int(os.getenv("ANALYZER_HTTP_PORT", "5001")),
        # Log settings
        analyzerPathToLog=os.getenv("ANALYZER_FILE_LOGGING_PATH", "/tmp/config.log"),
        logLevel=os.getenv("LOGGING_LEVEL", "INFO").strip(),
        # Debug settings, controls if AMQP handler runs in threaded mode to ease debugging
        debugMode=to_bool(os.getenv("DEBUG_MODE", "false")),
        # Features
        disableTrain=to_bool(os.getenv("DISABLE_TRAIN", "false")),
        # Processing settings: 0 workers means another instance consumes the queue
        analyzerWorkers=max(0, int(os.getenv("ANALYZER_WORKERS", "2"))),
        analyzerTrainWorkers=max(0, int(os.getenv("ANALYZER_TRAIN_WORKERS", "1"))),
        analyzerMaxQueuedTasks=max(1, int(os.getenv("ANALYZER_MAX_QUEUED_TASKS", "50"))),
        analyzerMaxQueuedBytes=max(1, int(os.getenv("ANALYZER_MAX_QUEUED_BYTES", str(256 * 1024 * 1024)))),
    )


def build_search_config() -> SearchConfig:
    """Read the search configuration from environment variables.

    :return: Search configuration
    """
    model_settings = read_model_settings()
    return SearchConfig(
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
        BoostModelFolder=file_utils.strip_path(model_settings["BOOST_MODEL_FOLDER"]),
        SuggestBoostModelFolder=file_utils.strip_path(model_settings["SUGGEST_BOOST_MODEL_FOLDER"]),
        GlobalDefectTypeModelFolder=file_utils.strip_path(model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]),
        MaxSuggestionsNumber=int(os.getenv("MAX_SUGGESTIONS_NUMBER", "3")),
        MaxAutoAnalysisItemsToProcess=int(
            os.getenv("ANALYZER_MAX_ITEMS_TO_PROCESS", os.getenv("MAX_AUTO_ANALYSIS_ITEMS_TO_PROCESS", "4000"))
        ),
        MlModelForSuggestions=os.getenv("ML_MODEL_FOR_SUGGESTIONS", ModelType.suggestion.name).strip(),
        AnalysisQueryBatchSize=int(os.getenv("ANALYZER_QUERY_BATCH_SIZE", "20")),
        ItemQueryTermsBudget=int(os.getenv("ANALYZER_QUERY_TERMS_BUDGET", "800")),
        HashMatchSameTest=to_bool(os.getenv("ANALYZER_HASH_MATCH_SAME_TEST", "false")),
        HashMatchAnyTest=to_bool(os.getenv("ANALYZER_HASH_MATCH_ANY_TEST", "false")),
    )


def except_train(request: str) -> bool:
    """Filters out requests that are for training models"""
    return request != "train_models"


def only_train(request: str) -> bool:
    """Filters out requests that are not related to training models"""
    return request == "train_models"


def create_pool(
    app_config: ApplicationConfig, search_config: SearchConfig, services: list[str], workers: int, name: str
) -> TaskPool:
    """Create the workers of a coordinator: processes, or one thread in debug mode."""
    if app_config.debugMode:
        return InlinePool(app_config, search_config, set(services))
    return WorkerPool.of_workers(app_config, search_config, set(services), workers, name)


def create_coordinator(
    app_config: ApplicationConfig,
    search_config: SearchConfig,
    *,
    queue: str,
    predicate: Any,
    services: list[str],
    workers: int,
) -> LaneCoordinator:
    """Create the coordinator of one queue, with its own AMQP connection, workers and budget."""
    return LaneCoordinator(
        app_config,
        name=queue,
        queue=queue,
        routing_key_predicate=predicate,
        pool=create_pool(app_config, search_config, services, workers, queue),
        budget=IntakeBudget(app_config.analyzerMaxQueuedTasks, app_config.analyzerMaxQueuedBytes),
        client=AsyncAmqpClient(app_config),
    )


def create_coordinators(app_config: ApplicationConfig, search_config: SearchConfig) -> list[Coordinator]:
    """Create the coordinators of the queues this instance handles.

    With training disabled, the "train" queue is consumed and its messages are discarded, so it doesn't grow forever,
    whatever the number of train workers is. Otherwise a queue with 0 workers is consumed by another instance, so
    this one doesn't touch it.

    :param app_config: Application configuration
    :param search_config: Search configuration
    :return: Coordinators, empty if this instance handles no queue
    """
    coordinators: list[Coordinator] = []
    if app_config.analyzerWorkers > 0:
        coordinators.append(
            create_coordinator(
                app_config,
                search_config,
                queue="all",
                predicate=except_train,
                services=ALL_SERVICES,
                workers=app_config.analyzerWorkers,
            )
        )
    if app_config.disableTrain:
        coordinators.append(DiscardingCoordinator(name="train", queue="train", client=AsyncAmqpClient(app_config)))
    elif app_config.analyzerTrainWorkers > 0:
        coordinators.append(
            create_coordinator(
                app_config,
                search_config,
                queue="train",
                predicate=only_train,
                services=TRAIN_SERVICES,
                workers=app_config.analyzerTrainWorkers,
            )
        )
    return coordinators


def describe_queue(workers: int, disabled: bool = False) -> str:
    """Describe how this instance handles a queue, for the startup log.

    :param workers: Number of worker processes of the queue
    :param disabled: Whether the feature of the queue is disabled
    :return: The description
    """
    if disabled:
        return "messages are discarded, training is disabled"
    if workers == 0:
        return "consumed by another instance"
    return f"{workers} worker process{'' if workers == 1 else 'es'}"


async def run(app_config: ApplicationConfig, coordinators: list[Coordinator]) -> None:
    """Run the coordinators and the health endpoint until SIGTERM or SIGINT.

    :param app_config: Application configuration
    :param coordinators: Coordinators of the queues this instance handles
    """
    health = HealthServer(app_config, coordinators, OpenSearchHealth(app_config))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signal_number, stop.set)

    await health.start()
    starting = [asyncio.create_task(coordinator.start()) for coordinator in coordinators]
    logger.info("The analyzer has started.")
    await stop.wait()

    logger.info("Stop signal received, shutting down.")
    for task in starting:
        task.cancel()
    await asyncio.gather(*starting, return_exceptions=True)
    await asyncio.gather(*(coordinator.shutdown() for coordinator in coordinators), return_exceptions=True)
    await health.stop()


def main() -> None:
    """Configure the application and run it."""
    app_config = build_app_config()
    search_config = build_search_config()
    my_logging.setup(app_config)
    coordinators = create_coordinators(app_config, search_config)
    if not coordinators:
        logger.error("Nothing to consume: both ANALYZER_WORKERS and ANALYZER_TRAIN_WORKERS are 0.")
        exit(1)
    if app_config.debugMode:
        logger.warning("The application is running in DEBUG mode, processing will run in a thread.")
    else:
        logger.info(
            "The application is running in PRODUCTION mode. "
            f"Queue 'all': {describe_queue(app_config.analyzerWorkers)}. "
            f"Queue 'train': {describe_queue(app_config.analyzerTrainWorkers, app_config.disableTrain)}."
        )
    asyncio.run(run(app_config, coordinators))
    logger.info("The analyzer has finished.")


if __name__ == "__main__":
    main()
    exit(0)
