#  Copyright 2025 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import time
from typing import Any, Optional

from app.commons import logging, model_chooser
from app.commons.esclient import EsClient
from app.commons.model import launch_objects, ml
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.service.analyzer_service import AnalyzerService
from app.service.auto_analyzer_service import AutoAnalyzerService
from app.service.clean_index_service import CleanIndexService
from app.service.cluster_service import ClusterService
from app.service.delete_index_service import DeleteIndexService
from app.service.namespace_finder_service import NamespaceFinderService
from app.service.retraining_service import RetrainingService
from app.service.search_service import SearchService
from app.service.suggest_info_service import SuggestInfoService
from app.service.suggest_patterns_service import SuggestPatternsService
from app.service.suggest_service import SuggestService

logger = logging.getLogger("analyzerApp.processor")


# Helper functions for data preparation and response formatting
def prepare_launches(launches: list) -> list[launch_objects.Launch]:
    """Function for deserializing array of launches"""
    return [launch_objects.Launch(**launch) for launch in launches]


def prepare_suggest_info_list(suggest_info_list: list) -> list[launch_objects.SuggestAnalysisResult]:
    """Function for deserializing array of suggest info results"""
    return [launch_objects.SuggestAnalysisResult(**res) for res in suggest_info_list]


def prepare_search_logs(search_data: dict) -> launch_objects.SearchLogs:
    """Function for deserializing search logs object"""
    return launch_objects.SearchLogs(**search_data)


def prepare_launch_info(launch_info: dict) -> launch_objects.LaunchInfoForClustering:
    """Function for deserializing search logs object"""
    return launch_objects.LaunchInfoForClustering(**launch_info)


def prepare_clean_index(clean_index: dict) -> launch_objects.CleanIndex:
    """Function for deserializing clean index object"""
    return launch_objects.CleanIndex(**clean_index)


def prepare_delete_index(body: Any) -> int:
    """Function for deserializing index id object"""
    return int(body)


def prepare_test_item_info(test_item_info: Any) -> launch_objects.TestItemInfo:
    """Function for deserializing test item info for suggestions"""
    return launch_objects.TestItemInfo(**test_item_info)


def prepare_train_info(train_info: dict) -> ml.TrainInfo:
    """Function for deserializing train info object"""
    return ml.TrainInfo(**train_info)


def prepare_search_response_data(response: list | dict) -> str:
    """Function for serializing response from search request"""
    import json

    return json.dumps(response)


def prepare_analyze_response_data(response: list) -> str:
    """Function for serializing response from analyze request"""
    import json

    return json.dumps([resp.dict() for resp in response])


def prepare_index_response_data(response: Any) -> str:
    """Function for serializing response from index request
    and other objects, which are pydantic objects"""
    return response.json()


def output_result(response: Any) -> str:
    """Function for serializing int object"""
    return str(response)


def same_data(data: Any) -> Any:
    """Function for returning data without changes"""
    return data


class ServiceProcessor:
    """Class for processing requests based on routing key and routing configuration"""

    def __init__(self, app_config: ApplicationConfig, search_config: SearchConfig):
        """Initialize all services based on instance task type"""
        self.app_config = app_config
        self.search_config = search_config
        self._model_chooser = model_chooser.ModelChooser(app_config, search_config)

        # Initialize services based on instance task type
        self._retraining_service = RetrainingService(self._model_chooser, app_config, search_config)
        self._es_client = EsClient(app_config)
        self._auto_analyzer_service = AutoAnalyzerService(self._model_chooser, app_config, search_config)
        self._delete_index_service = DeleteIndexService(self._model_chooser, app_config, search_config)
        self._clean_index_service = CleanIndexService(app_config)
        self._analyzer_service = AnalyzerService(self._model_chooser, search_config)
        self._suggest_service = SuggestService(self._model_chooser, app_config, search_config)
        self._suggest_info_service = SuggestInfoService(app_config)
        self._search_service = SearchService(app_config, search_config)
        self._cluster_service = ClusterService(app_config, search_config)
        self._namespace_finder_service = NamespaceFinderService(app_config)
        self._suggest_patterns_service = SuggestPatternsService(app_config, search_config)

        # Define routing configuration for different queue types
        self._routing_config = self._build_routing_config()

    def _build_routing_config(self) -> dict:
        """Build routing configuration for different routing keys"""
        config = {
            "train_models": {
                "handler": self._retraining_service.train_models,
                "prepare_data_func": prepare_train_info,
            },
            "index": {
                "handler": self._es_client.index_logs,
                "prepare_data_func": prepare_launches,
                "prepare_response_data": prepare_index_response_data,
            },
            "analyze": {
                "handler": self._auto_analyzer_service.analyze_logs,
                "prepare_data_func": prepare_launches,
                "prepare_response_data": prepare_analyze_response_data,
            },
            "delete": {
                "handler": self._delete_index_service.delete_index,
                "prepare_data_func": prepare_delete_index,
                "prepare_response_data": output_result,
            },
            "clean": {
                "handler": self._clean_index_service.delete_logs,
                "prepare_data_func": prepare_clean_index,
                "prepare_response_data": output_result,
            },
            "search": {
                "handler": self._search_service.search_logs,
                "prepare_data_func": prepare_search_logs,
                "prepare_response_data": prepare_analyze_response_data,
            },
            "suggest": {
                "handler": self._suggest_service.suggest_items,
                "prepare_data_func": prepare_test_item_info,
                "prepare_response_data": prepare_analyze_response_data,
            },
            "cluster": {
                "handler": self._cluster_service.find_clusters,
                "prepare_data_func": prepare_launch_info,
                "prepare_response_data": prepare_index_response_data,
            },
            "stats_info": {
                "handler": self._es_client.send_stats_info,
            },
            "namespace_finder": {
                "handler": self._namespace_finder_service.update_chosen_namespaces,
                "prepare_data_func": prepare_launches,
            },
            "suggest_patterns": {
                "handler": self._suggest_patterns_service.suggest_patterns,
                "prepare_data_func": prepare_delete_index,
                "prepare_response_data": prepare_index_response_data,
            },
            "index_suggest_info": {
                "handler": self._suggest_info_service.index_suggest_info,
                "prepare_data_func": prepare_suggest_info_list,
                "prepare_response_data": prepare_index_response_data,
            },
            "remove_suggest_info": {
                "handler": self._suggest_info_service.remove_suggest_info,
                "prepare_data_func": prepare_delete_index,
                "prepare_response_data": output_result,
            },
            "update_suggest_info": {
                "handler": self._suggest_info_service.update_suggest_info,
                "prepare_data_func": same_data,
                "prepare_response_data": prepare_index_response_data,
            },
            "remove_models": {
                "handler": self._analyzer_service.remove_models,
                "prepare_data_func": same_data,
                "prepare_response_data": output_result,
            },
            "get_model_info": {
                "handler": self._analyzer_service.get_model_info,
                "prepare_data_func": same_data,
                "prepare_response_data": prepare_search_response_data,
            },
            "defect_update": {
                "handler": self._es_client.defect_update,
                "prepare_data_func": same_data,
                "prepare_response_data": prepare_search_response_data,
            },
            "item_remove": {
                "handler": self._clean_index_service.delete_test_items,
                "prepare_data_func": same_data,
                "prepare_response_data": output_result,
            },
            "launch_remove": {
                "handler": self._clean_index_service.delete_launches,
                "prepare_data_func": same_data,
                "prepare_response_data": output_result,
            },
            "remove_by_launch_start_time": {
                "handler": self._clean_index_service.remove_by_launch_start_time,
                "prepare_data_func": same_data,
                "prepare_response_data": output_result,
            },
            "remove_by_log_time": {
                "handler": self._clean_index_service.remove_by_log_time,
                "prepare_data_func": same_data,
                "prepare_response_data": output_result,
            },
            "noop_sleep": {
                "handler": lambda x: time.sleep(x),
                "prepare_data_func": same_data,
            },
            "noop_echo": {
                "handler": lambda x: x,
                "prepare_data_func": same_data,
                "prepare_response_data": output_result,
            },
        }
        return config

    def process(self, routing_key: str, body: Any) -> Optional[str]:
        """Process request based on routing key and body data"""
        config = self._routing_config[routing_key]
        request_handler = config["handler"]
        prepare_data_func = config.get("prepare_data_func", None)
        prepare_response_data = config.get("prepare_response_data", None)

        # Prepare message body if needed
        message = body
        if prepare_data_func:
            try:
                message = prepare_data_func(message)
            except Exception as exc:
                logger.exception("Failed to prepare message body", exc_info=exc)
                return None

        # Handle request
        try:
            response = request_handler(message)
        except Exception as exc:
            logger.exception("Failed to handle message", exc_info=exc)
            return None

        logger.debug("Finished processing request")

        # Prepare response if applicable
        if response is None or not prepare_response_data:
            return None

        try:
            response_body = prepare_response_data(response)
        except Exception as exc:
            logger.exception("Failed to prepare response body", exc_info=exc)
            return None

        return response_body
