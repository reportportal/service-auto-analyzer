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

import json
import time
from typing import Any, Optional

from app.commons import logging, model_chooser
from app.commons.model import launch_objects, ml
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.service.analyzer_service import AnalyzerService
from app.service.auto_analyzer_service import AutoAnalyzerService
from app.service.clean_index_service import CleanIndexService
from app.service.cluster_service import ClusterService
from app.service.index_service import IndexService
from app.service.namespace_finder_service import NamespaceFinderService
from app.service.retraining_service import RetrainingService
from app.service.search_service import SearchService
from app.service.suggest_info_service import SuggestInfoService
from app.service.suggest_patterns_service import SuggestPatternsService
from app.service.suggest_service import SuggestService

LOGGER = logging.getLogger("analyzerApp.processor")


# Helper functions for data preparation and response formatting
def prepare_launches(launches: list) -> list[launch_objects.Launch]:
    """Function for deserializing array of launches"""
    return [launch_objects.Launch(**launch) for launch in launches]


def prepare_update(update_data: dict) -> launch_objects.DefectUpdate:
    return launch_objects.DefectUpdate(**update_data)


def prepare_suggest_info_list(suggest_info_list: list) -> list[launch_objects.SuggestAnalysisResult]:
    """Function for deserializing array of suggest info results"""
    return [launch_objects.SuggestAnalysisResult(**res) for res in suggest_info_list]


def prepare_search_logs(search_data: dict) -> launch_objects.SearchLogs:
    """Function for deserializing search logs object"""
    return launch_objects.SearchLogs(**search_data)


def prepare_launch_info(launch_info: dict) -> launch_objects.LaunchInfoForClustering:
    """Function for deserializing search logs object"""
    return launch_objects.LaunchInfoForClustering(**launch_info)


def prepare_clean_index(clean_index: dict) -> launch_objects.DeleteLogsRequest:
    """Function for deserializing clean index object"""
    return launch_objects.DeleteLogsRequest(**clean_index)


def prepare_delete_test_items(remove_items_info: dict) -> launch_objects.DeleteTestItemsRequest:
    """Function for deserializing delete test items payload"""
    return launch_objects.DeleteTestItemsRequest(**remove_items_info)


def prepare_delete_launches(launch_remove_info: dict) -> launch_objects.DeleteLaunchesRequest:
    """Function for deserializing delete launches payload"""
    return launch_objects.DeleteLaunchesRequest(**launch_remove_info)


def prepare_remove_by_dates(
    remove_by_dates: dict,
) -> launch_objects.RemoveByDatesRequest:
    """Function for deserializing remove by launch start time payload"""
    return launch_objects.RemoveByDatesRequest(**remove_by_dates)


def to_int(body: Any) -> int:
    """Function for deserializing index id object"""
    return int(body)


def prepare_test_item_info(test_item_info: Any) -> launch_objects.TestItemInfo:
    """Function for deserializing test item info for suggestions"""
    return launch_objects.TestItemInfo(**test_item_info)


def prepare_train_info(train_info: dict) -> ml.TrainInfo:
    """Function for deserializing train info object"""
    return ml.TrainInfo(**train_info)


def to_json(response: list | dict) -> str:
    """Function for serializing response to JSON format"""
    return json.dumps(response)


def prepare_analyze_response_data(response: list) -> str:
    """Function for serializing response from analyze request"""
    return to_json([resp.model_dump() for resp in response])


def prepare_index_response_data(response: Any) -> str:
    """Function for serializing response from index request
    and other objects, which are pydantic objects"""
    return response.model_dump_json()


def to_str(response: Any) -> str:
    """Function for serializing int object"""
    return str(response)


def same_data(data: Any) -> Any:
    """Function for returning data without changes"""
    return data


def raise_exception(message: str) -> None:
    """Function to raise an exception with a given message"""
    raise RuntimeError(message)


class ServiceProcessor:
    """Class for processing requests based on routing key and routing configuration"""

    _model_chooser: Optional[model_chooser.ModelChooser] = None
    _index_service: Optional[IndexService] = None
    _clean_index_service: Optional[CleanIndexService] = None
    _analyzer_service: Optional[AnalyzerService] = None
    _suggest_info_service: Optional[SuggestInfoService] = None

    __configs: dict[str, dict[str, Any]] = {
        "train_models": {
            "handler": lambda s: RetrainingService(s.model_chooser, s.app_config, s.search_config).train_models,
            "prepare_data_func": prepare_train_info,
        },
        "index": {
            "handler": lambda s: s.index_service.index_logs,
            "prepare_data_func": prepare_launches,
            "prepare_response_data": prepare_index_response_data,
        },
        "defect_update": {
            "handler": lambda s: s.index_service.defect_update,
            "prepare_data_func": prepare_update,
            "prepare_response_data": to_json,
        },
        "analyze": {
            "handler": lambda s: AutoAnalyzerService(s.model_chooser, s.app_config, s.search_config).analyze_logs,
            "prepare_data_func": prepare_launches,
            "prepare_response_data": prepare_analyze_response_data,
        },
        "delete": {
            "handler": lambda s: s.clean_index_service.delete_index,
            "prepare_data_func": to_int,
            "prepare_response_data": to_str,
        },
        "clean": {
            "handler": lambda s: s.clean_index_service.delete_logs,
            "prepare_data_func": prepare_clean_index,
            "prepare_response_data": to_str,
        },
        "item_remove": {
            "handler": lambda s: s.clean_index_service.delete_test_items,
            "prepare_data_func": prepare_delete_test_items,
            "prepare_response_data": to_str,
        },
        "launch_remove": {
            "handler": lambda s: s.clean_index_service.delete_launches,
            "prepare_data_func": prepare_delete_launches,
            "prepare_response_data": to_str,
        },
        "remove_by_launch_start_time": {
            "handler": lambda s: s.clean_index_service.remove_by_launch_start_time,
            "prepare_data_func": prepare_remove_by_dates,
            "prepare_response_data": to_str,
        },
        "remove_by_log_time": {
            "handler": lambda s: s.clean_index_service.remove_by_log_time,
            "prepare_data_func": prepare_remove_by_dates,
            "prepare_response_data": to_str,
        },
        "search": {
            "handler": lambda s: SearchService(s.app_config, s.search_config).search_logs,
            "prepare_data_func": prepare_search_logs,
            "prepare_response_data": prepare_analyze_response_data,
        },
        "suggest": {
            "handler": lambda s: SuggestService(s.model_chooser, s.app_config, s.search_config).suggest_items,
            "prepare_data_func": prepare_test_item_info,
            "prepare_response_data": prepare_analyze_response_data,
        },
        "cluster": {
            "handler": lambda s: ClusterService(s.app_config, s.search_config).find_clusters,
            "prepare_data_func": prepare_launch_info,
            "prepare_response_data": prepare_index_response_data,
        },
        "namespace_finder": {
            "handler": lambda s: NamespaceFinderService(s.app_config).update_chosen_namespaces,
            "prepare_data_func": prepare_launches,
        },
        "suggest_patterns": {
            "handler": lambda s: SuggestPatternsService(s.app_config, s.search_config).suggest_patterns,
            "prepare_data_func": to_int,
            "prepare_response_data": prepare_index_response_data,
        },
        "index_suggest_info": {
            "handler": lambda s: s.suggest_info_service.index_suggest_info,
            "prepare_data_func": prepare_suggest_info_list,
            "prepare_response_data": prepare_index_response_data,
        },
        "remove_suggest_info": {
            "handler": lambda s: s.suggest_info_service.remove_suggest_info,
            "prepare_data_func": to_int,
            "prepare_response_data": to_str,
        },
        "update_suggest_info": {
            "handler": lambda s: s.suggest_info_service.update_suggest_info,
            "prepare_data_func": same_data,
            "prepare_response_data": to_json,
        },
        "remove_models": {
            "handler": lambda s: lambda x: LOGGER.warning(
                f"Deprecated 'remove_models' route called with {json.dumps(x)}"
            ),
            "prepare_data_func": same_data,
            "prepare_response_data": to_str,
        },
        "get_model_info": {
            "handler": lambda s: lambda x: LOGGER.warning(
                f"Deprecated 'get_model_info' route called with {json.dumps(x)}"
            ),
            "prepare_data_func": same_data,
            "prepare_response_data": to_json,
        },
        "noop_sleep": {
            "handler": lambda s: lambda x: time.sleep(x),
            "prepare_data_func": same_data,
        },
        "noop_echo": {
            "handler": lambda s: lambda x: x,
            "prepare_data_func": same_data,
            "prepare_response_data": to_str,
        },
        "noop_fail": {
            "handler": lambda s: lambda x: raise_exception("Intentional failure for testing purposes: " + str(x)),
            "prepare_data_func": same_data,
        },
    }

    def __init__(
        self, app_config: ApplicationConfig, search_config: SearchConfig, services_to_init: Optional[set[str]] = None
    ):
        """Initialize all services based on instance task type.

        :param app_config: Application configuration object
        :param search_config: Search configuration object
        :param services_to_init: Set of service names to initialize. If None, all services are initialized.
        """
        self.app_config = app_config
        self.search_config = search_config

        # Define routing configuration for different queue types
        self._routing_config = self._build_routing_config(services_to_init)

    @property
    def model_chooser(self) -> model_chooser.ModelChooser:
        if not self._model_chooser:
            self._model_chooser = model_chooser.ModelChooser(self.app_config, self.search_config)
        return self._model_chooser

    @property
    def clean_index_service(self) -> CleanIndexService:
        if not self._clean_index_service:
            self._clean_index_service = CleanIndexService(self.model_chooser, self.app_config, self.search_config)
        return self._clean_index_service

    @property
    def index_service(self) -> IndexService:
        if not self._index_service:
            self._index_service = IndexService(self.app_config)
        return self._index_service

    @property
    def analyzer_service(self) -> AnalyzerService:
        if not self._analyzer_service:
            self._analyzer_service = AnalyzerService(self.model_chooser, self.search_config)
        return self._analyzer_service

    @property
    def suggest_info_service(self) -> SuggestInfoService:
        if not self._suggest_info_service:
            self._suggest_info_service = SuggestInfoService(self.app_config)
        return self._suggest_info_service

    def _build_routing_config(self, routing_keys: Optional[set[str]]) -> dict:
        """Build routing configuration for different routing keys"""

        config = {}
        for key, config_builder in self.__configs.items():
            if routing_keys is None or key in routing_keys:
                config_def = config_builder.copy()
                if "handler" in config_def:
                    handler_func = config_def["handler"]
                    config_def["handler"] = handler_func(self)
                config[key] = config_def

        LOGGER.debug(f"Routing configuration built: {config.keys()}")
        return config

    def process(self, routing_key: str, body: Any) -> Optional[str]:
        """Process request based on routing key and body data"""
        config = self._routing_config[routing_key]
        request_processor = config["handler"]
        prepare_data_func = config.get("prepare_data_func", None)
        prepare_response_data = config.get("prepare_response_data", None)

        # Prepare message body if needed
        message = body
        if prepare_data_func:
            try:
                message = prepare_data_func(message)
            except Exception as exc:
                LOGGER.exception("Failed to prepare message body", exc_info=exc)
                return None

        response = request_processor(message)
        LOGGER.debug("Finished processing request")

        # Prepare response if applicable
        if response is None or not prepare_response_data:
            return None

        try:
            response_body = prepare_response_data(response)
        except Exception as exc:
            LOGGER.exception("Failed to prepare response body", exc_info=exc)
            return None

        return response_body
