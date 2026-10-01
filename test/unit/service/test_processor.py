#  Copyright 2026 EPAM Systems
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

from typing import Any
from unittest import mock

import pytest

from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.os_client import OsClient
from app.service.processor import ServiceProcessor
from app.utils import utils

PROJECT_ID = 2


@pytest.fixture
def opensearch_mock() -> mock.Mock:
    client = mock.Mock()
    client.indices.get.return_value = {}
    client.indices.create.return_value = {"acknowledged": True}
    return client


@pytest.fixture
def processor(monkeypatch, tmp_path, opensearch_mock) -> ServiceProcessor:
    app_config = ApplicationConfig(esHost="http://localhost:9200", filesystemDefaultPath=str(tmp_path))
    model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
    search_config = SearchConfig(
        BoostModelFolder=utils.strip_path(model_settings["BOOST_MODEL_FOLDER"]),
        SuggestBoostModelFolder=utils.strip_path(model_settings["SUGGEST_BOOST_MODEL_FOLDER"]),
        GlobalDefectTypeModelFolder=utils.strip_path(model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]),
    )
    monkeypatch.setattr("app.service.processor.OsClient", lambda config: OsClient(config, os_client=opensearch_mock))
    monkeypatch.setattr("app.commons.os_client.utils.read_resource_file", lambda *args, **kwargs: {})
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", lambda *args, **kwargs: (1, []))
    return ServiceProcessor(app_config, search_config)


def _handler_services(processor: ServiceProcessor) -> list[Any]:
    services = []
    for config in processor._routing_config.values():
        service = getattr(config["handler"], "__self__", None)
        if service is not None and hasattr(service, "os_client"):
            services.append(service)
    return services


def test_services_share_one_os_client(processor: ServiceProcessor) -> None:
    services = _handler_services(processor)
    assert {type(service).__name__ for service in services} == {
        "AutoAnalyzerService",
        "CleanIndexService",
        "ClusterService",
        "IndexService",
        "SearchService",
        "SuggestPatternsService",
        "SuggestService",
    }
    for service in services:
        assert service.os_client is processor.os_client

    retraining = processor._routing_config["train_models"]["handler"].__self__
    for trigger_manager in (retraining.trigger_manager, processor.clean_index_service.trigger_manager):
        for _, training in trigger_manager.model_training_triggering.values():
            assert training.os_client is processor.os_client


def test_index_deleted_by_clean_service_is_recreated_by_index_service(
    processor: ServiceProcessor, opensearch_mock: mock.Mock
) -> None:
    item = TestItemIndexData(
        test_item_id="1", launch_id="1", logs=[LogData(log_id="1", log_level=40000, message="error")]
    )
    assert processor.index_service.os_client.bulk_index(PROJECT_ID, [item]).errors is False
    opensearch_mock.indices.create.assert_not_called()

    processor.clean_index_service.delete_index(PROJECT_ID)
    opensearch_mock.indices.get.side_effect = Exception("missing index")

    assert processor.index_service.os_client.bulk_index(PROJECT_ID, [item]).errors is False
    opensearch_mock.indices.create.assert_called_once()
