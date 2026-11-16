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

from unittest import mock

from app.commons.model.launch_objects import AnalyzerConf, TestItemInfo
from app.service.suggest_service import SuggestService
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG
from test.unit.service.test_similarity_filtering import REQUEST_MESSAGE, build_found_hit

CLUSTER_ID = 123456789
LAUNCH_ID = 10


def build_cluster_test_item_info() -> TestItemInfo:
    return TestItemInfo(testItemId=0, clusterId=CLUSTER_ID, launchId=LAUNCH_ID, project=1)


def test_cluster_request_item_is_found_with_one_plain_search():
    """The request Test Item of a cluster is the first found one, so it is taken without scrolling all matches."""
    found_hit = build_found_hit(7, [REQUEST_MESSAGE], AnalyzerConf(), {})
    os_client = mock.Mock()
    os_client.msearch_grouped.return_value = iter([[found_hit]])
    service = SuggestService(mock.Mock(), APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client)

    request_item, test_item_id_for_suggest = service._prepare_request_data(build_cluster_test_item_info())

    os_client.search.assert_not_called()
    os_client.msearch_grouped.assert_called_once()
    project_id, queries = os_client.msearch_grouped.call_args[0]
    assert project_id == 1
    assert len(queries) == 2
    assert queries[0] == {}
    assert queries[1]["size"] == 1
    assert {"term": {"launch_id": str(LAUNCH_ID)}} in queries[1]["query"]["bool"]["must"]
    assert test_item_id_for_suggest == 7
    assert request_item is not None
    assert request_item.test_item_id == "0", "identity fields of the found Test Item should be cleared"
    assert request_item.logs == found_hit.source.logs


def test_cluster_request_item_is_not_found():
    os_client = mock.Mock()
    os_client.msearch_grouped.return_value = iter([[]])
    service = SuggestService(mock.Mock(), APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client)

    request_item, test_item_id_for_suggest = service._prepare_request_data(build_cluster_test_item_info())

    os_client.msearch_grouped.assert_called_once()
    assert request_item is None
    assert test_item_id_for_suggest == 0
