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
from collections import defaultdict
from datetime import datetime
from time import monotonic
from typing import Optional

from app.commons import logging, request_factory
from app.commons.model.launch_objects import ApplicationConfig, DefectUpdate, ItemUpdate, Launch
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.processing import Reply, ReplyType
from app.commons.model.test_item_index import TestItemHistoryData, TestItemIndexData
from app.commons.os_client import OsClient

LOGGER = logging.getLogger("analyzerApp.indexService")


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for use in indexing"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def current_timestamp() -> str:
    """Return current timestamp in indexing format."""
    return format_timestamp(datetime.now())


TRAIN_ROUTING_KEY = "train_models"


def train_replies(update_list: list[tuple[int, int]]) -> list[Reply]:
    """Model training requests for the projects with new defect types: one per model type.

    :param update_list: (project ID, number of new defect types) pairs
    :return: Replies to the training queue
    """
    return [
        Reply(
            TrainInfo(model_type=model_type, project=project_id, gathered_metric_total=result_num).model_dump_json(),
            ReplyType.QUEUE_SEND,
            TRAIN_ROUTING_KEY,
        )
        for project_id, result_num in update_list
        if result_num and result_num > 0
        for model_type in (ModelType.suggestion, ModelType.auto_analysis, ModelType.defect_type)
    ]


class IndexService:
    """Service for indexing launches and managing launch-related data in OpenSearch"""

    app_config: ApplicationConfig
    os_client: OsClient

    def __init__(self, app_config: ApplicationConfig, *, os_client: Optional[OsClient] = None):
        """Initialize IndexService

        :param app_config: Application configuration object
        :param os_client: Optional OsClient instance. If not provided, a new one will be created.
        """
        self.app_config = app_config
        self.os_client = os_client or OsClient(app_config=self.app_config)

    def index_logs(self, launches: list[Launch]) -> Optional[list[Reply]]:
        """Index launches grouped by project using Test Item-centric documents.

        :param launches: Launches to index
        :return: Model training requests for the projects with new defect types, or None if there are none
        """
        if not launches:
            return None

        launch_ids = {str(launch_obj.launchId) for launch_obj in launches}
        launch_ids_str = ", ".join(launch_ids)
        projects = {str(launch_obj.project) for launch_obj in launches}
        projects_str = ", ".join(projects)
        LOGGER.info(f"Indexing {len(launch_ids)} launches of projects '{projects_str}': {launch_ids_str}")
        t_start = monotonic()

        project_test_items: dict[int, list[TestItemIndexData]] = defaultdict(list)
        defects_per_project: dict[int, int] = defaultdict(int)

        for launch in launches:
            config = launch.analyzerConfig
            prepared_items = request_factory.prepare_test_items(
                launch,
                number_of_logs_to_index=config.numberOfLogsToIndex,
                minimal_log_level=config.minimumLogLevel,
                similarity_threshold_to_drop=config.similarityThresholdToDrop,
            )
            project_test_items[launch.project].extend(prepared_items)
            defects_per_project[launch.project] += sum(
                1 for item in prepared_items if item.issue_type and not item.issue_type.startswith("ti")
            )

        indexed_count = 0
        any_errors = False

        for project_id, test_items in project_test_items.items():
            response = self.os_client.bulk_index(project_id, test_items)
            indexed_count += response.took  # `took` holds the number of indexed documents
            any_errors = any_errors or response.errors

        replies = train_replies(list(defects_per_project.items()))

        time_passed = round(monotonic() - t_start, 2)
        LOGGER.info(
            f"Indexing {len(launch_ids)} launches of projects '{projects_str}' finished: {launch_ids_str}. "
            f"It took {time_passed} sec., {indexed_count} test items indexed."
        )
        return replies or None

    @staticmethod
    def _normalize_items_to_update(
        items_to_update: dict[int | str, str | ItemUpdate],
    ) -> dict[str, dict[str, str]]:
        """Normalize incoming itemsToUpdate payload to a uniform structure."""
        normalized: dict[str, dict[str, str]] = {}
        for key, raw_value in items_to_update.items():
            item_id = str(key)
            issue_comment = ""
            if isinstance(raw_value, ItemUpdate):
                issue_type = raw_value.issueType
                issue_comment = raw_value.issueComment
                # The last element is Java nanoseconds, which do not fit into Python microseconds
                timestamp = format_timestamp(datetime(*raw_value.timestamp[:6]))
            else:
                issue_type = raw_value
                timestamp = current_timestamp()

            if issue_type.strip():
                normalized[item_id] = {
                    "issue_type": issue_type,
                    "issue_comment": issue_comment,
                    "timestamp": timestamp,
                }
        return normalized

    def defect_update(self, defect_update_info: DefectUpdate) -> list[Reply]:
        """Update defect types and issue history for provided test items.

        Updates are sent for all requested Test Items without checking first which of them are indexed: a script update
        of a missing document fails on its own, and such Test Items are reported as not updated, so ReportPortal
        re-indexes them.

        :param defect_update_info: Defect update request
        :return: The reply with the IDs of Test Items which were not updated, then the model training requests
        """
        LOGGER.info("Started updating defect types")
        t_start = monotonic()

        project_id = defect_update_info.project
        is_auto_analyzed = defect_update_info.isAutoAnalyzed
        normalized_updates = self._normalize_items_to_update(defect_update_info.itemsToUpdate)

        history_updates: list[TestItemHistoryData] = [
            TestItemHistoryData(
                test_item_id=test_item_id,
                is_auto_analyzed=is_auto_analyzed,
                issue_type=request_factory.transform_issue_type_into_lowercase(update_payload["issue_type"]),
                timestamp=update_payload["timestamp"],
                issue_comment=update_payload["issue_comment"],
            )
            for test_item_id, update_payload in normalized_updates.items()
        ]

        failed_test_items: set[str] = set()
        defect_num = 0
        if history_updates:
            response = self.os_client.bulk_update_issue_history(project_id, history_updates)
            defect_num = response.took
            if response.errors and not response.failed_ids:
                # The whole request failed, e.g. the index does not exist, so nothing was updated
                failed_test_items.update(normalized_updates.keys())
            else:
                # Missing Test Items, e.g. not indexed or removed by a cleanup request, fail individually
                failed_test_items.update(response.failed_ids)
        items_not_updated = sorted(int(test_item_id) for test_item_id in failed_test_items)
        LOGGER.debug("Not updated test items: %s", items_not_updated)
        LOGGER.info("Finished updating defect types. It took %.2f sec", monotonic() - t_start)
        return [Reply(json.dumps(items_not_updated))] + train_replies([(int(project_id), defect_num)])
