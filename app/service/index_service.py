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

from collections import defaultdict
from datetime import datetime
from time import time
from typing import Optional, Union

from app.amqp.amqp import AmqpClient
from app.commons import logging, request_factory
from app.commons.model.launch_objects import ApplicationConfig, BulkResponse, DefectUpdate, ItemUpdate, Launch
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.os_client import OsClient

LOGGER = logging.getLogger("analyzerApp.indexService")


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for use in indexing"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def current_timestamp() -> str:
    """Return current timestamp in indexing format."""
    return format_timestamp(datetime.now())


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

    def index_logs(self, launches: list[Launch]) -> BulkResponse:
        """Index launches grouped by project using Test Item-centric documents."""
        if not launches:
            return BulkResponse(took=0, errors=False)

        launch_ids = {str(launch_obj.launchId) for launch_obj in launches}
        launch_ids_str = ", ".join(launch_ids)
        LOGGER.info(f"Indexing {len(launch_ids)} launches: {launch_ids_str}")
        t_start = time()

        project_test_items: dict[int, list[TestItemIndexData]] = defaultdict(list)
        defect_types_num_per_project: dict[int, int] = defaultdict(int)

        for launch in launches:
            config = launch.analyzerConfig
            prepared_items = request_factory.prepare_test_items(
                launch,
                number_of_logs_to_index=config.numberOfLogsToIndex,
                minimal_log_level=config.minimumLogLevel,
                similarity_threshold_to_drop=config.similarityThresholdToDrop,
            )
            project_test_items[launch.project].extend(prepared_items)
            defect_types_num_per_project[launch.project] += sum(
                1 for item in prepared_items if item.issue_type and not item.issue_type.startswith("ti")
            )

        total_took = 0
        any_errors = False
        amqp_client = AmqpClient(self.app_config) if self.app_config.amqpUrl else None

        for project_id, test_items in project_test_items.items():
            response = self.os_client.bulk_index(project_id, test_items)
            total_took += response.took
            any_errors = any_errors or response.errors

            if amqp_client:
                defect_types_num = defect_types_num_per_project.get(project_id, 0)
                try:
                    amqp_client.send_to_inner_queue(
                        "train_models",
                        TrainInfo(
                            model_type=ModelType.defect_type,
                            project=project_id,
                            gathered_metric_total=defect_types_num,
                        ).model_dump_json(),
                    )
                except Exception as e:
                    LOGGER.exception(f"Unable to update model train data for project {project_id}", exc_info=e)

        if amqp_client:
            amqp_client.close()

        time_passed = round(time() - t_start, 2)
        LOGGER.info(f"Indexing {len(launch_ids)} launches finished: {launch_ids_str}. It took {time_passed} sec.")
        return BulkResponse(took=total_took, errors=any_errors)

    @staticmethod
    def _normalize_items_to_update(
        items_to_update: dict[int | str, Union[str, ItemUpdate]],
    ) -> dict[str, dict[str, str]]:
        """Normalize incoming itemsToUpdate payload to a uniform structure."""
        normalized: dict[str, dict[str, str]] = {}
        for key, raw_value in items_to_update.items():
            item_id = str(key)
            issue_comment = ""
            if isinstance(raw_value, ItemUpdate):
                issue_type = raw_value.issueType
                issue_comment = raw_value.issueComment
                timestamp = format_timestamp(datetime(*raw_value.timestamp))
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

    def defect_update(self, defect_update_info: DefectUpdate) -> list[int]:
        """Update defect types and issue history for provided test items."""
        LOGGER.info("Started updating defect types")
        t_start = time()

        project_id = defect_update_info.project
        normalized_updates = self._normalize_items_to_update(defect_update_info.itemsToUpdate)
        test_item_ids = list(normalized_updates.keys())

        batch_size = self.app_config.esChunkNumber
        found_test_items: set[str] = set()
        history_updates: list[dict[str, str | bool]] = []

        for i in range(int(len(test_item_ids) / batch_size) + 1):
            batch_ids = test_item_ids[i * batch_size : (i + 1) * batch_size]
            if not batch_ids:
                continue

            existing_items = self.os_client.get_test_items_by_ids(project_id, batch_ids)
            for item in existing_items:
                found_test_items.add(item.test_item_id)
                update_payload = normalized_updates.get(item.test_item_id)
                if not update_payload:
                    continue

                issue_type = request_factory.transform_issue_type_into_lowercase(update_payload["issue_type"])
                issue_comment = update_payload["issue_comment"]
                timestamp = update_payload["timestamp"]

                history_updates.append(
                    {
                        "test_item_id": item.test_item_id,
                        "is_auto_analyzed": False,
                        "issue_type": issue_type,
                        "timestamp": timestamp,
                        "issue_comment": issue_comment,
                    }
                )

        if history_updates:
            self.os_client.bulk_update_issue_history(project_id, history_updates)

        items_not_updated = [int(test_item_id) for test_item_id in set(test_item_ids) - set(found_test_items)]
        LOGGER.debug("Not updated test items: %s", items_not_updated)
        LOGGER.info("Finished updating defect types. It took %.2f sec", time() - t_start)
        return items_not_updated
