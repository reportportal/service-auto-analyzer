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
from collections import deque
from time import time
from typing import Optional

import opensearchpy.helpers

from app.amqp.amqp import AmqpClient
from app.commons import logging, request_factory
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import ApplicationConfig, BulkResponse, Launch, TestItem
from app.commons.model.ml import ModelType, TrainInfo
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.indexService")


class IndexService:
    """Service for indexing launches and managing launch-related data in OpenSearch"""

    app_config: ApplicationConfig
    es_client: EsClient

    def __init__(self, app_config: ApplicationConfig, es_client: Optional[EsClient] = None):
        """Initialize IndexService

        :param app_config: Application configuration object
        :param es_client: Optional EsClient instance. If not provided, a new one will be created.
        """
        self.app_config = app_config
        self.es_client = es_client or EsClient(app_config=self.app_config)

    def _to_launch_test_item_list(self, launches: list[Launch]) -> deque[tuple[Launch, TestItem]]:
        test_item_queue = deque()
        for launch in launches:
            test_items = launch.testItems
            launch.testItems = []
            for test_item in test_items:
                for log in test_item.logs:
                    if str(log.clusterId) in launch.clusters:
                        log.clusterMessage = launch.clusters[str(log.clusterId)]
                test_item_queue.append((launch, test_item))
        return test_item_queue

    def _to_index_bodies(
        self, project_with_prefix: str, test_item_queue: deque[tuple[Launch, TestItem]]
    ) -> tuple[list[str], list[dict]]:
        bodies = []
        test_item_ids = []
        while len(test_item_queue) > 0:
            launch, test_item = test_item_queue.popleft()
            logs_added = False
            for log in test_item.logs:
                if log.logLevel < utils.ERROR_LOGGING_LEVEL or not log.message.strip():
                    continue

                bodies.append(request_factory.prepare_log(launch, test_item, log, project_with_prefix))
                logs_added = True
            if logs_added:
                test_item_ids.append(str(test_item.testItemId))
        return test_item_ids, bodies

    def _delete_merged_logs(self, test_items_to_delete: list[str], project: str) -> None:
        LOGGER.debug("Delete merged logs for %d test items", len(test_items_to_delete))
        bodies = []
        batch_size = 1000
        for i in range(int(len(test_items_to_delete) / batch_size) + 1):
            test_item_ids = test_items_to_delete[i * batch_size : (i + 1) * batch_size]
            if not test_item_ids:
                continue
            for log in opensearchpy.helpers.scan(
                self.es_client.es_client,
                query=self.es_client.get_test_item_query(test_item_ids, True, False),
                index=project,
            ):
                bodies.append({"_op_type": "delete", "_id": log["_id"], "_index": project})
        if bodies:
            self.es_client.bulk_index(bodies)

    def index_logs(self, launches: list[Launch]) -> BulkResponse:
        """Index launches to the index with project name"""
        launch_ids = {str(launch_obj.launchId) for launch_obj in launches}
        launch_ids_str = ", ".join(launch_ids)
        project = launches[0].project if launches else None
        LOGGER.info(f"Indexing {len(launch_ids)} launches of project '{project}': {launch_ids_str}")
        t_start = time()
        test_item_queue = self._to_launch_test_item_list(launches)
        if project is None:
            return BulkResponse(took=0, errors=False)

        project_with_prefix = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        self.es_client.create_index_if_not_exists(project_with_prefix)
        test_item_ids, bodies = self._to_index_bodies(project_with_prefix, test_item_queue)
        logs_with_exceptions = utils.extract_all_exceptions(bodies)
        result = self.es_client.bulk_index(bodies)
        result.logResults = logs_with_exceptions
        _, num_logs_with_defect_types = self.es_client.merge_logs(test_item_ids, project_with_prefix)

        if self.app_config.amqpUrl:
            amqp_client = AmqpClient(self.app_config)
            amqp_client.send_to_inner_queue(
                "train_models",
                TrainInfo(
                    model_type=ModelType.defect_type, project=project, gathered_metric_total=num_logs_with_defect_types
                ).json(),
            )
            amqp_client.close()

        time_passed = round(time() - t_start, 2)
        LOGGER.info(
            f"Indexing {len(launch_ids)} launches of project '{project}' finished: {launch_ids_str}. "
            f"It took {time_passed} sec."
        )
        return result

    def send_stats_info(self, stats_info: dict) -> None:
        LOGGER.info("Started sending stats about analysis")

        stat_info_array = []
        for obj_info in stats_info.values():
            rp_aa_stats_index = "rp_aa_stats"
            if "method" in obj_info and obj_info["method"] == "training":
                rp_aa_stats_index = "rp_model_train_stats"
            self.es_client.create_index_for_stats_info(rp_aa_stats_index)
            stat_info_array.append({"_index": rp_aa_stats_index, "_source": obj_info})
        self.es_client.bulk_index(stat_info_array)
        LOGGER.info("Finished sending stats about analysis")

    @utils.ignore_warnings
    def defect_update(self, defect_update_info: dict) -> list[int]:
        LOGGER.info("Started updating defect types")
        t_start = time()
        test_item_ids = [int(key_) for key_ in defect_update_info["itemsToUpdate"].keys()]
        defect_update_info["itemsToUpdate"] = {
            int(key_): val for key_, val in defect_update_info["itemsToUpdate"].items()
        }
        index_name = text_processing.unite_project_name(
            defect_update_info["project"], self.app_config.esProjectIndexPrefix
        )
        if not self.es_client.index_exists(index_name):
            return test_item_ids
        batch_size = 1000
        log_update_queries = []
        found_test_items = set()
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            sub_test_item_ids = test_item_ids[i * batch_size : (i + 1) * batch_size]
            if not sub_test_item_ids:
                continue
            for log in opensearchpy.helpers.scan(
                self.es_client.es_client,
                query=self.es_client.get_test_items_by_ids_query(sub_test_item_ids),
                index=index_name,
            ):
                issue_type = ""
                try:
                    test_item_id = int(log["_source"]["test_item"])
                    found_test_items.add(test_item_id)
                    issue_type = defect_update_info["itemsToUpdate"][test_item_id]
                except:  # noqa
                    pass
                if issue_type.strip():
                    log_update_queries.append(
                        {
                            "_op_type": "update",
                            "_id": log["_id"],
                            "_index": index_name,
                            "doc": {"issue_type": issue_type, "is_auto_analyzed": False},
                        }
                    )
        self.es_client.bulk_index(log_update_queries)
        items_not_updated = list(set(test_item_ids) - found_test_items)
        LOGGER.debug("Not updated test items: %s", items_not_updated)
        if self.app_config.amqpUrl:
            amqp_client = AmqpClient(self.app_config)
            amqp_client.send_to_inner_queue("update_suggest_info", json.dumps(defect_update_info))
            amqp_client.close()
        LOGGER.info("Finished updating defect types. It took %.2f sec", time() - t_start)
        return items_not_updated
