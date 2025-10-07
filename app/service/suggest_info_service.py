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
from datetime import datetime
from time import time
from typing import Any, Optional

import opensearchpy.helpers

from app.amqp.amqp import AmqpClient
from app.commons import logging
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import ApplicationConfig, BulkResponse, CleanIndexStrIds
from app.commons.model.ml import ModelType, TrainInfo
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.suggestInfoService")


class SuggestInfoService:
    """This service saves and manage `SuggestAnalysisResult` entities to {project_id}_suggest ES/OS index.

    This is necessary for further use in custom model training.
    """

    app_config: ApplicationConfig
    es_client: EsClient
    rp_suggest_index_template: str
    rp_suggest_metrics_index_template: str

    def __init__(self, app_config: ApplicationConfig, es_client: Optional[EsClient] = None):
        """Initialize SuggestInfoService

        :param app_config: Application configuration object
        :param es_client: Optional EsClient instance. If not provided, a new one will be created.
        """
        self.app_config = app_config
        self.es_client = es_client or EsClient(app_config=self.app_config)
        self.rp_suggest_index_template = "rp_suggestions_info"
        self.rp_suggest_metrics_index_template = "rp_suggestions_info_metrics"

    def build_index_name(self, project_id):
        return str(project_id) + "_suggest"

    @utils.ignore_warnings
    def index_suggest_info(self, suggest_info_list):
        LOGGER.info("Started saving suggest_info_list")
        t_start = time()
        bodies = []
        project_index_names = set()
        if len(suggest_info_list):
            self.es_client.create_index_for_stats_info(self.rp_suggest_metrics_index_template)
        metrics_data_by_test_item = {}
        for obj in suggest_info_list:
            obj_info = json.loads(obj.json())
            obj_info["savedDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            obj_info["modelInfo"] = [obj.strip() for obj in obj_info["modelInfo"].split(";") if obj.strip()]
            obj_info["module_version"] = [self.app_config.appVersion]
            if obj_info["testItem"] not in metrics_data_by_test_item:
                metrics_data_by_test_item[obj_info["testItem"]] = []
            metrics_data_by_test_item[obj_info["testItem"]].append(obj_info)
            project_index_name = self.build_index_name(obj_info["project"])
            project_index_name = text_processing.unite_project_name(
                project_index_name, self.app_config.esProjectIndexPrefix
            )
            if project_index_name not in project_index_names:
                self.es_client.create_index_for_stats_info(
                    self.rp_suggest_index_template, override_index_name=project_index_name
                )
                project_index_names.add(project_index_name)
            bodies.append({"_index": project_index_name, "_source": obj_info})
        bulk_result = self.es_client.bulk_index(bodies)
        self.index_data_for_metrics(metrics_data_by_test_item)
        LOGGER.info("Finished saving %.2f s", time() - t_start)
        return bulk_result

    def index_data_for_metrics(self, metrics_data_by_test_item):
        bodies = []
        for test_item in metrics_data_by_test_item:
            sorted_metrics_data = sorted(metrics_data_by_test_item[test_item], key=lambda x: x["resultPosition"])
            chosen_data = sorted_metrics_data[0]
            for result in sorted_metrics_data:
                if result["userChoice"] == 1:
                    chosen_data = result
                    break
            if chosen_data["methodName"] == "auto_analysis":
                continue
            chosen_data["notFoundResults"] = 0
            if chosen_data["userChoice"] == 1:
                chosen_data["reciprocalRank"] = 1 / (chosen_data["resultPosition"] + 1)
            else:
                chosen_data["reciprocalRank"] = 0.0
            chosen_data["reciprocalRank"] = int(chosen_data["reciprocalRank"] * 100)
            bodies.append({"_index": self.rp_suggest_metrics_index_template, "_source": chosen_data})
        self.es_client.bulk_index(bodies)

    def remove_suggest_info(self, project_id):
        LOGGER.info("Removing suggest_info index")
        project_index_name = self.build_index_name(project_id)
        project_index_name = text_processing.unite_project_name(
            project_index_name, self.app_config.esProjectIndexPrefix
        )
        return self.es_client.delete_index(project_index_name)

    def build_suggest_info_ids_query(self, log_ids: list[str]) -> dict[str, Any]:
        return {
            "_source": ["testItem"],
            "size": self.app_config.esChunkNumber,
            "query": {
                "bool": {"should": [{"terms": {"testItemLogId": log_ids}}, {"terms": {"relevantLogId": log_ids}}]}
            },
        }

    def build_suggest_info_ids_query_by_test_item(self, test_item_ids):
        return {
            "query": {
                "bool": {
                    "should": [{"terms": {"testItem": test_item_ids}}, {"terms": {"relevantItem": test_item_ids}}]
                }
            }
        }

    def build_suggest_info_ids_query_by_launch_ids(self, launch_ids):
        return {"query": {"bool": {"filter": [{"terms": {"launchId": launch_ids}}]}}}

    def clean_suggest_info_logs(self, clean_index: CleanIndexStrIds):
        """Delete logs from OpenSearch"""
        index_name = self.build_index_name(clean_index.project)
        index_name = text_processing.unite_project_name(index_name, self.app_config.esProjectIndexPrefix)
        LOGGER.info("Delete logs %s for the index %s", clean_index.ids, index_name)
        t_start = time()
        if not self.es_client.index_exists(index_name, print_error=False):
            LOGGER.info("Didn't find index '%s'", index_name)
            return 0
        suggest_log_ids = set()
        try:
            search_query = self.build_suggest_info_ids_query(clean_index.ids)
            for res in opensearchpy.helpers.scan(
                self.es_client.es_client, query=search_query, index=index_name, scroll="5m"
            ):
                suggest_log_ids.add(res["_id"])
        except Exception as exc:
            LOGGER.exception("Couldn't find logs with specified ids", exc_info=exc)
        bodies = []
        for _id in suggest_log_ids:
            bodies.append(
                {
                    "_op_type": "delete",
                    "_id": _id,
                    "_index": index_name,
                }
            )
        result = self.es_client.bulk_index(bodies)
        LOGGER.info(
            "Finished deleting logs %s for the project %s. It took %.2f sec",
            clean_index.ids,
            index_name,
            time() - t_start,
        )
        return result.took

    def clean_suggest_info_logs_by_test_item(self, remove_items_info):
        """Delete logs from OpenSearch"""
        index_name = self.build_index_name(remove_items_info["project"])
        index_name = text_processing.unite_project_name(index_name, self.app_config.esProjectIndexPrefix)
        LOGGER.info("Delete test items %s for the index %s", remove_items_info["itemsToDelete"], index_name)
        t_start = time()
        deleted_logs = self.es_client.delete_by_query(
            index_name, remove_items_info["itemsToDelete"], self.build_suggest_info_ids_query_by_test_item
        )
        LOGGER.info(
            "Finished deleting logs %s for the project %s. It took %.2f sec",
            remove_items_info["itemsToDelete"],
            index_name,
            time() - t_start,
        )
        return deleted_logs

    def clean_suggest_info_logs_by_launch_id(self, launch_remove_info):
        """Delete logs with specified launch ids from OpenSearch"""
        project = launch_remove_info["project"]
        launch_ids = launch_remove_info["launch_ids"]
        index_name = self.build_index_name(project)
        index_name = text_processing.unite_project_name(index_name, self.app_config.esProjectIndexPrefix)
        LOGGER.info("Delete launches %s for the index %s", launch_ids, index_name)
        t_start = time()
        deleted_logs = self.es_client.delete_by_query(
            index_name, launch_ids, self.build_suggest_info_ids_query_by_launch_ids
        )
        LOGGER.info(
            "Finished deleting launches %s for the index %s. It took %.2f sec. %s logs deleted",
            launch_ids,
            index_name,
            time() - t_start,
            deleted_logs,
        )
        return deleted_logs

    def build_query_for_getting_suggest_info(self, test_item_ids):
        return {
            "_source": ["testItem", "issueType"],
            "size": self.app_config.esChunkNumber,
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"testItem": test_item_ids}},
                        {"term": {"methodName": "auto_analysis"}},
                        {"term": {"userChoice": 1}},
                    ]
                }
            },
        }

    def update_train_data(self, project_id: int, result: BulkResponse) -> None:
        if self.app_config.amqpUrl:
            amqp_client = AmqpClient(self.app_config)
            for model_type in [ModelType.suggestion, ModelType.auto_analysis]:
                amqp_client.send_to_inner_queue(
                    "train_models",
                    TrainInfo(model_type=model_type, project=project_id, gathered_metric_total=result.took).json(),
                )
            amqp_client.close()

    def update_suggest_info(self, defect_update_info):
        LOGGER.info("Started updating suggest info")
        t_start = time()
        test_item_ids = [int(key_) for key_ in defect_update_info["itemsToUpdate"].keys()]
        defect_update_info["itemsToUpdate"] = {
            int(key_): val for key_, val in defect_update_info["itemsToUpdate"].items()
        }
        index_name = self.build_index_name(defect_update_info["project"])
        index_name = text_processing.unite_project_name(index_name, self.app_config.esProjectIndexPrefix)
        if not self.es_client.index_exists(index_name):
            return 0
        batch_size = 1000
        log_update_queries = []
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            sub_test_item_ids = test_item_ids[i * batch_size : (i + 1) * batch_size]
            if not sub_test_item_ids:
                continue
            for res in opensearchpy.helpers.scan(
                self.es_client.es_client,
                query=self.build_query_for_getting_suggest_info(sub_test_item_ids),
                index=index_name,
            ):
                issue_type = ""
                try:
                    test_item_id = int(res["_source"]["testItem"])
                    issue_type = defect_update_info["itemsToUpdate"][test_item_id]
                except:  # noqa
                    pass
                if issue_type.strip() and issue_type != res["_source"]["issueType"]:
                    log_update_queries.append(
                        {"_op_type": "update", "_id": res["_id"], "_index": index_name, "doc": {"userChoice": 0}}
                    )
        result = self.es_client.bulk_index(log_update_queries)
        self.update_train_data(defect_update_info["project"], result)
        LOGGER.info("Finished updating suggest info for %.2f sec.", time() - t_start)
        return result.took
