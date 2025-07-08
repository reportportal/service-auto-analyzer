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
import traceback
from collections import deque
from time import time
from typing import Optional

import elasticsearch
import elasticsearch.helpers
import requests
import urllib3
from elasticsearch import RequestsHttpConnection
from urllib3.exceptions import InsecureRequestWarning

from app.amqp.amqp import AmqpClient
from app.commons import log_merger, logging, request_factory
from app.commons.model.launch_objects import ApplicationConfig, BulkResponse, Launch, Response, TestItem
from app.commons.model.ml import ModelType, TrainInfo
from app.utils import text_processing, utils

ES_URL_MESSAGE = "ES Url %s"

logger = logging.getLogger("analyzerApp.esclient")


class EsClient:
    """Elasticsearch client implementation"""

    app_config: ApplicationConfig
    es_client: elasticsearch.Elasticsearch
    host: str
    tables_to_recreate: list[str]

    def __init__(self, app_config: ApplicationConfig, es_client: elasticsearch.Elasticsearch = None):
        self.app_config = app_config
        self.host = app_config.esHost
        if es_client:
            logger.info("Creating service using provided client")
        else:
            logger.info(f"Creating service using host URL: {text_processing.remove_credentials_from_url(self.host)}")
        self.es_client = es_client or self.create_es_client(app_config)
        self.tables_to_recreate = ["rp_aa_stats", "rp_model_train_stats", "rp_suggestions_info_metrics"]

    def create_es_client(self, app_config: ApplicationConfig) -> elasticsearch.Elasticsearch:
        if not app_config.esVerifyCerts:
            urllib3.disable_warnings(InsecureRequestWarning)
        kwargs = {
            "timeout": 30,
            "max_retries": 5,
            "retry_on_timeout": True,
            "use_ssl": app_config.esUseSsl,
            "verify_certs": app_config.esVerifyCerts,
            "ssl_show_warn": app_config.esSslShowWarn,
            "ca_certs": app_config.esCAcert,
            "client_cert": app_config.esClientCert,
            "client_key": app_config.esClientKey,
        }

        if app_config.esUser:
            kwargs["http_auth"] = (app_config.esUser, app_config.esPassword)

        if app_config.turnOffSslVerification:
            kwargs["connection_class"] = RequestsHttpConnection

        return elasticsearch.Elasticsearch([self.host], **kwargs)

    def get_test_item_query(self, test_item_ids, is_merged, full_log):
        """Build test item query"""
        if full_log:
            return {
                "size": self.app_config.esChunkNumber,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"test_item": [str(_id) for _id in test_item_ids]}},
                            {"term": {"is_merged": is_merged}},
                        ]
                    }
                },
            }
        else:
            return {
                "_source": ["test_item"],
                "size": self.app_config.esChunkNumber,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"test_item": [str(_id) for _id in test_item_ids]}},
                            {"term": {"is_merged": is_merged}},
                        ]
                    }
                },
            }

    def build_search_test_item_ids_query(self, log_ids):
        """Build search test item ids query"""
        return {
            "_source": ["test_item"],
            "size": self.app_config.esChunkNumber,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}},
                        {"terms": {"_id": [str(log_id) for log_id in log_ids]}},
                    ]
                }
            },
        }

    def __get_base_url(self):
        """Get base URL for Elasticsearch"""
        if not self.host:
            logger.error("Elasticsearch host is not set")
            return None

        if self.host.startswith("http"):
            return self.host
        else:
            protocol = "https" if self.app_config.esUseSsl else "http"
            return f"{protocol}://{self.host}"

    def is_healthy(self) -> bool:
        """Check whether elasticsearch is healthy"""
        base_url = self.__get_base_url()
        if not base_url:
            return False

        try:
            url = text_processing.build_url(base_url, ["_cluster/health"])
            res = utils.send_request(url, "GET", self.app_config.esUser, self.app_config.esPassword)
            return res["status"] in ["green", "yellow"]
        except Exception as err:
            logger.error("Elasticsearch is not healthy")
            logger.error(err)
            return False

    def update_settings_after_read_only(self) -> None:
        base_url = self.__get_base_url()
        if not base_url:
            return

        try:
            requests.put(
                f"{base_url}/_all/_settings",
                headers={"Content-Type": "application/json"},
                data='{"index.blocks.read_only_allow_delete": null}',
            ).raise_for_status()
        except Exception as err:
            logger.error(err)
            logger.error("Can't reset read only mode for elastic indices")

    def create_index(self, index_name: str) -> Response:
        """Create index in elasticsearch"""
        logger.info(f"Creating index: {index_name}")
        response = self.es_client.indices.create(
            index=index_name,
            body={
                "settings": utils.read_json_file("res", "index_settings.json", to_json=True),
                "mappings": utils.read_json_file("res", "index_mapping_settings.json", to_json=True),
            },
        )
        logger.debug(f"Index '{index_name}' created")
        return Response(**response)

    def list_indices(self) -> Optional[list]:
        """Get all indices from elasticsearch"""
        base_url = self.__get_base_url()
        if not base_url:
            return None

        url = text_processing.build_url(base_url, ["_cat", "indices?format=json"])
        res = utils.send_request(url, "GET", self.app_config.esUser, self.app_config.esPassword)
        return res

    def index_exists(self, index_name: str, print_error: bool = True):
        """Checks whether index exists"""
        try:
            index = self.es_client.indices.get(index=str(index_name))
            return index is not None
        except Exception as err:
            if print_error:
                logger.exception(f"Index '{index_name}' was not found", exc_info=err)
            return False

    def delete_index(self, index_name):
        """Delete the whole index"""
        logger.info(f"Deleting index: {index_name}")
        try:
            self.es_client.indices.delete(index=str(index_name))
            logger.debug(f"Index '{str(index_name)}' deleted")
            return True
        except Exception as err:
            logger.exception(f"Failed to delete index: {str(index_name)}", exc_info=err)
            return False

    def create_index_if_not_exists(self, index_name: str) -> bool:
        """Creates index if it doesn't exist"""
        if not self.index_exists(index_name, print_error=False):
            response = self.create_index(index_name)
            return response.acknowledged
        return True

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

    def index_logs(self, launches: list[Launch]):
        """Index launches to the index with project name"""
        launch_ids = set(map(lambda launch_obj: str(launch_obj.launchId), launches))
        launch_ids_str = ", ".join(launch_ids)
        project = next(map(lambda launch_obj: launch_obj.project, launches))
        logger.info(f"Indexing {len(launch_ids)} launches of project '{project}': {launch_ids_str}")
        t_start = time()
        test_item_queue = self._to_launch_test_item_list(launches)
        del launches
        if project is None:
            return BulkResponse(took=0, errors=False)

        project_with_prefix = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        self.create_index_if_not_exists(project_with_prefix)
        test_item_ids, bodies = self._to_index_bodies(project_with_prefix, test_item_queue)
        logs_with_exceptions = utils.extract_all_exceptions(bodies)
        result = self._bulk_index(bodies)
        result.logResults = logs_with_exceptions
        _, num_logs_with_defect_types = self._merge_logs(test_item_ids, project_with_prefix)

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
        logger.info(
            f"Indexing {len(launch_ids)} launches of project '{project}' finished: {launch_ids_str}. "
            f"It took {time_passed} sec."
        )
        return result

    def _merge_logs(self, test_item_ids, project):
        bodies = []
        batch_size = 1000
        self._delete_merged_logs(test_item_ids, project)
        num_logs_with_defect_types = 0
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            test_items = test_item_ids[i * batch_size : (i + 1) * batch_size]
            if not test_items:
                continue
            test_items_dict = {}
            for r in elasticsearch.helpers.scan(
                self.es_client, query=self.get_test_item_query(test_items, False, True), index=project
            ):
                test_item_id = r["_source"]["test_item"]
                if test_item_id not in test_items_dict:
                    test_items_dict[test_item_id] = []
                test_items_dict[test_item_id].append(r)
            for test_item_id in test_items_dict:
                merged_logs, _ = log_merger.decompose_logs_merged_and_without_duplicates(test_items_dict[test_item_id])
                for log in merged_logs:
                    if log["_source"]["is_merged"]:
                        bodies.append(log)
                    else:
                        bodies.append(
                            {
                                "_op_type": "update",
                                "_id": log["_id"],
                                "_index": log["_index"],
                                "doc": {"merged_small_logs": log["_source"]["merged_small_logs"]},
                            }
                        )
                    log_issue_type = log["_source"]["issue_type"]
                    if log_issue_type.strip() and not log_issue_type.lower().startswith("ti"):
                        num_logs_with_defect_types += 1
        return self._bulk_index(bodies), num_logs_with_defect_types

    def _delete_merged_logs(self, test_items_to_delete, project):
        logger.debug("Delete merged logs for %d test items", len(test_items_to_delete))
        bodies = []
        batch_size = 1000
        for i in range(int(len(test_items_to_delete) / batch_size) + 1):
            test_item_ids = test_items_to_delete[i * batch_size : (i + 1) * batch_size]
            if not test_item_ids:
                continue
            for log in elasticsearch.helpers.scan(
                self.es_client, query=self.get_test_item_query(test_item_ids, True, False), index=project
            ):
                bodies.append({"_op_type": "delete", "_id": log["_id"], "_index": project})
        if bodies:
            self._bulk_index(bodies)

    def _recreate_index_if_needed(self, bodies, formatted_exception):
        index_name = ""
        if bodies:
            index_name = bodies[0]["_index"]
        if not index_name.strip():
            return
        if (
            "'type': 'mapper_parsing_exception'" in formatted_exception
            or "RequestError(400, 'illegal_argument_exception'" in formatted_exception
        ) and index_name in self.tables_to_recreate:
            self.delete_index(index_name)
            self.create_index_for_stats_info(index_name)

    def _bulk_index(self, bodies, refresh=True, chunk_size=None):
        if not bodies:
            return BulkResponse(took=0, errors=False)
        start_time = time()
        logger.debug(f"Indexing {len(bodies)} logs")
        es_chunk_number = self.app_config.esChunkNumber
        if chunk_size is not None:
            es_chunk_number = chunk_size
        try:
            try:
                success_count, errors = elasticsearch.helpers.bulk(
                    self.es_client, bodies, chunk_size=es_chunk_number, request_timeout=30, refresh=refresh
                )
            except:  # noqa
                formatted_exception = traceback.format_exc()
                self._recreate_index_if_needed(bodies, formatted_exception)
                self.update_settings_after_read_only()
                success_count, errors = elasticsearch.helpers.bulk(
                    self.es_client, bodies, chunk_size=es_chunk_number, request_timeout=30, refresh=refresh
                )
            error_str = ""
            if errors:
                error_str = ", ".join([str(error) for error in errors])
            logger.debug(
                f"{success_count} logs were successfully indexed{'. Errors:' + error_str if error_str else ''}"
            )
            logger.debug("Finished indexing for %.2f s", time() - start_time)
            return BulkResponse(took=success_count, errors=len(errors) > 0)
        except Exception as exc:
            logger.exception("Error in bulk indexing", exc_info=exc)
            return BulkResponse(took=0, errors=True)

    def delete_logs(self, clean_index):
        """Delete logs from elasticsearch"""
        index_name = text_processing.unite_project_name(clean_index.project, self.app_config.esProjectIndexPrefix)
        log_ids = ", ".join(map(lambda log_id: str(log_id), clean_index.ids))
        logger.info(f"Delete project '{index_name}' logs: {log_ids}")
        t_start = time()
        if not self.index_exists(index_name):
            return 0
        test_item_ids = set()
        try:
            search_query = self.build_search_test_item_ids_query(clean_index.ids)
            for res in elasticsearch.helpers.scan(self.es_client, query=search_query, index=index_name, scroll="5m"):
                test_item_ids.add(res["_source"]["test_item"])
        except Exception as err:
            logger.exception("Couldn't find test items for logs", exc_info=err)

        bodies = []
        for _id in clean_index.ids:
            bodies.append(
                {
                    "_op_type": "delete",
                    "_id": _id,
                    "_index": index_name,
                }
            )
        result = self._bulk_index(bodies)
        self._merge_logs(list(test_item_ids), index_name)
        logger.info(
            f"Finished deleting project '{index_name}' logs: {log_ids}. It took {round(time() - t_start, 2)} sec"
        )
        return result.took

    def create_index_for_stats_info(self, rp_aa_stats_index, override_index_name=None):
        index_name = rp_aa_stats_index
        if override_index_name is not None:
            index_name = override_index_name
        index = None
        try:
            index = self.es_client.indices.get(index=index_name)
        except Exception:
            pass
        if index is None:
            self.es_client.indices.create(
                index=index_name,
                body={
                    "settings": utils.read_json_file("res", "index_settings.json", to_json=True),
                    "mappings": utils.read_json_file("res", "%s_mappings.json" % rp_aa_stats_index, to_json=True),
                },
            )
        else:
            try:
                self.es_client.indices.put_mapping(
                    index=index_name,
                    body=utils.read_json_file("res", "%s_mappings.json" % rp_aa_stats_index, to_json=True),
                )
            except:  # noqa
                formatted_exception = traceback.format_exc()
                self._recreate_index_if_needed([{"_index": index_name}], formatted_exception)

    def send_stats_info(self, stats_info: dict) -> None:
        logger.info("Started sending stats about analysis")

        stat_info_array = []
        for obj_info in stats_info.values():
            rp_aa_stats_index = "rp_aa_stats"
            if "method" in obj_info and obj_info["method"] == "training":
                rp_aa_stats_index = "rp_model_train_stats"
            self.create_index_for_stats_info(rp_aa_stats_index)
            stat_info_array.append({"_index": rp_aa_stats_index, "_source": obj_info})
        self._bulk_index(stat_info_array)
        logger.info("Finished sending stats about analysis")

    def get_test_items_by_ids_query(self, test_item_ids):
        return {
            "_source": ["test_item"],
            "size": self.app_config.esChunkNumber,
            "query": {"bool": {"filter": [{"terms": {"test_item": test_item_ids}}]}},
        }

    @utils.ignore_warnings
    def defect_update(self, defect_update_info):
        logger.info("Started updating defect types")
        t_start = time()
        test_item_ids = [int(key_) for key_ in defect_update_info["itemsToUpdate"].keys()]
        defect_update_info["itemsToUpdate"] = {
            int(key_): val for key_, val in defect_update_info["itemsToUpdate"].items()
        }
        index_name = text_processing.unite_project_name(
            defect_update_info["project"], self.app_config.esProjectIndexPrefix
        )
        if not self.index_exists(index_name):
            return test_item_ids
        batch_size = 1000
        log_update_queries = []
        found_test_items = set()
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            sub_test_item_ids = test_item_ids[i * batch_size : (i + 1) * batch_size]
            if not sub_test_item_ids:
                continue
            for log in elasticsearch.helpers.scan(
                self.es_client, query=self.get_test_items_by_ids_query(sub_test_item_ids), index=index_name
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
        self._bulk_index(log_update_queries)
        items_not_updated = list(set(test_item_ids) - found_test_items)
        logger.debug("Not updated test items: %s", items_not_updated)
        if self.app_config.amqpUrl:
            amqp_client = AmqpClient(self.app_config)
            amqp_client.send_to_inner_queue("update_suggest_info", json.dumps(defect_update_info))
            amqp_client.close()
        logger.info("Finished updating defect types. It took %.2f sec", time() - t_start)
        return items_not_updated

    def build_delete_query_by_test_items(self, sub_test_item_ids):
        return {"query": {"bool": {"filter": [{"terms": {"test_item": sub_test_item_ids}}]}}}

    def build_delete_query_by_launch_ids(self, launch_ids):
        return {"query": {"bool": {"filter": [{"terms": {"launch_id": launch_ids}}]}}}

    @utils.ignore_warnings
    def remove_test_items(self, remove_items_info):
        logger.info("Started removing test items")
        t_start = time()
        index_name = text_processing.unite_project_name(
            str(remove_items_info["project"]), self.app_config.esProjectIndexPrefix
        )
        deleted_logs = self.delete_by_query(
            index_name, remove_items_info["itemsToDelete"], self.build_delete_query_by_test_items
        )
        logger.debug("Removed %s logs by test item ids", deleted_logs)
        logger.info("Finished removing test items. It took %.2f sec", time() - t_start)
        return deleted_logs

    @utils.ignore_warnings
    def remove_launches(self, remove_launches_info):
        project = remove_launches_info["project"]
        launch_ids = remove_launches_info["launch_ids"]
        logger.info("Started removing launches")
        t_start = time()
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        deleted_logs = self.delete_by_query(
            index_name,
            launch_ids,
            self.build_delete_query_by_launch_ids,
        )
        logger.debug("Removed %s logs by launch ids", deleted_logs)
        logger.info("Finished removing launches. It took %.2f sec", time() - t_start)
        return deleted_logs

    @utils.ignore_warnings
    def delete_by_query(self, index_name, ids_for_removal, delete_query_deriver):
        if not self.index_exists(index_name):
            return 0
        batch_size = 1000
        deleted_logs = 0
        for i in range(int(len(ids_for_removal) / batch_size) + 1):
            sub_ids_for_removal = ids_for_removal[i * batch_size : (i + 1) * batch_size]
            if not sub_ids_for_removal:
                continue
            result = self.es_client.delete_by_query(index_name, body=delete_query_deriver(sub_ids_for_removal))
            if "deleted" in result:
                deleted_logs += result["deleted"]
        return deleted_logs

    def __time_range_query(
        self,
        time_field: str,
        gte_time: str,
        lte_time: str,
        for_scan: bool = False,
    ) -> dict:
        query = {"query": {"range": {time_field: {"gte": gte_time, "lte": lte_time}}}}
        if for_scan:
            query["size"] = self.app_config.esChunkNumber
        return query

    @utils.ignore_warnings
    def get_launch_ids_by_start_time_range(self, project: int, start_date: str, end_date: str) -> list[str]:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("launch_start_time", start_date, end_date, for_scan=True)
        launch_ids = set()
        for log in elasticsearch.helpers.scan(self.es_client, query=query, index=index_name):
            launch_ids.add(log["_source"]["launch_id"])
        return list(launch_ids)

    @utils.ignore_warnings
    def remove_by_launch_start_time_range(self, project: int, start_date: str, end_date: str) -> int:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("launch_start_time", start_date, end_date)
        delete_response = self.es_client.delete_by_query(index_name, body=query)
        return delete_response["deleted"]

    @utils.ignore_warnings
    def get_log_ids_by_log_time_range(self, project: int, start_date: str, end_date: str) -> list[str]:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("log_time", start_date, end_date, for_scan=True)
        log_ids = set()
        for log in elasticsearch.helpers.scan(self.es_client, query=query, index=index_name):
            log_ids.add(log["_id"])
        return list(log_ids)

    @utils.ignore_warnings
    def remove_by_log_time_range(self, project: int, start_date: str, end_date: str) -> int:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("log_time", start_date, end_date)
        delete_response = self.es_client.delete_by_query(index_name, body=query)
        return delete_response["deleted"]
