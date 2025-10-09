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

import traceback
from time import time
from typing import Any, Callable, Optional

import opensearchpy.helpers
import urllib3
from opensearchpy import OpenSearch, RequestsHttpConnection
from urllib3.exceptions import InsecureRequestWarning

from app.commons import log_merger, logging
from app.commons.model.launch_objects import ApplicationConfig, BulkResponse, CleanIndex, Response
from app.utils import text_processing, utils

ES_URL_MESSAGE = "OpenSearch Url %s"
TABLES_TO_RECREATE = ["rp_aa_stats", "rp_model_train_stats", "rp_suggestions_info_metrics"]

LOGGER = logging.getLogger("analyzerApp.esclient")


class EsClient:
    """OpenSearch client implementation"""

    app_config: ApplicationConfig
    es_client: OpenSearch
    host: str

    def __init__(self, app_config: ApplicationConfig, es_client: Optional[OpenSearch] = None) -> None:
        self.app_config = app_config
        self.host = app_config.esHost
        if es_client:
            LOGGER.debug("Creating service using provided client")
            self.es_client = es_client
        else:
            LOGGER.debug(f"Creating service using host URL: {text_processing.remove_credentials_from_url(self.host)}")
            self.es_client = self.create_es_client(app_config)

    def create_es_client(self, app_config: ApplicationConfig) -> OpenSearch:
        if not app_config.esVerifyCerts:
            urllib3.disable_warnings(InsecureRequestWarning)
        kwargs: dict[str, Any] = {
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

        return OpenSearch([self.host], **kwargs)

    def get_test_item_query(self, test_item_ids: list, is_merged: bool, full_log: bool) -> dict[str, Any]:
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

    def build_search_test_item_ids_query(self, log_ids: list) -> dict[str, Any]:
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

    def __get_base_url(self) -> Optional[str]:
        """Get base URL for OpenSearch"""
        if not self.host:
            LOGGER.error("OpenSearch host is not set")
            return None
        url = self.host
        if not self.host.startswith("http"):
            protocol = "https" if self.app_config.esUseSsl else "http"
            url = f"{protocol}://{self.host}"
        return url

    def is_healthy(self) -> bool:
        """Check whether OpenSearch is healthy"""
        base_url = self.__get_base_url()
        if not base_url:
            return False

        url = text_processing.build_url(base_url, ["_cluster/health"])
        res = utils.send_request(url, "GET", self.app_config.esUser, self.app_config.esPassword)
        return res.get("status", "") in {"green", "yellow"} if res else False

    def update_settings_after_read_only(self) -> None:
        base_url = self.__get_base_url()
        if not base_url:
            return

        utils.send_request(
            f"{base_url}/_all/_settings",
            "PUT",
            self.app_config.esUser,
            self.app_config.esPassword,
            data='{"index.blocks.read_only_allow_delete": null}',
        )

    def create_index(self, index_name: str) -> Response:
        """Create index in OpenSearch"""
        LOGGER.info(f"Creating index: {index_name}")
        response = self.es_client.indices.create(
            index=index_name,
            body={
                "settings": utils.read_json_file("res", "index_settings.json", to_json=True),
                "mappings": utils.read_json_file("res", "index_mapping_settings.json", to_json=True),
            },
        )
        LOGGER.debug(f"Index '{index_name}' created")
        return Response(**response)

    def index_exists(self, index_name: str, print_error: bool = True) -> bool:
        """Checks whether index exists"""
        try:
            index = self.es_client.indices.get(index=str(index_name))
            return index is not None
        except Exception as err:
            if print_error:
                LOGGER.exception(f"Index '{index_name}' was not found", exc_info=err)
            return False

    def delete_index(self, index_name: str) -> bool:
        """Delete the whole index"""
        LOGGER.info(f"Deleting index: {index_name}")
        try:
            self.es_client.indices.delete(index=str(index_name))
            LOGGER.debug(f"Index '{str(index_name)}' deleted")
            return True
        except Exception as err:
            LOGGER.exception(f"Failed to delete index: {str(index_name)}", exc_info=err)
            return False

    def create_index_if_not_exists(self, index_name: str) -> bool:
        """Creates index if it doesn't exist"""
        if not self.index_exists(index_name, print_error=False):
            response = self.create_index(index_name)
            return response.acknowledged
        return True

    def _delete_merged_logs(self, test_items_to_delete: list, project: str) -> None:
        LOGGER.debug("Delete merged logs for %d test items", len(test_items_to_delete))
        bodies = []
        batch_size = 1000
        for i in range(int(len(test_items_to_delete) / batch_size) + 1):
            test_item_ids = test_items_to_delete[i * batch_size : (i + 1) * batch_size]
            if not test_item_ids:
                continue
            for log in opensearchpy.helpers.scan(
                self.es_client, query=self.get_test_item_query(test_item_ids, True, False), index=project
            ):
                bodies.append({"_op_type": "delete", "_id": log["_id"], "_index": project})
        if bodies:
            self.bulk_index(bodies)

    def merge_logs(self, test_item_ids: list, project: str) -> tuple[BulkResponse, int]:
        bodies = []
        batch_size = 1000
        self._delete_merged_logs(test_item_ids, project)
        num_logs_with_defect_types = 0
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            test_items = test_item_ids[i * batch_size : (i + 1) * batch_size]
            if not test_items:
                continue
            test_items_dict = {}
            for r in opensearchpy.helpers.scan(
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
        return self.bulk_index(bodies), num_logs_with_defect_types

    def _recreate_index_if_needed(self, bodies: list[dict[str, Any]], formatted_exception: str) -> None:
        index_name = ""
        if bodies:
            index_name = bodies[0]["_index"]
        if not index_name.strip():
            return
        if (
            "'type': 'mapper_parsing_exception'" in formatted_exception
            or "RequestError(400, 'illegal_argument_exception'" in formatted_exception
        ) and index_name in TABLES_TO_RECREATE:
            self.delete_index(index_name)
            self.create_index_for_stats_info(index_name)

    def bulk_index(
        self, bodies: list[dict[str, Any]], refresh: bool = True, chunk_size: Optional[int] = None
    ) -> BulkResponse:
        if not bodies:
            return BulkResponse(took=0, errors=False)
        start_time = time()
        LOGGER.debug(f"Indexing {len(bodies)} logs")
        es_chunk_number = self.app_config.esChunkNumber
        if chunk_size is not None:
            es_chunk_number = chunk_size
        try:
            try:
                success_count, errors = opensearchpy.helpers.bulk(
                    self.es_client, bodies, chunk_size=es_chunk_number, request_timeout=30, refresh=refresh
                )
            except:  # noqa
                formatted_exception = traceback.format_exc()
                self._recreate_index_if_needed(bodies, formatted_exception)
                self.update_settings_after_read_only()
                success_count, errors = opensearchpy.helpers.bulk(
                    self.es_client, bodies, chunk_size=es_chunk_number, request_timeout=30, refresh=refresh
                )
            error_str = ""
            if errors:
                error_str = ", ".join([str(error) for error in errors])
            LOGGER.debug(
                f"{success_count} logs were successfully indexed{'. Errors:' + error_str if error_str else ''}"
            )
            LOGGER.debug("Finished indexing for %.2f s", time() - start_time)
            return BulkResponse(took=success_count, errors=len(errors) > 0)
        except Exception as exc:
            LOGGER.exception("Error in bulk indexing", exc_info=exc)
            return BulkResponse(took=0, errors=True)

    def delete_logs(self, clean_index: CleanIndex) -> int:
        """Delete logs from OpenSearch"""
        index_name = text_processing.unite_project_name(clean_index.project, self.app_config.esProjectIndexPrefix)
        log_ids = ", ".join([str(log_id) for log_id in clean_index.ids])
        LOGGER.info(f"Delete project '{index_name}' logs: {log_ids}")
        t_start = time()
        if not self.index_exists(index_name):
            return 0
        test_item_ids = set()
        try:
            search_query = self.build_search_test_item_ids_query(clean_index.ids)
            for res in opensearchpy.helpers.scan(self.es_client, query=search_query, index=index_name, scroll="5m"):
                test_item_ids.add(res["_source"]["test_item"])
        except Exception as err:
            LOGGER.exception("Couldn't find test items for logs", exc_info=err)

        bodies = []
        for _id in clean_index.ids:
            bodies.append(
                {
                    "_op_type": "delete",
                    "_id": _id,
                    "_index": index_name,
                }
            )
        result = self.bulk_index(bodies)
        self.merge_logs(list(test_item_ids), index_name)
        LOGGER.info(
            f"Finished deleting project '{index_name}' logs: {log_ids}. It took {round(time() - t_start, 2)} sec"
        )
        return result.took

    def create_index_for_stats_info(self, rp_aa_stats_index: str, override_index_name: Optional[str] = None) -> None:
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

    def get_test_items_by_ids_query(self, test_item_ids: list) -> dict[str, Any]:
        return {
            "_source": ["test_item"],
            "size": self.app_config.esChunkNumber,
            "query": {"bool": {"filter": [{"terms": {"test_item": test_item_ids}}]}},
        }

    def build_delete_query_by_test_items(self, sub_test_item_ids: list) -> dict[str, Any]:
        return {"query": {"bool": {"filter": [{"terms": {"test_item": sub_test_item_ids}}]}}}

    def build_delete_query_by_launch_ids(self, launch_ids: list) -> dict[str, Any]:
        return {"query": {"bool": {"filter": [{"terms": {"launch_id": launch_ids}}]}}}

    @utils.ignore_warnings
    def remove_test_items(self, remove_items_info: dict[str, Any]) -> int:
        LOGGER.info("Started removing test items")
        t_start = time()
        index_name = text_processing.unite_project_name(
            str(remove_items_info["project"]), self.app_config.esProjectIndexPrefix
        )
        deleted_logs = self.delete_by_query(
            index_name, remove_items_info["itemsToDelete"], self.build_delete_query_by_test_items
        )
        LOGGER.debug("Removed %s logs by test item ids", deleted_logs)
        LOGGER.info("Finished removing test items. It took %.2f sec", time() - t_start)
        return deleted_logs

    @utils.ignore_warnings
    def remove_launches(self, remove_launches_info: dict[str, Any]) -> int:
        project = remove_launches_info["project"]
        launch_ids = remove_launches_info["launch_ids"]
        LOGGER.info("Started removing launches")
        t_start = time()
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        deleted_logs = self.delete_by_query(
            index_name,
            launch_ids,
            self.build_delete_query_by_launch_ids,
        )
        LOGGER.debug("Removed %s logs by launch ids", deleted_logs)
        LOGGER.info("Finished removing launches. It took %.2f sec", time() - t_start)
        return deleted_logs

    @utils.ignore_warnings
    def delete_by_query(
        self, index_name: str, ids_for_removal: list, delete_query_deriver: Callable[[list], dict[str, Any]]
    ) -> int:
        if not self.index_exists(index_name):
            return 0
        batch_size = 1000
        deleted_logs = 0
        for i in range(int(len(ids_for_removal) / batch_size) + 1):
            sub_ids_for_removal = ids_for_removal[i * batch_size : (i + 1) * batch_size]
            if not sub_ids_for_removal:
                continue
            result = self.es_client.delete_by_query(index=index_name, body=delete_query_deriver(sub_ids_for_removal))
            if "deleted" in result:
                deleted_logs += result["deleted"]
        return deleted_logs

    def __time_range_query(
        self,
        time_field: str,
        gte_time: str,
        lte_time: str,
        for_scan: bool = False,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"query": {"range": {time_field: {"gte": gte_time, "lte": lte_time}}}}
        if for_scan:
            query["size"] = self.app_config.esChunkNumber
        return query

    def get_launch_ids_by_start_time_range(self, project: int, start_date: str, end_date: str) -> list[str]:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("launch_start_time", start_date, end_date, for_scan=True)
        launch_ids = set()
        for log in opensearchpy.helpers.scan(self.es_client, query=query, index=index_name):
            launch_ids.add(log["_source"]["launch_id"])
        return list(launch_ids)

    def remove_by_launch_start_time_range(self, project: int, start_date: str, end_date: str) -> int:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("launch_start_time", start_date, end_date)
        delete_response = self.es_client.delete_by_query(index=index_name, body=query)
        return delete_response["deleted"]

    def get_log_ids_by_log_time_range(self, project: int, start_date: str, end_date: str) -> list[str]:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("log_time", start_date, end_date, for_scan=True)
        log_ids = set()
        for log in opensearchpy.helpers.scan(self.es_client, query=query, index=index_name):
            log_ids.add(log["_id"])
        return list(log_ids)

    def remove_by_log_time_range(self, project: int, start_date: str, end_date: str) -> int:
        index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
        query = self.__time_range_query("log_time", start_date, end_date)
        delete_response = self.es_client.delete_by_query(index=index_name, body=query)
        return delete_response["deleted"]
