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

"""OpenSearch client for Test Item-centric indexing approach."""

import traceback
from datetime import datetime, timezone
from time import time
from typing import Any, Callable, Generic, Optional, TypeVar

import opensearchpy.helpers
import urllib3
from opensearchpy import OpenSearch, RequestsHttpConnection
from pydantic import BaseModel, Field
from urllib3.exceptions import InsecureRequestWarning

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, BulkResponse, Response
from app.commons.model.test_item_index import LogData, TestItemIndexData, TestItemUpdateData
from app.utils import text_processing, utils

INDEX_SETTINGS_FILE = "index_settings.json"
TEST_ITEM_INDEX_MAPPINGS_FILE = "test_item_index_mappings.json"

LOGGER = logging.getLogger("analyzerApp.osClient")

T = TypeVar("T")


class Hit(BaseModel, Generic[T]):
    """Typed representation of an OpenSearch search hit."""

    index: str = Field(validation_alias="_index")
    id: str = Field(validation_alias="_id")
    score: Optional[float] = Field(default=None, validation_alias="_score")
    source: T = Field(validation_alias="_source")
    sort: Optional[list[Any]] = None
    highlight: Optional[dict[str, Any]] = None
    fields: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, hit: dict[str, Any]) -> "Hit[T]":
        """
        Build a typed Hit object from OpenSearch raw hit using pydantic validation.

        :param hit: Raw hit dictionary returned by OpenSearch
        :return: Hit instance with mapped source
        """
        return cls.model_validate(hit)


def get_test_item_index_name(project_id: str | int, prefix: str) -> str:
    """
    Build the Test Item index name for a given project.

    :param project_id: The project identifier
    :param prefix: Index name prefix from configuration
    :return: Full index name
    """
    return f"{prefix}{project_id}"


class OsClient:
    """OpenSearch client for Test Item-centric document indexing."""

    app_config: ApplicationConfig
    os_client: OpenSearch
    host: str
    _checked_indexes: set[str]

    def __init__(
        self,
        app_config: ApplicationConfig,
        *,
        os_client: Optional[OpenSearch] = None,
    ) -> None:
        """Initialize the OpenSearch client.

        :param app_config: Application configuration
        :param os_client: Optional pre-configured OpenSearch client for testing
        """
        self.app_config = app_config
        self.host = app_config.esHost
        if os_client:
            LOGGER.debug("Creating service using provided client")
            self.os_client = os_client
        else:
            LOGGER.debug(f"Creating service using host URL: {text_processing.remove_credentials_from_url(self.host)}")
            self.os_client = self._create_os_client(app_config)
        self._checked_indexes = set()

    def _create_os_client(self, app_config: ApplicationConfig) -> OpenSearch:
        """Create an OpenSearch client with the given configuration.

        :param app_config: Application configuration
        :return: Configured OpenSearch client
        """
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

    def _get_base_url(self) -> Optional[str]:
        """Get base URL for OpenSearch.

        :return: Base URL or None if host is not set
        """
        if not self.host:
            LOGGER.error("OpenSearch host is not set")
            return None
        url = self.host
        if not self.host.startswith("http"):
            protocol = "https" if self.app_config.esUseSsl else "http"
            url = f"{protocol}://{self.host}"
        return url

    def is_healthy(self) -> bool:
        """Check whether OpenSearch is healthy.

        :return: True if cluster status is green or yellow
        """
        base_url = self._get_base_url()
        if not base_url:
            return False

        url = text_processing.build_url(base_url, ["_cluster/health"])
        res = utils.send_request(url, "GET", self.app_config.esUser, self.app_config.esPassword)
        return res.get("status", "") in {"green", "yellow"} if res else False

    def delete_index(self, project_id: str | int) -> bool:
        """Delete the Test Item index for a project.

        :param project_id: The project identifier
        :return: True if deletion was successful
        """
        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        LOGGER.info(f"Deleting index: {index_name}")
        try:
            self.os_client.indices.delete(index=index_name)
            self._checked_indexes.discard(index_name)
            LOGGER.debug(f"Index '{index_name}' deleted")
            return True
        except Exception as err:
            LOGGER.exception(f"Failed to delete index: {index_name}", exc_info=err)
            return False

    def _create_index(self, index_name: str) -> Response:
        """
        Create a Test Item index in OpenSearch.

        :param index_name: The index name to create
        :return: Response with acknowledgment status
        """
        LOGGER.info(f"Creating Test Item index: {index_name}")
        response = self.os_client.indices.create(
            index=index_name,
            body={
                "settings": utils.read_resource_file(INDEX_SETTINGS_FILE, to_json=True),
                "mappings": utils.read_resource_file(TEST_ITEM_INDEX_MAPPINGS_FILE, to_json=True),
            },
        )
        LOGGER.debug(f"Index '{index_name}' created")
        return Response(**response)

    def _index_exists(self, index_name: str, print_error: bool = True) -> bool:
        """Check whether an index exists.

        :param index_name: The index name to check
        :param print_error: Whether to log errors
        :return: True if index exists
        """
        try:
            index = self.os_client.indices.get(index=index_name)
            return index is not None
        except Exception as err:
            if print_error:
                LOGGER.exception(f"Index '{index_name}' was not found", exc_info=err)
            return False

    def _ensure_index_exists(self, index_name: str) -> bool:
        """Ensure the Test Item index exists, creating it if necessary.

        This method caches checked indexes to avoid repeated existence checks.
        The cache is cleared on application restart.

        :param index_name: The index name to check/create
        :return: True if index exists or was created successfully
        """
        if index_name in self._checked_indexes:
            return True

        try:
            if self._index_exists(index_name, print_error=False):
                self._checked_indexes.add(index_name)
                return True

            response = self._create_index(index_name)
            if response.acknowledged:
                self._checked_indexes.add(index_name)
                return True
            return False
        except Exception as err:
            LOGGER.exception(f"Failed to ensure index exists: {index_name}", exc_info=err)
            return False

    def _update_settings_after_read_only(self) -> None:
        """Reset read-only settings on all indices after disk space recovery."""
        base_url = self._get_base_url()
        if not base_url:
            return

        utils.send_request(
            f"{base_url}/_all/_settings",
            "PUT",
            self.app_config.esUser,
            self.app_config.esPassword,
            data='{"index.blocks.read_only_allow_delete": null}',
        )

    def _execute_bulk(
        self,
        bodies: list[dict[str, Any]],
        refresh: bool = True,
        chunk_size: Optional[int] = None,
    ) -> BulkResponse:
        """Execute bulk indexing operation.

        :param bodies: List of bulk operation bodies
        :param refresh: Whether to refresh the index after indexing
        :param chunk_size: Optional chunk size for bulk operations
        :return: BulkResponse with indexing results
        """
        if not bodies:
            return BulkResponse(took=0, errors=False)

        start_time = time()
        LOGGER.debug(f"Indexing {len(bodies)} documents")
        es_chunk_number = self.app_config.esChunkNumber
        if chunk_size is not None:
            es_chunk_number = chunk_size

        try:
            try:
                success_count, errors = opensearchpy.helpers.bulk(
                    self.os_client,
                    bodies,
                    chunk_size=es_chunk_number,
                    request_timeout=30,
                    refresh=refresh,
                )
            except Exception:
                formatted_exception = traceback.format_exc()
                LOGGER.warning(f"Bulk indexing failed, retrying: {formatted_exception}")
                self._update_settings_after_read_only()
                success_count, errors = opensearchpy.helpers.bulk(
                    self.os_client,
                    bodies,
                    chunk_size=es_chunk_number,
                    request_timeout=30,
                    refresh=refresh,
                )

            error_str = ""
            if errors:
                error_str = ", ".join([str(error) for error in errors])
            LOGGER.debug(
                f"{success_count} documents were successfully indexed"
                f"{'. Errors: ' + error_str if error_str else ''}"
            )
            LOGGER.debug("Finished indexing for %.2f s", time() - start_time)
            return BulkResponse(took=success_count, errors=len(errors) > 0)
        except Exception as exc:
            LOGGER.exception("Error in bulk indexing", exc_info=exc)
            return BulkResponse(took=0, errors=True)

    def bulk_index_raw(
        self,
        bodies: list[dict[str, Any]],
        refresh: bool = True,
        chunk_size: Optional[int] = None,
    ) -> BulkResponse:
        """Bulk index raw document bodies.

        Useful for update and delete operations.

        :param bodies: List of bulk operation bodies
        :param refresh: Whether to refresh the index after indexing
        :param chunk_size: Optional chunk size for bulk operations
        :return: BulkResponse with indexing results
        """
        if not bodies:
            return BulkResponse(took=0, errors=False)

        index_names = set(body["_index"] for body in bodies)

        for index_name in index_names:
            if not self._ensure_index_exists(index_name):
                LOGGER.error(f"Failed to ensure index exists: {index_name}")
                return BulkResponse(took=0, errors=True)

        return self._execute_bulk(bodies, refresh, chunk_size)

    def _prepare_bulk_bodies(self, index_name: str, test_items: list[TestItemIndexData]) -> list[dict[str, Any]]:
        """Prepare bulk index bodies from TestItemIndexData objects.

        :param index_name: Target index name
        :param test_items: List of TestItemIndexData objects
        :return: List of bulk operation bodies
        """
        bodies: list[dict[str, Any]] = []
        for item in test_items:
            bodies.append(
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": item.test_item_id,
                    "_source": item.to_index_dict(),
                }
            )
        return bodies

    def bulk_index(
        self,
        project_id: str | int,
        test_items: list[TestItemIndexData],
        refresh: bool = True,
        chunk_size: Optional[int] = None,
    ) -> BulkResponse:
        """Bulk index Test Item documents.

        Automatically ensures the index exists before indexing.

        :param project_id: The project identifier
        :param test_items: List of TestItemIndexData objects to index
        :param refresh: Whether to refresh the index after indexing
        :param chunk_size: Optional chunk size for bulk operations
        :return: BulkResponse with indexing results
        """
        if not test_items:
            return BulkResponse(took=0, errors=False)

        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        bodies = self._prepare_bulk_bodies(index_name, test_items)

        return self.bulk_index_raw(bodies, refresh, chunk_size)

    def _delete_by_query(
        self,
        index_name: str,
        ids: list[str],
        query_builder: Callable[[list[str]], dict[str, Any]],
        chunk_size: Optional[int] = None,
    ) -> int:
        """Delete documents by query in batches.

        :param index_name: Target index name
        :param ids: List of IDs to include in deletion query
        :param query_builder: Function to build the deletion query from IDs
        :param chunk_size: Optional chunk size for bulk operations
        :return: Total number of deleted documents
        """
        batch_size = self.app_config.esChunkNumber
        if chunk_size is not None:
            batch_size = chunk_size

        deleted = 0
        for i in range(int(len(ids) / batch_size) + 1):
            batch_ids = ids[i * batch_size : (i + 1) * batch_size]
            if not batch_ids:
                continue
            try:
                result = self.os_client.delete_by_query(index=index_name, body=query_builder(batch_ids))
                if "deleted" in result:
                    deleted += result["deleted"]
            except Exception as err:
                LOGGER.exception("Error in delete_by_query", exc_info=err)
        return deleted

    def delete_test_items(self, project_id: str | int, test_item_ids: list[str]) -> int:
        """Delete Test Items by their IDs.

        :param project_id: The project identifier
        :param test_item_ids: List of test item IDs to delete
        :return: Number of deleted documents
        """
        if not test_item_ids:
            return 0

        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        LOGGER.info(f"Deleting {len(test_item_ids)} test items from '{index_name}'")
        t_start = time()

        if not self._index_exists(index_name, print_error=False):
            return 0

        deleted = self._delete_by_query(
            index_name,
            test_item_ids,
            lambda ids: {"query": {"terms": {"test_item_id": ids}}},
        )
        LOGGER.info(f"Deleted {deleted} test items in {round(time() - t_start, 2)} sec")
        return deleted

    def delete_by_launch_ids(self, project_id: str | int, launch_ids: list[str]) -> int:
        """Delete Test Items by launch IDs.

        :param project_id: The project identifier
        :param launch_ids: List of launch IDs whose test items should be deleted
        :return: Number of deleted documents
        """
        if not launch_ids:
            return 0

        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        LOGGER.info(f"Deleting test items for {len(launch_ids)} launches from '{index_name}'")
        t_start = time()

        if not self._index_exists(index_name, print_error=False):
            return 0

        deleted = self._delete_by_query(
            index_name,
            launch_ids,
            lambda ids: {"query": {"terms": {"launch_id": ids}}},
        )
        LOGGER.info(f"Deleted {deleted} test items in {round(time() - t_start, 2)} sec")
        return deleted

    def _time_range_query(
        self, time_field: str, gte_time: str, lte_time: str, for_scan: bool = False
    ) -> dict[str, Any]:
        """Build a time range query.

        :param time_field: The field to query on
        :param gte_time: Greater than or equal time
        :param lte_time: Less than or equal time
        :param for_scan: Whether to include size for scan operations
        :return: Query dictionary
        """
        query: dict[str, Any] = {"query": {"range": {time_field: {"gte": gte_time, "lte": lte_time}}}}
        if for_scan:
            query["size"] = self.app_config.esChunkNumber
        return query

    def delete_by_launch_start_time_range(self, project_id: str | int, start_date: str, end_date: str) -> int:
        """Delete Test Items by launch start time range.

        :param project_id: The project identifier
        :param start_date: Start date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :param end_date: End date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :return: Number of deleted documents
        """
        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        if not self._index_exists(index_name, print_error=False):
            return 0

        query = self._time_range_query("launch_start_time", start_date, end_date)
        try:
            result = self.os_client.delete_by_query(index=index_name, body=query)
            return result.get("deleted", 0)
        except Exception as err:
            LOGGER.exception("Error in delete_by_launch_start_time_range", exc_info=err)
            return 0

    def delete_by_test_item_start_time_range(self, project_id: str | int, start_date: str, end_date: str) -> int:
        """Delete Test Items by test item start time range.

        :param project_id: The project identifier
        :param start_date: Start date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :param end_date: End date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :return: Number of deleted documents
        """
        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        if not self._index_exists(index_name, print_error=False):
            return 0

        query = self._time_range_query("start_time", start_date, end_date)
        try:
            result = self.os_client.delete_by_query(index=index_name, body=query)
            return result.get("deleted", 0)
        except Exception as err:
            LOGGER.exception("Error in delete_by_test_item_start_time_range", exc_info=err)
            return 0

    def search(
        self, project_id: str | int, query: dict[str, Any], scroll: Optional[str] = None
    ) -> list[Hit[TestItemIndexData]]:
        """Execute a search query and return all results.

        :param project_id: The project identifier
        :param query: OpenSearch query
        :param scroll: Optional custom scroll timeout for large result sets
        :return: List of typed search hits
        """
        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        if not self._index_exists(index_name, print_error=False):
            return []

        hits: list[Hit[TestItemIndexData]] = []
        try:
            kwargs: dict[str, Any] = {}
            if scroll:
                kwargs["scroll"] = scroll
            for doc in opensearchpy.helpers.scan(self.os_client, query=query, index=index_name, **kwargs):
                hits.append(Hit[TestItemIndexData].from_dict(doc))
        except Exception as err:
            LOGGER.exception("Error in search", exc_info=err)
        return hits

    def get_launch_ids_by_start_time_range(self, project_id: str | int, start_date: str, end_date: str) -> list[str]:
        """Get launch IDs within a start time range.

        :param project_id: The project identifier
        :param start_date: Start date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :param end_date: End date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :return: List of launch IDs
        """
        query = self._time_range_query("launch_start_time", start_date, end_date, for_scan=True)
        query["_source"] = ["launch_id", "test_item_id"]  # Narrow down output to just two required fields
        launch_ids: set[str] = set()
        for hit in self.search(project_id, query):
            launch_ids.add(hit.source.launch_id)
        return list(launch_ids)

    def get_test_item_ids_by_start_time_range(
        self, project_id: str | int, start_date: str, end_date: str
    ) -> list[str]:
        """Get Test Item IDs within a start time range.

        :param project_id: The project identifier
        :param start_date: Start date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :param end_date: End date in 'yyyy-MM-dd HH:mm:ss' or 'yyyy-MM-dd' format
        :return: List of test item IDs
        """
        query = self._time_range_query("start_time", start_date, end_date, for_scan=True)
        query["_source"] = ["launch_id", "test_item_id"]  # Narrow down output to just two required fields
        test_item_ids: set[str] = set()
        for hit in self.search(project_id, query):
            test_item_ids.add(hit.source.test_item_id)
        return list(test_item_ids)

    def bulk_update_issue_history(self, project_id: str | int, updates: list[TestItemUpdateData]) -> BulkResponse:
        """Bulk update issue_history for multiple Test Items.

        :param project_id: The project identifier
        :param updates: List of TestItemUpdateData payloads
        :return: BulkResponse with update results
        """
        if not updates:
            return BulkResponse(took=0, errors=False)

        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        if not self._index_exists(index_name, print_error=True):
            LOGGER.error(f"Index not exists: {index_name}")
            return BulkResponse(took=0, errors=True)

        bodies: list[dict[str, Any]] = []
        for update in updates:
            entry = update.to_update_dict()
            bodies.append(
                {
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": update.test_item_id,
                    "script": {
                        "source": """
                            if (ctx._source.issue_history == null) {
                                ctx._source.issue_history = [];
                            }
                            ctx._source.issue_history.add(params.entry);
                        """,
                        "params": {"entry": entry},
                    },
                }
            )

        return self._execute_bulk(bodies)

    def get_test_item(self, project_id: str | int, test_item_id: str) -> Optional[TestItemIndexData]:
        """Get a single Test Item by ID.

        :param project_id: The project identifier
        :param test_item_id: The test item ID
        :return: Test Item object or None if not found
        """
        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        if not self._index_exists(index_name, print_error=False):
            return None

        try:
            response = self.os_client.get(index=index_name, id=test_item_id)
            source = response.get("_source")
            if source is None:
                return None
            return TestItemIndexData.from_dict(source)
        except opensearchpy.exceptions.NotFoundError:
            LOGGER.debug(f"Test item {test_item_id} not found")
            return None
        except Exception as err:
            LOGGER.exception(f"Error getting test item {test_item_id}", exc_info=err)
            return None

    def get_test_items_by_ids(self, project_id: str | int, test_item_ids: list[str]) -> list[TestItemIndexData]:
        """Get multiple Test Items by their IDs.

        :param project_id: The project identifier
        :param test_item_ids: List of test item IDs
        :return: List of Test Item objects
        """
        if not test_item_ids:
            return []

        query = {
            "size": self.app_config.esChunkNumber,
            "query": {"terms": {"test_item_id": test_item_ids}},
        }

        results: list[TestItemIndexData] = []
        for hit in self.search(project_id, query):
            results.append(hit.source)
        return results

    def _remove_logs_with_predicate(
        self,
        project_id: str | int,
        query: dict[str, Any],
        predicate: Callable[[LogData], bool],
    ) -> int:
        """
        Remove logs matching predicate from Test Items and delete empty Test Items.

        :param project_id: Project identifier
        :param query: Query to fetch candidate Test Items
        :param predicate: Predicate to decide which logs to remove
        :return: Number of removed logs
        """
        my_query = {**query, "_source": ["logs", "test_item_id", "log_count"], "size": self.app_config.esChunkNumber}
        hits = self.search(project_id, my_query)
        if not hits:
            return 0

        index_name = get_test_item_index_name(project_id, self.app_config.esProjectIndexPrefix)
        bodies: list[dict[str, Any]] = []
        removed_logs = 0

        for hit in hits:
            source = hit.source
            logs = list(source.logs or [])
            remaining_logs: list[LogData] = []
            removed_from_item = 0

            for log in logs:
                if predicate(log):
                    removed_from_item += 1
                else:
                    remaining_logs.append(log)

            if removed_from_item == 0:
                continue

            removed_logs += removed_from_item
            if not remaining_logs:
                bodies.append({"_op_type": "delete", "_index": index_name, "_id": source.test_item_id})
            else:
                bodies.append(
                    {
                        "_op_type": "update",
                        "_index": index_name,
                        "_id": source.test_item_id,
                        "doc": {
                            "logs": [log.model_dump() for log in remaining_logs],
                            "log_count": len(remaining_logs),
                        },
                    }
                )

        if not bodies:
            return removed_logs

        bulk_response = self._execute_bulk(bodies)
        if bulk_response.errors:
            LOGGER.error("Errors encountered while removing logs for project %s", project_id)
        return removed_logs

    def delete_logs_by_ids(self, project_id: str | int, log_ids: list[str]) -> int:
        """
        Delete logs by their identifiers and remove Test Items that become empty.

        :param project_id: Project identifier
        :param log_ids: List of log identifiers to delete
        :return: Number of deleted logs
        """
        if not log_ids:
            return 0

        log_id_set = {str(log_id) for log_id in log_ids}
        query = {
            "query": {
                "nested": {
                    "path": "logs",
                    "query": {"terms": {"logs.log_id": list(log_id_set)}},
                }
            }
        }

        return self._remove_logs_with_predicate(project_id, query, lambda log: log.log_id in log_id_set)

    @staticmethod
    def _normalize_datetime(value: str) -> Optional[datetime]:
        """Parse and normalize datetime strings to naive UTC for comparisons."""
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None

        if parsed is None:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue

        if parsed and parsed.tzinfo:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def delete_by_log_time_range(self, project_id: str | int, start_date: str, end_date: str) -> int:
        """
        Delete logs by log_time range and remove Test Items that become empty.

        :param project_id: Project identifier
        :param start_date: Start of the log_time interval
        :param end_date: End of the log_time interval
        :return: Number of deleted logs
        """
        start_dt = self._normalize_datetime(start_date)
        end_dt = self._normalize_datetime(end_date)

        def is_in_range(log: LogData) -> bool:
            log_dt = self._normalize_datetime(log.log_time)
            if start_dt and end_dt and log_dt:
                return start_dt <= log_dt <= end_dt
            return start_date <= log.log_time <= end_date

        query = {
            "query": {
                "nested": {
                    "path": "logs",
                    "query": {"range": {"logs.log_time": {"gte": start_date, "lte": end_date}}},
                }
            }
        }

        return self._remove_logs_with_predicate(project_id, query, is_in_range)
