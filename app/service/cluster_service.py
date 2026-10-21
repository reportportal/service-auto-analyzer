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

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from app.commons import clustering, log_merger, logging, request_factory
from app.commons.model import TestItemIndexData
from app.commons.model.db import Hit
from app.commons.model.launch_objects import (
    ApplicationConfig,
    ClusterInfo,
    ClusterResult,
    LaunchInfoForClustering,
    SearchConfig,
)
from app.commons.model.test_item_index import (
    ClusterLogData,
    LogClusterData,
    LogData,
    MergedLogData,
    MergedLogsClusterData,
)
from app.commons.os_client import OsClient
from app.commons.prepared_log import PreparedLogMessageClustering
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.clusterService")

LOGS_PATH = "logs"
MERGED_LOGS_PATH = "merged_small_logs"
CLUSTER_LOG_FIELDS = [
    "whole_message",
    "cluster_id",
    "cluster_message",
    "cluster_with_numbers",
    "found_exceptions",
    "potential_status_codes",
    "log_level",
]


@dataclass
class Log:
    test_item_id: str
    message: str
    data: ClusterLogData
    log_ids: list[str]
    launch_id: Optional[str] = None
    merged: bool = False
    # Source of the cluster message, numbers are kept here and removed later according to `cleanNumbers`
    whole_message: str = ""

    @classmethod
    def create(
        cls,
        test_item_id: str,
        message: str,
        data: ClusterLogData,
        launch_id: Optional[str] = None,
        whole_message: Optional[str] = None,
    ) -> "Log":
        """Create a log entry for clustering from a single or a merged log entry.

        :param test_item_id: Identifier of the Test Item the log entry belongs to
        :param message: Message prepared for clustering
        :param data: Single (`LogData`) or merged (`MergedLogData`) log entry
        :param launch_id: Identifier of the Launch the log entry belongs to
        :param whole_message: Source of the cluster message, `data.whole_message` is used if not specified
        :return: Log entry for clustering
        """
        if whole_message is None:
            whole_message = data.whole_message or ""
        if isinstance(data, MergedLogData):
            return cls(
                test_item_id,
                message,
                data,
                list(data.log_ids),
                launch_id=launch_id,
                merged=True,
                whole_message=whole_message,
            )
        if isinstance(data, LogData):
            return cls(test_item_id, message, data, [data.log_id], launch_id=launch_id, whole_message=whole_message)
        raise ValueError(f"Unsupported log data type: {type(data).__name__}")


def get_number_of_log_lines(log: Log, launch_info: LaunchInfoForClustering) -> int:
    """Get the number of log lines to use in clustering: all lines for merged log entries."""
    return -1 if log.merged else launch_info.numberOfLogLines


def find_non_zero_id_non_empty_message(group: list[int], logs: dict[int, Log]) -> tuple[Optional[int], Optional[str]]:
    cluster_id: Optional[int] = None
    cluster_message: Optional[str] = None
    for ind in group:
        if ind == 0:  # not to use old cluster id and message
            continue
        current_cluster_id = logs[ind].data.cluster_id.strip()
        if current_cluster_id and int(current_cluster_id) != 0:
            cluster_id = int(current_cluster_id)
            current_cluster_message = logs[ind].data.cluster_message.strip()
            if current_cluster_message:
                cluster_message = current_cluster_message
    return cluster_id, cluster_message


def collect_log_and_item_ids(
    cluster: list[int], logs: dict[int, Log], launch_info: LaunchInfoForClustering
) -> tuple[list[int], list[int]]:
    new_group_log_ids: list[int] = []
    new_group_test_items: list[int] = []
    for ind in cluster:
        if ind == 0:
            continue
        if logs[ind].launch_id != str(launch_info.launch.launchId):
            continue
        # Item IDs are repeated for each log ID to keep both lists aligned
        for log_id in logs[ind].log_ids:
            new_group_log_ids.append(int(log_id))
            new_group_test_items.append(int(logs[ind].test_item_id))
    return new_group_log_ids, new_group_test_items


def regroup_logs_by_error_and_status_codes(logs: list[Log]) -> list[list[int]]:
    regrouped_by_error = defaultdict(list)
    for i, log in enumerate(logs):
        found_exceptions_raw: str = log.data.found_exceptions or ""
        found_exceptions = " ".join(sorted(found_exceptions_raw.split()))
        # Status codes are not sorted: they are stored in the order they appeared, and
        # "expected 400, but was 401" is a different failure from its inverse.
        potential_status_codes = " ".join((log.data.potential_status_codes or "").split())
        group_key = (found_exceptions, potential_status_codes)
        regrouped_by_error[group_key].append(i)
    return list(regrouped_by_error.values())


def cluster_messages_with_grouping_by_error(logs: list[Log], min_match_threshold: float) -> dict[int, list[int]]:
    regrouped_by_error = regroup_logs_by_error_and_status_codes(logs)
    all_clusters: dict[int, list[int]] = defaultdict(list)
    start_cluster_id = 0
    for group in regrouped_by_error:
        similar_messages = []
        similar_messages_indexes = {}
        for i, idx in enumerate(group):
            similar_messages.append(logs[idx].message)
            similar_messages_indexes[i] = idx
        clusters = clustering.find_clusters(similar_messages, threshold=min_match_threshold)
        max_group_id = max(clusters.keys())
        for cluster_id, cluster in clusters.items():
            global_idx = start_cluster_id + cluster_id
            for i in cluster:
                all_clusters[global_idx].append(similar_messages_indexes[i])
        start_cluster_id = start_cluster_id + max_group_id + 1
    return dict(all_clusters)


def calculate_hash(
    group_ids: list[int],
    logs: list[Log],
    launch_info: LaunchInfoForClustering,
) -> tuple[int, str]:
    group_logs = []
    log_message = ""
    for i in range(min(100, len(group_ids))):
        ind = group_ids[i]
        group_logs.append(logs[ind].message)
        if not log_message:
            log_message = (
                text_processing.prepare_message_for_clustering(
                    logs[ind].whole_message,
                    get_number_of_log_lines(logs[ind], launch_info),
                    launch_info.cleanNumbers,
                    leave_log_structure=True,
                )
                or ""
            ).strip()
    _cnt_vectorizer = CountVectorizer(binary=True, analyzer="word", token_pattern="[^ ]+", ngram_range=(2, 2))
    group_res = _cnt_vectorizer.fit_transform(group_logs).astype(np.int8)
    res_bitwise = np.bitwise_and.reduce(group_res.toarray(), axis=0)
    bigrams_list = []
    for i, feature_name in enumerate(_cnt_vectorizer.get_feature_names_out()):
        if res_bitwise[i] == 1:
            bigrams_list.append(feature_name)
    hash_message = int(hashlib.sha1(" ".join(bigrams_list).encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % (
        10**16
    )
    hash_message = hash_message * 10 + int(not launch_info.cleanNumbers)
    return hash_message, log_message


def create_new_cluster(
    similar_logs: dict[int, Log], launch_info: LaunchInfoForClustering, min_should_match: float
) -> Optional[ClusterInfo]:
    similar_log_messages = [similar_logs[idx].message for idx in sorted(similar_logs.keys())]
    local_clusters = clustering.find_clusters(similar_log_messages, threshold=min_should_match)
    for local_cluster in local_clusters.values():
        if 0 not in local_cluster:
            # We only need a cluster with idx==0 (with original message)
            continue
        cluster_id, cluster_message = find_non_zero_id_non_empty_message(local_cluster, similar_logs)
        if not cluster_id or not cluster_message:
            continue

        new_group_log_ids, new_group_test_items = collect_log_and_item_ids(local_cluster, similar_logs, launch_info)
        new_group = ClusterInfo(
            logIds=new_group_log_ids,
            itemIds=new_group_test_items,
            clusterMessage=cluster_message,
            clusterId=cluster_id,
        )
        return new_group
    return None


def _prepare_clustering_message(message: Optional[str], number_of_log_lines: int, clean_numbers: bool) -> str:
    return (text_processing.prepare_message_for_clustering(message, number_of_log_lines, clean_numbers) or "").strip()


def _prepare_whole_message_for_clustering(original_message: Optional[str]) -> str:
    """Prepare combined exception message and stacktrace, which is used as a source of the cluster message.

    Unlike the indexed `whole_message` field, numbers are kept in the exception message, so they can be removed or
    left in the cluster message according to the `cleanNumbers` option of the request.

    :param original_message: Raw log message
    :return: Combined exception message and stacktrace
    """
    if not original_message:
        return ""
    prepared_log = PreparedLogMessageClustering(original_message, -1)
    return "\n".join([prepared_log.exception_message_no_params, prepared_log.stacktrace])


def generate_clustering_messages(
    launch_info: LaunchInfoForClustering, prepared_items: list[TestItemIndexData]
) -> list[Log]:
    """Generate messages which will be used for clustering, save their metadata.

    Small log entries of one log level of a Test Item are merged into one entry, see `log_merger.merge_small_logs`.
    """
    logs: list[Log] = []
    for item in prepared_items:
        big_logs, small_log_groups = log_merger.merge_small_logs(item.get_sorted_logs())
        for log in big_logs:
            log_message = _prepare_clustering_message(
                log.message_for_clustering, launch_info.numberOfLogLines, launch_info.cleanNumbers
            )
            if log_message:
                whole_message = _prepare_whole_message_for_clustering(log.original_message)
                logs.append(Log.create(item.test_item_id, log_message, log, whole_message=whole_message))
        for small_logs in small_log_groups:
            small_logs = [
                log
                for log in small_logs
                if _prepare_clustering_message(log.message_for_clustering, -1, launch_info.cleanNumbers)
            ]
            if not small_logs:
                continue
            log_message = _prepare_clustering_message(
                "\n".join(log.message_for_clustering or "" for log in small_logs), -1, launch_info.cleanNumbers
            )
            whole_message = "\n".join(
                _prepare_whole_message_for_clustering(log.original_message).strip() for log in small_logs
            )
            logs.append(
                Log.create(
                    item.test_item_id, log_message, MergedLogData.from_logs(small_logs), whole_message=whole_message
                )
            )
    return logs


def compile_results(
    cluster_message_by_id: dict[Any, Any], clusters_found: dict[int, tuple[list[int], list[int]]]
) -> list[Any]:
    results_to_return = []
    for cluster_id in clusters_found:
        results_to_return.append(
            ClusterInfo(
                clusterId=cluster_id,
                clusterMessage=cluster_message_by_id[cluster_id],
                logIds=clusters_found[cluster_id][0],
                itemIds=list(set(clusters_found[cluster_id][1])),
            )
        )
    return results_to_return


def gather_cluster_results(
    initial_clusters: dict[int, list[int]],
    additional_clusters: dict[int, ClusterInfo],
    logs: list[Log],
    launch_info: LaunchInfoForClustering,
) -> tuple[list[ClusterInfo], list[LogClusterData], list[MergedLogsClusterData]]:
    updates: list[LogClusterData] = []
    merged_logs_by_item: dict[str, list[MergedLogData]] = defaultdict(list)
    clusters_found: dict[int, tuple[list[int], list[int]]] = {}
    cluster_message_by_id = {}
    for cluster_idx, cluster in initial_clusters.items():
        cluster_id = 0
        cluster_message = ""
        if cluster_idx in additional_clusters:
            cluster_id = additional_clusters[cluster_idx].clusterId
            cluster_message = additional_clusters[cluster_idx].clusterMessage
        if not cluster_id or not cluster_message:
            cluster_id, cluster_message = calculate_hash(cluster, logs, launch_info)
        log_ids: list[int] = []
        test_item_ids: list[int] = []
        for ind in cluster:
            log = logs[ind]
            for log_id in log.log_ids:
                log_ids.append(int(log_id))
                test_item_ids.append(int(log.test_item_id))
                updates.append(
                    LogClusterData(
                        log_id=log_id,
                        test_item_id=log.test_item_id,
                        cluster_id=str(cluster_id),
                        cluster_message=cluster_message,
                        cluster_with_numbers=not launch_info.cleanNumbers,
                    )
                )
            if isinstance(log.data, MergedLogData):
                merged_logs_by_item[log.test_item_id].append(
                    log.data.model_copy(
                        update={
                            "cluster_id": str(cluster_id),
                            "cluster_message": cluster_message,
                            "cluster_with_numbers": not launch_info.cleanNumbers,
                        }
                    )
                )
        if cluster_idx in additional_clusters:
            for additional_log_id, additional_item_id in zip(
                additional_clusters[cluster_idx].logIds, additional_clusters[cluster_idx].itemIds
            ):
                log_ids.append(additional_log_id)
                test_item_ids.append(additional_item_id)
                updates.append(
                    LogClusterData(
                        log_id=str(additional_log_id),
                        test_item_id=str(additional_item_id),
                        cluster_id=str(cluster_id),
                        cluster_message=cluster_message,
                        cluster_with_numbers=not launch_info.cleanNumbers,
                    )
                )
        if cluster_id not in clusters_found:
            clusters_found[cluster_id] = (log_ids, test_item_ids)
        else:
            cluster_log_ids, cluster_test_items = clusters_found[cluster_id]
            cluster_log_ids.extend(log_ids)
            cluster_test_items.extend(test_item_ids)
            clusters_found[cluster_id] = (cluster_log_ids, cluster_test_items)
        cluster_message_by_id[cluster_id] = cluster_message
    results_to_return = compile_results(cluster_message_by_id, clusters_found)
    merged_updates = [
        MergedLogsClusterData(test_item_id=test_item_id, merged_small_logs=merged_logs)
        for test_item_id, merged_logs in merged_logs_by_item.items()
    ]
    return results_to_return, updates, merged_updates


class ClusterService:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient

    def __init__(
        self, app_config: ApplicationConfig, search_cfg: SearchConfig, *, os_client: Optional[OsClient] = None
    ) -> None:
        """Initialize ClusterService

        :param app_config: Application configuration object
        :param search_cfg: Search configuration object
        :param os_client: Optional OsClient instance. If not provided, a new one will be created.
        """
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.os_client = os_client or OsClient(app_config=self.app_config)

    def _get_query_with_start_time_decay(self, main_query: dict[str, Any]) -> dict[str, Any]:
        return {
            "_source": main_query["_source"],
            "size": main_query["size"],
            "query": {
                "function_score": {
                    "query": main_query["query"],
                    "functions": [
                        {
                            "exp": {
                                "start_time": {
                                    "origin": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "scale": "7d",
                                    "offset": "1d",
                                    "decay": self.search_cfg.TimeWeightDecay,
                                }
                            }
                        },
                        {"script_score": {"script": {"source": "0.2"}}},
                    ],
                    "score_mode": "max",
                    "boost_mode": "multiply",
                }
            },
        }

    def _build_search_similar_items_query(
        self,
        queried_log: Log,
        launch_info: LaunchInfoForClustering,
        min_should_match: str = "95%",
    ) -> dict[str, Any]:
        """Build search query.

        Merged log entries are searched among merged log entries only, and single log entries among single ones.
        """
        path = MERGED_LOGS_PATH if queried_log.merged else LOGS_PATH
        log_id_field = "log_ids" if queried_log.merged else "log_id"
        nested_must = [
            utils.build_more_like_this_query(
                min_should_match,
                queried_log.message,
                field_name=f"{path}.whole_message",
                boost=utils.BOOST_NEUTRAL,
                max_query_terms=self.search_cfg.MaxQueryTerms,
            )
        ]
        found_exceptions = queried_log.data.found_exceptions
        if found_exceptions and found_exceptions.strip():
            nested_must.append(
                utils.build_more_like_this_query(
                    "1",
                    found_exceptions,
                    field_name=f"{path}.found_exceptions",
                    boost=utils.BOOST_NEUTRAL,
                    override_min_should_match="1",
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
            )
        # Clustering requires the same codes in the same order, so the exact sequence is matched.
        status_codes_query = utils.build_status_codes_exact_query(
            queried_log.data.potential_status_codes or "", field_name=f"{path}.potential_status_codes"
        )
        if status_codes_query:
            nested_must.append(status_codes_query)

        nested_query = {
            "nested": {
                "path": path,
                "score_mode": "max",
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    f"{path}.log_level": {"gte": launch_info.launch.analyzerConfig.minimumLogLevel}
                                }
                            },
                            {"term": {f"{path}.cluster_with_numbers": not launch_info.cleanNumbers}},
                        ],
                        "must": [{"wildcard": {f"{path}.cluster_message": "*"}}, *nested_must],
                    }
                },
                "inner_hits": {
                    "size": 5,
                    "_source": [f"{path}.{field}" for field in [log_id_field, *CLUSTER_LOG_FIELDS]],
                },
            }
        }

        query: dict[str, Any] = {
            "_source": ["launch_id", "launch_name", "test_item_id"],
            "size": 10,
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": "issue_type"}}],
                    "must_not": [{"term": {"test_item_id": queried_log.test_item_id}}],
                    "should": [],
                    "must": [nested_query],
                }
            },
        }
        if launch_info.forUpdate:
            query["query"]["bool"]["should"].append({"term": {"launch_id": launch_info.launch.launchId}})
        else:
            query["query"]["bool"]["must_not"].append({"term": {"launch_id": launch_info.launch.launchId}})
        query["query"]["bool"]["should"].append({"term": {"launch_name": launch_info.launch.launchName}})
        return self._get_query_with_start_time_decay(query)

    def _find_similar_logs(
        self, log: Log, launch_info: LaunchInfoForClustering, min_should_match: float
    ) -> dict[int, Log]:
        log_dict_part: dict[int, Log] = {0: log}
        query = self._build_search_similar_items_query(
            log,
            launch_info,
            min_should_match=text_processing.prepare_es_min_should_match(min_should_match),
        )
        path = MERGED_LOGS_PATH if log.merged else LOGS_PATH
        number_of_log_lines = get_number_of_log_lines(log, launch_info)
        ind = 0
        for hit in self.os_client.search(launch_info.project, query):
            inner_hits = hit.inner_hits
            if not inner_hits:
                continue
            inner_hits_logs = inner_hits.get(path, {})
            if not inner_hits_logs:
                continue
            for raw_inner_hit in inner_hits_logs.get("hits", {}).get("hits", []):
                inner_log_data: ClusterLogData
                if log.merged:
                    inner_log_data = Hit[MergedLogData].from_dict(raw_inner_hit).source
                else:
                    inner_log_data = Hit[LogData].from_dict(raw_inner_hit).source
                log_message = text_processing.prepare_message_for_clustering(
                    inner_log_data.whole_message, number_of_log_lines, launch_info.cleanNumbers
                )
                if not log_message or not log_message.strip():
                    continue
                inner_log = Log.create(
                    hit.source.test_item_id, log_message, inner_log_data, launch_id=hit.source.launch_id
                )
                cluster_message_original = inner_log.data.cluster_message.strip()
                cluster_message_processed = text_processing.prepare_message_for_clustering(
                    cluster_message_original,
                    number_of_log_lines,
                    launch_info.cleanNumbers,
                    leave_log_structure=True,
                )
                if cluster_message_original and cluster_message_processed != cluster_message_original:
                    continue

                equal = True
                # `found_exceptions` is compared as a set, `potential_status_codes` in order.
                for column, keep_order in (("found_exceptions", False), ("potential_status_codes", True)):
                    candidate_parts = (getattr(inner_log.data, column, None) or "").split()
                    compare_parts = (getattr(log.data, column, None) or "").split()
                    if not keep_order:
                        candidate_parts, compare_parts = sorted(candidate_parts), sorted(compare_parts)
                    if candidate_parts != compare_parts:
                        equal = False
                        break
                if not equal:
                    continue
                ind += 1
                log_dict_part[ind] = inner_log
        return log_dict_part

    def _find_similar_clusters(
        self,
        groups: dict[int, list[int]],
        logs: list[Log],
        launch_info: LaunchInfoForClustering,
        min_match_threshold: float,
    ) -> dict[int, ClusterInfo]:
        additional_clusters: dict[int, ClusterInfo] = {}
        for global_group_idx, global_group in groups.items():
            first_item_ind = global_group[0]
            log = logs[first_item_ind]
            min_should_match = utils.calculate_threshold_for_text(log.message, min_match_threshold)

            similar_logs = self._find_similar_logs(log, launch_info, min_should_match)

            new_cluster = create_new_cluster(similar_logs, launch_info, min_should_match)
            if new_cluster:
                LOGGER.debug(
                    f"Found cluster Id: '{new_cluster.clusterId}'; Log IDs: {new_cluster.logIds}; Cluster message: "
                    + new_cluster.clusterMessage,
                )
                additional_clusters[global_group_idx] = new_cluster
        return additional_clusters

    def find_clusters(self, launch_info: LaunchInfoForClustering) -> ClusterResult:
        LOGGER.info(
            f"Started clusterizing logs for launch {launch_info.launch.launchId} in project {launch_info.project}"
        )
        t_start = time()
        errors_found = []
        errors_count = 0
        common_clusters: list[ClusterInfo] = []
        unique_log_ids: set[str] = set()
        try:
            min_match_threshold = launch_info.launch.analyzerConfig.uniqueErrorsMinShouldMatch / 100.0
            config = launch_info.launch.analyzerConfig
            # Convert Test Items and their logs, which we are going to cluster, to the common storage model
            prepared_items = request_factory.prepare_test_items(
                launch_info.launch,
                number_of_logs_to_index=config.numberOfLogsToIndex,
                minimal_log_level=config.minimumLogLevel,
                similarity_threshold_to_drop=config.similarityThresholdToDrop,
            )

            logs: list[Log] = generate_clustering_messages(launch_info, prepared_items)
            if not logs:
                return ClusterResult(project=launch_info.project, launchId=launch_info.launch.launchId, clusters=[])

            # Cluster messages which were sent by the Backend by error messages and HTTP status codes
            initial_clusters = cluster_messages_with_grouping_by_error(logs, min_match_threshold)
            LOGGER.debug(f"Initial clusters: {json.dumps(initial_clusters)}")

            # Find similar patterns in DB
            additional_clusters = self._find_similar_clusters(initial_clusters, logs, launch_info, min_match_threshold)

            # Collect all unique log IDs for statistics
            for cluster in additional_clusters.values():
                for log_id in cluster.logIds:
                    unique_log_ids.add(str(log_id))
            for log in logs:
                unique_log_ids.update(log.log_ids)

            # Form final clusters (sent + additionally collected)
            common_clusters, db_updates, merged_db_updates = gather_cluster_results(
                initial_clusters, additional_clusters, logs, launch_info
            )

            # Update cluster info in DB if anything found
            if common_clusters:
                for result in common_clusters:
                    LOGGER.debug(
                        f"Cluster Id: {result.clusterId}; Cluster message: '{result.clusterMessage}'; Log IDs: "
                        + str(result.logIds)
                    )
                self.os_client.bulk_update_cluster_info(
                    launch_info.project,
                    db_updates,
                    merged_updates=merged_db_updates,
                    refresh=False,
                    chunk_size=self.app_config.esChunkNumberUpdateClusters,
                )
        except Exception as exc:
            LOGGER.exception(exc)
            errors_found.append(utils.extract_exception(exc))
            errors_count += 1

        results_to_share = {
            launch_info.launch.launchId: {
                "not_found": int(len(common_clusters) == 0),
                "items_to_process": len(unique_log_ids),
                "processed_time": time() - t_start,
                "found_clusters": len(common_clusters),
                "launch_id": launch_info.launch.launchId,
                "launch_name": launch_info.launch.launchName,
                "project_id": launch_info.project,
                "method": "find_clusters",
                "gather_date": datetime.now().strftime("%Y-%m-%d"),
                "gather_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "module_version": [self.app_config.appVersion],
                "model_info": [],
                "errors": errors_found,
                "errors_count": errors_count,
            }
        }

        LOGGER.debug(f"Stats info: {json.dumps(results_to_share)}")
        LOGGER.info("Processed the launch. It took %.2f sec.", time() - t_start)
        LOGGER.info("Finished clustering for the launch with %d clusters.", len(common_clusters))
        for cluster in common_clusters:
            # Put readable text instead of tokens
            cluster.clusterMessage = text_processing.replace_tokens_with_readable_text(cluster.clusterMessage)
        return ClusterResult(
            project=launch_info.project, launchId=launch_info.launch.launchId, clusters=common_clusters
        )
