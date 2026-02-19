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

from app.commons import clustering, logging, request_factory
from app.commons.model import TestItemIndexData
from app.commons.model.db import Hit
from app.commons.model.launch_objects import (
    ApplicationConfig,
    ClusterInfo,
    ClusterResult,
    LaunchInfoForClustering,
    SearchConfig,
)
from app.commons.model.test_item_index import LogClusterData, LogData
from app.commons.os_client import OsClient
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.clusterService")


@dataclass
class Log:
    test_item_id: str
    message: str
    data: LogData
    launch_id: Optional[str] = None


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
        new_group_log_ids.append(int(logs[ind].data.log_id))
        new_group_test_items.append(int(logs[ind].test_item_id))
    return new_group_log_ids, new_group_test_items


def regroup_logs_by_error_and_status_codes(logs: list[Log]) -> list[list[int]]:
    regrouped_by_error = defaultdict(list)
    for i, log in enumerate(logs):
        found_exceptions_raw: str = log.data.found_exceptions or ""
        found_exceptions = " ".join(sorted(found_exceptions_raw.split()))
        potential_status_codes_raw: str = log.data.potential_status_codes or ""
        potential_status_codes = " ".join(sorted(potential_status_codes_raw.split()))
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
                    logs[ind].data.whole_message,
                    launch_info.numberOfLogLines,
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


def generate_clustering_messages(
    launch_info: LaunchInfoForClustering, prepared_items: list[TestItemIndexData]
) -> list[Log]:
    """Generate messages which will be used for clustering, save their metadata."""
    logs: list[Log] = []
    for item in prepared_items:
        item_logs: list[LogData] = item.logs or []
        for log in item_logs:
            log_message = text_processing.prepare_message_for_clustering(
                log.message_for_clustering,
                launch_info.numberOfLogLines,
                launch_info.cleanNumbers,
            )
            if not log_message or not log_message.strip():
                continue
            logs.append(Log(test_item_id=item.test_item_id, message=log_message.strip(), data=log))
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
) -> tuple[list[ClusterInfo], list[LogClusterData]]:
    updates: list[LogClusterData] = []
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
            additional_log_id_str = logs[ind].data.log_id
            log_ids.append(int(additional_log_id_str))
            test_item_ids.append(int(logs[ind].test_item_id))
            updates.append(
                LogClusterData(
                    log_id=additional_log_id_str,
                    test_item_id=logs[ind].test_item_id,
                    cluster_id=str(cluster_id),
                    cluster_message=cluster_message,
                    cluster_with_numbers=not launch_info.cleanNumbers,
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
    return results_to_return, updates


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
        """Build search query"""
        nested_must = [
            utils.build_more_like_this_query(
                min_should_match,
                queried_log.message,
                field_name="logs.whole_message",
                boost=1.0,
                max_query_terms=self.search_cfg.MaxQueryTerms,
            )
        ]
        found_exceptions = queried_log.data.found_exceptions
        if found_exceptions and found_exceptions.strip():
            nested_must.append(
                utils.build_more_like_this_query(
                    "1",
                    found_exceptions,
                    field_name="logs.found_exceptions",
                    boost=1.0,
                    override_min_should_match="1",
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
            )
        potential_status_codes = queried_log.data.potential_status_codes
        if potential_status_codes:
            number_of_status_codes = str(len(set(potential_status_codes.split())))
            nested_must.append(
                utils.build_more_like_this_query(
                    "1",
                    potential_status_codes,
                    field_name="logs.potential_status_codes",
                    boost=1.0,
                    override_min_should_match=number_of_status_codes,
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
            )

        nested_query = {
            "nested": {
                "path": "logs",
                "score_mode": "max",
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"logs.log_level": {"gte": launch_info.launch.analyzerConfig.minimumLogLevel}}},
                            {"term": {"logs.cluster_with_numbers": not launch_info.cleanNumbers}},
                        ],
                        "must": [{"wildcard": {"logs.cluster_message": "*"}}, *nested_must],
                    }
                },
                "inner_hits": {
                    "size": 5,
                    "_source": [
                        "logs.log_id",
                        "logs.whole_message",
                        "logs.cluster_id",
                        "logs.cluster_message",
                        "logs.cluster_with_numbers",
                        "logs.found_exceptions",
                        "logs.potential_status_codes",
                        "logs.log_level",
                    ],
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
        ind = 0
        for hit in self.os_client.search(launch_info.project, query):
            inner_hits = hit.inner_hits
            if not inner_hits:
                continue
            inner_hits_logs = inner_hits.get("logs", {})
            if not inner_hits_logs:
                continue
            for raw_inner_hit in inner_hits_logs.get("hits", {}).get("hits", []):
                inner_hit = Hit[LogData].from_dict(raw_inner_hit)
                number_of_log_lines = launch_info.numberOfLogLines
                log_message = text_processing.prepare_message_for_clustering(
                    inner_hit.source.whole_message, number_of_log_lines, launch_info.cleanNumbers
                )
                if not log_message or not log_message.strip():
                    continue
                inner_log = Log(
                    test_item_id=hit.source.test_item_id,
                    message=log_message,
                    data=inner_hit.source,
                    launch_id=hit.source.launch_id,
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
                for column in ["found_exceptions", "potential_status_codes"]:
                    candidate_text = " ".join(sorted((getattr(inner_log.data, column, None) or "").split())).strip()
                    text_to_compare = " ".join(sorted((getattr(log.data, column, None) or "").split())).strip()
                    if candidate_text != text_to_compare:
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
                unique_log_ids.add(log.data.log_id)

            # Form final clusters (sent + additionally collected)
            common_clusters, db_updates = gather_cluster_results(
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
