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
from datetime import datetime
from time import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from app.amqp.amqp import AmqpClient
from app.commons import clusterizer, logging
from app.commons.esclient import EsClient
from app.commons.log_merger import LogMerger
from app.commons.log_requests import LogRequests
from app.commons.model.launch_objects import (ClusterResult, ClusterInfo, SearchConfig, ApplicationConfig,
                                              LaunchInfoForClustering)
from app.utils import utils, text_processing

logger = logging.getLogger("analyzerApp.clusterService")


class ClusterService:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    es_client: EsClient
    log_requests: LogRequests
    log_merger: LogMerger

    def __init__(self, app_config: ApplicationConfig, search_cfg: SearchConfig):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = EsClient(app_config=self.app_config)
        self.log_requests = LogRequests()
        self.log_merger = LogMerger()

    def build_search_similar_items_query(self, queried_log, message, launch_info, min_should_match="95%"):
        """Build search query"""
        query = {
            "_source": ["whole_message", "test_item", "is_merged",
                        "detected_message", "stacktrace", "launch_id", "cluster_id",
                        "cluster_message", "potential_status_codes", "found_exceptions"],
            "size": 10,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}}
                    ],
                    "must_not": [{
                        "term": {"test_item": {"value": queried_log["_source"]["test_item"],
                                               "boost": 1.0}}
                    }],
                    "should": [],
                    "must": [
                        {"wildcard": {"cluster_message": "*"}},
                        utils.build_more_like_this_query(
                            min_should_match, message,
                            field_name="whole_message", boost=1.0,
                            override_min_should_match=None,
                            max_query_terms=self.search_cfg.MaxQueryTerms)
                    ]}}}
        if launch_info.forUpdate:
            query["query"]["bool"]["should"].append(
                {"term": {"launch_id": queried_log["_source"]["launch_id"]}})
        else:
            query["query"]["bool"]["must_not"].append(
                {"term": {"launch_id": queried_log["_source"]["launch_id"]}})
        query["query"]["bool"]["filter"].append(
            {"term": {"is_merged": queried_log["_source"]["is_merged"]}})
        query["query"]["bool"]["filter"].append(
            {"term": {"cluster_with_numbers": not launch_info.cleanNumbers}})
        query["query"]["bool"]["should"].append(
            {"term": {"launch_name": launch_info.launch.launchName}})
        if queried_log["_source"]["found_exceptions"].strip():
            query["query"]["bool"]["must"].append(
                utils.build_more_like_this_query(
                    "1",
                    queried_log["_source"]["found_exceptions"],
                    field_name="found_exceptions", boost=1.0,
                    override_min_should_match="1",
                    max_query_terms=self.search_cfg.MaxQueryTerms))
        utils.append_potential_status_codes(query, queried_log, boost=1.0,
                                            max_query_terms=self.search_cfg.MaxQueryTerms)
        return self.add_query_with_start_time_decay(query)

    def add_query_with_start_time_decay(self, main_query):
        return {
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
                                    "decay": self.search_cfg.TimeWeightDecay
                                }
                            }
                        },
                        {
                            "script_score": {"script": {"source": "0.2"}}
                        }],
                    "score_mode": "max",
                    "boost_mode": "multiply"
                }
            }
        }

    def find_similar_items_from_es(
            self, groups, log_dict,
            log_messages, log_ids, launch_info,
            additional_results, unique_errors_min_should_match):
        new_clusters = {}
        _clusterizer = clusterizer.Clusterizer()
        for global_group in groups:
            first_item_ind = groups[global_group][0]
            min_should_match = utils.calculate_threshold_for_text(
                log_messages[first_item_ind],
                unique_errors_min_should_match)
            query = self.build_search_similar_items_query(
                log_dict[first_item_ind],
                log_messages[first_item_ind],
                launch_info,
                min_should_match=text_processing.prepare_es_min_should_match(
                    min_should_match))
            search_results = self.es_client.es_client.search(
                index=log_dict[first_item_ind]["_index"],
                body=query)
            log_messages_part = [log_messages[first_item_ind]]
            log_dict_part = {0: log_dict[first_item_ind]}
            ind = 1
            for res in search_results["hits"]["hits"]:
                if str(res["_id"]) in log_ids:
                    continue
                log_dict_part[ind] = res
                number_of_log_lines = launch_info.numberOfLogLines
                if res["_source"]["is_merged"]:
                    number_of_log_lines = -1
                log_message = text_processing.prepare_message_for_clustering(
                    res["_source"]["whole_message"], number_of_log_lines, launch_info.cleanNumbers)
                cluster_message_processed = text_processing.prepare_message_for_clustering(
                    res["_source"]["cluster_message"], number_of_log_lines, launch_info.cleanNumbers,
                    leave_log_structure=True).strip()
                cluster_message_original = res["_source"]["cluster_message"].strip()
                if cluster_message_original and cluster_message_processed != cluster_message_original:
                    continue
                equal = True
                for column in ["found_exceptions", "potential_status_codes"]:
                    candidate_text = " ".join(sorted(res["_source"][column].split())).strip()
                    text_to_compare = " ".join(
                        sorted(log_dict[first_item_ind]["_source"][column].split())).strip()
                    if candidate_text != text_to_compare:
                        equal = False
                        break
                if not log_message.strip() or not equal:
                    continue
                log_messages_part.append(log_message)
                ind += 1
            groups_part = _clusterizer.find_clusters(log_messages_part, threshold=min_should_match)
            new_group = None
            for group in groups_part:
                if 0 in groups_part[group]:
                    cluster_id = 0
                    cluster_message = ""
                    for ind in groups_part[group]:
                        if ind == 0:  # not to use old cluster id and message
                            continue
                        if log_dict_part[ind]["_source"]["cluster_id"].strip() and int(
                                log_dict_part[ind]["_source"]["cluster_id"].strip()) != 0:
                            cluster_id = int(log_dict_part[ind]["_source"]["cluster_id"].strip())
                            if log_dict_part[ind]["_source"]["cluster_message"].strip():
                                cluster_message = log_dict_part[ind]["_source"]["cluster_message"]
                    new_group_log_ids = []
                    new_test_items = set()
                    for ind in groups_part[group]:
                        if ind == 0:
                            continue
                        if log_dict_part[ind]["_source"]["launch_id"] != launch_info.launch.launchId:
                            continue
                        log_ids.add(str(log_dict_part[ind]["_id"]))
                        new_group_log_ids.append(str(log_dict_part[ind]["_id"]))
                        new_test_items.add(int(log_dict_part[ind]["_source"]["test_item"]))
                    if not cluster_id or not cluster_message:
                        continue
                    new_group = ClusterInfo(
                        logIds=new_group_log_ids,
                        itemIds=list(new_test_items),
                        clusterMessage=cluster_message,
                        clusterId=cluster_id)
                    logger.debug("ES found cluster Id: %s log ids %s cluster message: % s",
                                 cluster_id, new_group_log_ids, cluster_message)
                    break
            if new_group:
                new_clusters[global_group] = new_group
        for group in new_clusters:
            if group in additional_results:
                additional_results[group].logIds.extend(new_clusters[group].logIds)
                additional_results[group].itemIds.extend(new_clusters[group].itemIds)
            else:
                additional_results[group] = new_clusters[group]
        return additional_results

    def calculate_hash(self, group_ids, log_dict, log_messages, launch_info):
        group_logs = []
        log_message = ""
        for i in range(min(100, len(group_ids))):
            ind = group_ids[i]
            group_logs.append(log_messages[ind])
            if not log_message:
                number_of_log_lines = launch_info.numberOfLogLines
                if log_dict[ind]["_source"]["is_merged"]:
                    number_of_log_lines = -1
                log_message = text_processing.prepare_message_for_clustering(
                    log_dict[ind]["_source"]["whole_message"],
                    number_of_log_lines, launch_info.cleanNumbers,
                    leave_log_structure=True).strip()
        _cnt_vectorizer = CountVectorizer(
            binary=True, analyzer="word", token_pattern="[^ ]+", ngram_range=(2, 2))
        group_res = _cnt_vectorizer.fit_transform(group_logs).astype(np.int8)
        res_bitwise = np.bitwise_and.reduce(group_res.toarray(), axis=0)
        bigrams_list = []
        for i, feature_name in enumerate(_cnt_vectorizer.get_feature_names_out()):
            if res_bitwise[i] == 1:
                bigrams_list.append(feature_name)
        hash_message = int(
            hashlib.sha1(" ".join(bigrams_list).encode("utf-8")).hexdigest(), 16) % (10 ** 16)
        hash_message = hash_message * 10 + int(not launch_info.cleanNumbers)
        return hash_message, log_message

    def gather_cluster_results(
            self, groups, additional_results, log_dict, log_messages,
            log_ids_for_merged_logs, launch_info):
        merged_logs_to_update = {}
        clusters_found = {}
        cluster_message_by_id = {}
        for group in groups:
            cnt_items = len(groups[group])
            cluster_id = 0
            cluster_message = ""
            if group in additional_results:
                cnt_items += len(additional_results[group].logIds)
                cluster_id = additional_results[group].clusterId
                cluster_message = additional_results[group].clusterMessage
            if not cluster_id or not cluster_message:
                cluster_id, cluster_message = self.calculate_hash(
                    groups[group], log_dict, log_messages, launch_info)
            log_ids = []
            test_item_ids = []
            for ind in groups[group]:
                if str(utils.extract_real_id(log_dict[ind]["_id"])) != str(log_dict[ind]["_id"]):
                    merged_logs_to_update[log_dict[ind]["_id"]] = (cluster_id, cluster_message)
                    if log_dict[ind]["_id"] in log_ids_for_merged_logs:
                        for _id in log_ids_for_merged_logs[log_dict[ind]["_id"]]:
                            log_ids.append(_id)
                else:
                    log_ids.append(log_dict[ind]["_id"])
                test_item_ids.append(int(log_dict[ind]["_source"]["test_item"]))
            if group in additional_results:
                for log_id in additional_results[group].logIds:
                    if str(utils.extract_real_id(log_id)) != str(log_id):
                        continue
                    log_ids.append(log_id)
            if cluster_id not in clusters_found:
                clusters_found[cluster_id] = (log_ids, test_item_ids)
            else:
                cluster_log_ids, cluster_test_items = clusters_found[cluster_id]
                cluster_log_ids.extend(log_ids)
                cluster_test_items.extend(test_item_ids)
                clusters_found[cluster_id] = (cluster_log_ids, cluster_test_items)
            cluster_message_by_id[cluster_id] = cluster_message
        results_to_return = []
        for cluster_id in clusters_found:
            results_to_return.append(ClusterInfo(
                clusterId=cluster_id,
                clusterMessage=cluster_message_by_id[cluster_id],
                logIds=clusters_found[cluster_id][0],
                itemIds=list(set(clusters_found[cluster_id][1]))))
        return results_to_return, len(results_to_return), merged_logs_to_update

    def regroup_by_error_ans_status_codes(self, log_messages, log_dict):
        regroupped_by_error = {}
        for i in range(len(log_messages)):
            found_exceptions = " ".join(
                sorted(log_dict[i]["_source"]["found_exceptions"].split()))
            potential_status_codes = " ".join(
                sorted(log_dict[i]["_source"]["potential_status_codes"].split()))
            group_key = (found_exceptions, potential_status_codes)
            if group_key not in regroupped_by_error:
                regroupped_by_error[group_key] = []
            regroupped_by_error[group_key].append(i)
        return regroupped_by_error

    def cluster_messages_with_grouping_by_error(self, log_messages, log_dict, unique_errors_min_should_match):
        regroupped_by_error = self.regroup_by_error_ans_status_codes(
            log_messages, log_dict)
        _clusterizer = clusterizer.Clusterizer()
        all_groups = {}
        start_group_id = 0
        for group_key in regroupped_by_error:
            log_messages_part = []
            log_messages_idx_dict = {}
            for i, idx in enumerate(regroupped_by_error[group_key]):
                log_messages_part.append(log_messages[idx])
                log_messages_idx_dict[i] = idx
            groups = _clusterizer.find_clusters(
                log_messages_part,
                threshold=unique_errors_min_should_match)
            max_group_id = max(groups.keys())
            for group_id in groups:
                global_idx = start_group_id + group_id
                if global_idx not in all_groups:
                    all_groups[global_idx] = []
                for i in groups[group_id]:
                    all_groups[global_idx].append(log_messages_idx_dict[i])
            start_group_id = start_group_id + max_group_id + 1
        return all_groups

    @utils.ignore_warnings
    def find_clusters(self, launch_info: LaunchInfoForClustering):
        logger.info("Started clusterizing logs")
        index_name = text_processing.unite_project_name(
            str(launch_info.project), self.app_config.esProjectIndexPrefix)
        if not self.es_client.index_exists(index_name):
            logger.info("Project %s doesn't exist", index_name)
            logger.info("Finished clustering log with 0 clusters.")
            return ClusterResult(
                project=launch_info.project,
                launchId=launch_info.launch.launchId,
                clusters=[])
        t_start = time()
        errors_found = []
        errors_count = 0
        cluster_num = 0
        clusters = []
        log_ids = []
        try:
            unique_errors_min_should_match = launch_info.launch.analyzerConfig.uniqueErrorsMinShouldMatch / 100  # noqa
            log_messages, log_dict, log_ids_for_merged_logs = self.log_requests.prepare_logs_for_clustering(  # noqa
                launch_info.launch, launch_info.numberOfLogLines, launch_info.cleanNumbers, index_name)
            log_ids = set([str(log["_id"]) for log in log_dict.values()])

            groups = self.cluster_messages_with_grouping_by_error(
                log_messages, log_dict,
                unique_errors_min_should_match)
            logger.debug("Groups: %s", groups)
            additional_results = self.find_similar_items_from_es(
                groups, log_dict, log_messages,
                log_ids, launch_info,
                {}, unique_errors_min_should_match)
            clusters, cluster_num, merged_logs_to_update = self.gather_cluster_results(
                groups, additional_results, log_dict, log_messages,
                log_ids_for_merged_logs, launch_info)
            if clusters:
                bodies = []
                for result in clusters:
                    logger.debug("Cluster Id: %s, Cluster message: %s",
                                 result.clusterId, result.clusterMessage)
                    logger.debug("Cluster Ids: %s", result.logIds)
                    for log_id in result.logIds:
                        bodies.append({
                            "_op_type": "update",
                            "_id": log_id,
                            "_index": index_name,
                            "doc": {"cluster_id": str(result.clusterId),
                                    "cluster_message": result.clusterMessage,
                                    "cluster_with_numbers": not launch_info.cleanNumbers}})
                for log_id in merged_logs_to_update:
                    cluster_id, cluster_message = merged_logs_to_update[log_id]
                    bodies.append({
                        "_op_type": "update",
                        "_id": log_id,
                        "_index": index_name,
                        "doc": {"cluster_id": str(cluster_id),
                                "cluster_message": cluster_message,
                                "cluster_with_numbers": not launch_info.cleanNumbers}})
                self.es_client._bulk_index(
                    bodies, refresh=False, chunk_size=self.app_config.esChunkNumberUpdateClusters)
        except Exception as exc:
            logger.exception(exc)
            errors_found.append(utils.extract_exception(exc))
            errors_count += 1

        results_to_share = {launch_info.launch.launchId: {
            "not_found": int(cluster_num == 0), "items_to_process": len(log_ids),
            "processed_time": time() - t_start, "found_clusters": cluster_num,
            "launch_id": launch_info.launch.launchId, "launch_name": launch_info.launch.launchName,
            "project_id": launch_info.project, "method": "find_clusters",
            "gather_date": datetime.now().strftime("%Y-%m-%d"),
            "gather_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "module_version": [self.app_config.appVersion],
            "model_info": [],
            "errors": errors_found,
            "errors_count": errors_count}}
        if self.app_config.amqpUrl.strip():
            AmqpClient(self.app_config.amqpUrl).send_to_inner_queue(
                self.app_config.exchangeName, 'stats_info', json.dumps(results_to_share))

        logger.debug("Stats info %s", results_to_share)
        logger.info("Processed the launch. It took %.2f sec.", time() - t_start)
        logger.info("Finished clustering for the launch with %d clusters.", cluster_num)
        return ClusterResult(
            project=launch_info.project,
            launchId=launch_info.launch.launchId,
            clusters=clusters)
