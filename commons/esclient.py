"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import re
import json
import logging
import requests
import elasticsearch
import elasticsearch.helpers
import commons.launch_objects
from commons.launch_objects import SuggestAnalysisResult, SearchLogInfo
from commons.launch_objects import AnalysisResult, SearchLogInfo
import utils.utils as utils
from boosting_decision_making import boosting_featurizer
from boosting_decision_making.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from time import time, sleep
from datetime import datetime
from boosting_decision_making import weighted_similarity_calculator
from boosting_decision_making import similarity_calculator
from boosting_decision_making import boosting_decision_maker
from commons.es_query_builder import EsQueryBuilder
from commons.log_merger import LogMerger
from queue import Queue
from threading import Thread

ERROR_LOGGING_LEVEL = 40000

logger = logging.getLogger("analyzerApp.esclient")


class EsClient:
    """Elasticsearch client implementation"""
    def __init__(self, host="http://localhost:9200", search_cfg={}):
        self.host = host
        self.search_cfg = search_cfg
        self.es_client = elasticsearch.Elasticsearch([host], timeout=30,
                                                     max_retries=5, retry_on_timeout=True)
        self.boosting_decision_maker = None
        self.suggest_decision_maker = None
        self.weighted_log_similarity_calculator = None
        self.suggest_threshold = 0.4
        self.es_query_builder = EsQueryBuilder(self.search_cfg, ERROR_LOGGING_LEVEL)
        self.initialize_decision_makers()

    def initialize_decision_makers(self):
        if self.search_cfg["BoostModelFolder"].strip():
            self.boosting_decision_maker = boosting_decision_maker.BoostingDecisionMaker(
                folder=self.search_cfg["BoostModelFolder"])
        if self.search_cfg["SuggestBoostModelFolder"].strip():
            self.suggest_decision_maker = boosting_decision_maker.BoostingDecisionMaker(
                folder=self.search_cfg["SuggestBoostModelFolder"])
        if self.search_cfg["SimilarityWeightsFolder"].strip():
            self.weighted_log_similarity_calculator = weighted_similarity_calculator.\
                WeightedSimilarityCalculator(folder=self.search_cfg["SimilarityWeightsFolder"])

    def create_index(self, index_name):
        """Create index in elasticsearch"""
        logger.debug("Creating '%s' Elasticsearch index", str(index_name))
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        try:
            response = self.es_client.indices.create(index=str(index_name), body={
                'settings': utils.read_json_file("", "index_settings.json", to_json=True),
                'mappings': utils.read_json_file("", "index_mapping_settings.json", to_json=True)
            })
            logger.debug("Created '%s' Elasticsearch index", str(index_name))
            return commons.launch_objects.Response(**response)
        except Exception as err:
            logger.error("Couldn't create index")
            logger.error("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.error(err)
            return commons.launch_objects.Response()

    @staticmethod
    def send_request(url, method):
        """Send request with specified url and http method"""
        try:
            response = requests.get(url) if method == "GET" else {}
            data = response._content.decode("utf-8")
            content = json.loads(data, strict=False)
            return content
        except Exception as err:
            logger.error("Error with loading url: %s", url)
            logger.error(err)
        return []

    def is_healthy(self):
        """Check whether elasticsearch is healthy"""
        try:
            url = utils.build_url(self.host, ["_cluster/health"])
            res = EsClient.send_request(url, "GET")
            return res["status"] in ["green", "yellow"]
        except Exception as err:
            logger.error("Elasticsearch is not healthy")
            logger.error(err)
            return False

    def list_indices(self):
        """Get all indices from elasticsearch"""
        url = utils.build_url(self.host, ["_cat", "indices?format=json"])
        res = EsClient.send_request(url, "GET")
        return res

    def index_exists(self, index_name):
        """Checks whether index exists"""
        try:
            index = self.es_client.indices.get(index=str(index_name))
            return index is not None
        except Exception as err:
            logger.error("Index %s was not found", str(index_name))
            logger.error("ES Url %s", self.host)
            logger.error(err)
            return False

    def delete_index(self, index_name):
        """Delete the whole index"""
        try:
            self.es_client.indices.delete(index=str(index_name))
            logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.debug("Deleted index %s", str(index_name))
            return 1
        except Exception as err:
            logger.error("Not found %s for deleting", str(index_name))
            logger.error("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.error(err)
            return 0

    def create_index_if_not_exists(self, index_name):
        """Creates index if it doesn't not exist"""
        if not self.index_exists(index_name):
            return self.create_index(index_name)
        return True

    def index_logs(self, launches):
        """Index launches to the index with project name"""
        logger.info("Indexing logs for %d launches", len(launches))
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        t_start = time()
        bodies = []
        test_item_ids = []
        project = None
        for launch in launches:
            self.create_index_if_not_exists(str(launch.project))
            project = str(launch.project)

            for test_item in launch.testItems:
                logs_added = False
                for log in test_item.logs:

                    if log.logLevel < ERROR_LOGGING_LEVEL or not log.message.strip():
                        continue

                    bodies.append(self._prepare_log(launch, test_item, log))
                    logs_added = True
                if logs_added:
                    test_item_ids.append(str(test_item.testItemId))
        result = self._bulk_index(bodies)
        self._merge_logs(test_item_ids, project)
        logger.info("Finished indexing logs for %d launches. It took %.2f sec.",
                    len(launches), time() - t_start)
        return result

    def clean_message(self, message):
        message = utils.replace_tabs_for_newlines(message)
        message = utils.fix_big_encoded_urls(message)
        message = utils.reverse_log_if_needed(message)
        message = utils.remove_generated_parts(message)
        message = utils.clean_html(message)
        message = utils.delete_empty_lines(message)
        message = utils.leave_only_unique_lines(message)
        return message

    def _create_log_template(self):
        return {
            "_id":    "",
            "_index": "",
            "_source": {
                "launch_id":        "",
                "launch_name":      "",
                "test_item":        "",
                "unique_id":        "",
                "test_case_hash":   0,
                "is_auto_analyzed": False,
                "issue_type":       "",
                "log_level":        0,
                "original_message_lines": 0,
                "original_message_words_number": 0,
                "message":          "",
                "is_merged":        False,
                "start_time":       "",
                "merged_small_logs":  "",
                "detected_message": "",
                "detected_message_with_numbers": "",
                "stacktrace":                    "",
                "only_numbers":                  "",
                "found_exceptions":              ""}}

    def _fill_launch_test_item_fields(self, log_template, launch, test_item):
        log_template["_index"] = launch.project
        log_template["_source"]["launch_id"] = launch.launchId
        log_template["_source"]["launch_name"] = launch.launchName
        log_template["_source"]["test_item"] = test_item.testItemId
        log_template["_source"]["unique_id"] = test_item.uniqueId
        log_template["_source"]["test_case_hash"] = test_item.testCaseHash
        log_template["_source"]["is_auto_analyzed"] = test_item.isAutoAnalyzed
        log_template["_source"]["issue_type"] = test_item.issueType
        log_template["_source"]["start_time"] = datetime(
            *test_item.startTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
        return log_template

    def _fill_log_fields(self, log_template, log, number_of_lines):
        cleaned_message = self.clean_message(log.message)

        message = utils.first_lines(cleaned_message, number_of_lines)
        message_without_params = message
        message = utils.sanitize_text(message)

        message_without_params = utils.clean_from_urls(message_without_params)
        message_without_params = utils.clean_from_paths(message_without_params)
        message_without_params = utils.clean_from_params(message_without_params)
        message_without_params = utils.sanitize_text(message_without_params)

        detected_message, stacktrace = utils.detect_log_description_and_stacktrace(cleaned_message)

        detected_message_without_params = detected_message
        urls = " ".join(utils.extract_urls(detected_message_without_params))
        detected_message_without_params = utils.clean_from_urls(detected_message_without_params)
        paths = " ".join(utils.extract_paths(detected_message_without_params))
        detected_message_without_params = utils.clean_from_paths(detected_message_without_params)
        message_params = " ".join(utils.extract_message_params(detected_message_without_params))
        detected_message_without_params = utils.clean_from_params(detected_message_without_params)
        detected_message_without_params = utils.sanitize_text(detected_message_without_params)

        detected_message_with_numbers = utils.remove_starting_datetime(detected_message)
        detected_message_only_numbers = utils.find_only_numbers(detected_message_with_numbers)
        detected_message = utils.sanitize_text(detected_message)
        stacktrace = utils.sanitize_text(stacktrace)
        found_exceptions = utils.get_found_exceptions(detected_message)
        found_exceptions_extended = utils.enrich_found_exceptions(found_exceptions)

        log_template["_id"] = log.logId
        log_template["_source"]["log_level"] = log.logLevel
        log_template["_source"]["original_message_lines"] = utils.calculate_line_number(cleaned_message)
        log_template["_source"]["original_message_words_number"] = len(
            utils.split_words(cleaned_message, split_urls=False))
        log_template["_source"]["message"] = message
        log_template["_source"]["detected_message"] = detected_message
        log_template["_source"]["detected_message_with_numbers"] = detected_message_with_numbers
        log_template["_source"]["stacktrace"] = stacktrace
        log_template["_source"]["only_numbers"] = detected_message_only_numbers
        log_template["_source"]["urls"] = urls
        log_template["_source"]["paths"] = paths
        log_template["_source"]["message_params"] = message_params
        log_template["_source"]["found_exceptions"] = found_exceptions
        log_template["_source"]["found_exceptions_extended"] = found_exceptions_extended
        log_template["_source"]["detected_message_extended"] =\
            utils.enrich_text_with_method_and_classes(detected_message)
        log_template["_source"]["detected_message_without_params_extended"] =\
            utils.enrich_text_with_method_and_classes(detected_message_without_params)
        log_template["_source"]["stacktrace_extended"] =\
            utils.enrich_text_with_method_and_classes(stacktrace)
        log_template["_source"]["message_extended"] =\
            utils.enrich_text_with_method_and_classes(message)
        log_template["_source"]["message_without_params_extended"] =\
            utils.enrich_text_with_method_and_classes(message_without_params)

        for field in ["message", "detected_message", "detected_message_with_numbers",
                      "stacktrace", "only_numbers", "found_exceptions", "found_exceptions_extended",
                      "detected_message_extended", "detected_message_without_params_extended",
                      "stacktrace_extended", "message_extended", "message_without_params_extended"]:
            log_template["_source"][field] = utils.leave_only_unique_lines(log_template["_source"][field])
            log_template["_source"][field] = utils.clean_colon_stacking(log_template["_source"][field])
        return log_template

    def _prepare_log(self, launch, test_item, log):
        log_template = self._create_log_template()
        log_template = self._fill_launch_test_item_fields(log_template, launch, test_item)
        log_template = self._fill_log_fields(log_template, log, launch.analyzerConfig.numberOfLogLines)
        return log_template

    def _fill_test_item_info_fields(self, log_template, test_item_info):
        log_template["_index"] = test_item_info.project
        log_template["_source"]["launch_id"] = test_item_info.launchId
        log_template["_source"]["launch_name"] = test_item_info.launchName
        log_template["_source"]["test_item"] = test_item_info.testItemId
        log_template["_source"]["unique_id"] = test_item_info.uniqueId
        log_template["_source"]["test_case_hash"] = test_item_info.testCaseHash
        log_template["_source"]["is_auto_analyzed"] = False
        log_template["_source"]["issue_type"] = ""
        log_template["_source"]["start_time"] = ""
        return log_template

    def _prepare_log_for_suggests(self, test_item_info, log):
        log_template = self._create_log_template()
        log_template = self._fill_test_item_info_fields(log_template, test_item_info)
        log_template = self._fill_log_fields(
            log_template, log, test_item_info.analyzerConfig.numberOfLogLines)
        return log_template

    def _merge_logs(self, test_item_ids, project):
        bodies = []
        batch_size = 1000
        self._delete_merged_logs(test_item_ids, project)
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            test_items = test_item_ids[i * batch_size: (i + 1) * batch_size]
            if not test_items:
                continue
            test_items_dict = {}
            for r in elasticsearch.helpers.scan(self.es_client,
                                                query=self.es_query_builder.get_test_item_query(
                                                    test_items, False),
                                                index=project):
                test_item_id = r["_source"]["test_item"]
                if test_item_id not in test_items_dict:
                    test_items_dict[test_item_id] = []
                test_items_dict[test_item_id].append(r)
            for test_item_id in test_items_dict:
                merged_logs = LogMerger.decompose_logs_merged_and_without_duplicates(
                    test_items_dict[test_item_id])
                for log in merged_logs:
                    if log["_source"]["is_merged"]:
                        bodies.append(log)
                    else:
                        bodies.append({
                            "_op_type": "update",
                            "_id": log["_id"],
                            "_index": log["_index"],
                            "doc": {"merged_small_logs": log["_source"]["merged_small_logs"]}
                        })
        return self._bulk_index(bodies)

    def _delete_merged_logs(self, test_items_to_delete, project):
        logger.debug("Delete merged logs for %d test items", len(test_items_to_delete))
        bodies = []
        batch_size = 1000
        for i in range(int(len(test_items_to_delete) / batch_size) + 1):
            test_item_ids = test_items_to_delete[i * batch_size: (i + 1) * batch_size]
            if not test_item_ids:
                continue
            for log in elasticsearch.helpers.scan(self.es_client,
                                                  query=self.es_query_builder.get_test_item_query(
                                                      test_item_ids, True),
                                                  index=project):
                bodies.append({
                    "_op_type": "delete",
                    "_id": log["_id"],
                    "_index": project
                })
        if bodies:
            self._bulk_index(bodies)

    @staticmethod
    def get_test_item_query(test_item_ids, is_merged):
        """Build test item query"""
        return {"size": 10000,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"test_item": [str(_id) for _id in test_item_ids]}},
                            {"term": {"is_merged": is_merged}}
                        ]
                    }
                }}

    @staticmethod
    def merge_big_and_small_logs(logs, log_level_ids_to_add,
                                 log_level_messages, log_level_ids_merged):
        """Merge big message logs with small ones"""
        new_logs = []
        for log in logs:
            if not log["_source"]["message"].strip():
                continue
            log_level = log["_source"]["log_level"]

            if log["_id"] in log_level_ids_to_add[log_level]:
                merged_small_logs = utils.compress(
                    log_level_messages[log["_source"]["log_level"]])
                new_logs.append(EsClient.prepare_new_log(
                    log, log["_id"], False, merged_small_logs))

        for log_level in log_level_messages:

            if not log_level_ids_to_add[log_level] \
                    and log_level_messages[log_level].strip():
                log = log_level_ids_merged[log_level]
                new_logs.append(EsClient.prepare_new_log(
                    log, str(log["_id"]) + "_m", True,
                    utils.compress(log_level_messages[log_level]),
                    fields_to_clean=["message", "detected_message", "only_numbers",
                                     "detected_message_with_numbers", "stacktrace"]))
        return new_logs

    @staticmethod
    def decompose_logs_merged_and_without_duplicates(logs):
        """Merge big logs with small ones without duplcates"""
        log_level_messages = {}
        log_level_ids_to_add = {}
        log_level_ids_merged = {}
        logs_unique_log_level = {}

        for log in logs:
            if not log["_source"]["message"].strip():
                continue

            log_level = log["_source"]["log_level"]

            if log_level not in log_level_messages:
                log_level_messages[log_level] = ""
            if log_level not in log_level_ids_to_add:
                log_level_ids_to_add[log_level] = []
            if log_level not in logs_unique_log_level:
                logs_unique_log_level[log_level] = set()

            if log["_source"]["original_message_lines"] <= 2 and\
                    log["_source"]["original_message_words_number"] <= 100:
                if log_level not in log_level_ids_merged:
                    log_level_ids_merged[log_level] = log
                message = log["_source"]["message"]
                normalized_msg = " ".join(message.strip().lower().split())
                if normalized_msg not in logs_unique_log_level[log_level]:
                    logs_unique_log_level[log_level].add(normalized_msg)
                    log_level_messages[log_level] = log_level_messages[log_level]\
                        + message + "\r\n"
            else:
                log_level_ids_to_add[log_level].append(log["_id"])

        return EsClient.merge_big_and_small_logs(logs, log_level_ids_to_add,
                                                 log_level_messages, log_level_ids_merged)

    @staticmethod
    def prepare_new_log(old_log, new_id, is_merged, merged_small_logs, fields_to_clean=[]):
        """Prepare updated log"""
        merged_log = copy.deepcopy(old_log)
        merged_log["_source"]["is_merged"] = is_merged
        merged_log["_id"] = new_id
        merged_log["_source"]["merged_small_logs"] = merged_small_logs
        for field in fields_to_clean:
            merged_log["_source"][field] = ""
        return merged_log

    def _bulk_index(self, bodies, refresh=True):
        if not bodies:
            return commons.launch_objects.BulkResponse(took=0, errors=False)
        logger.debug("Indexing %d logs...", len(bodies))
        try:
            success_count, errors = elasticsearch.helpers.bulk(self.es_client,
                                                               bodies,
                                                               chunk_size=1000,
                                                               request_timeout=30,
                                                               refresh=refresh)

            logger.debug("Processed %d logs", success_count)
            if errors:
                logger.debug("Occured errors %s", errors)
            return commons.launch_objects.BulkResponse(took=success_count, errors=len(errors) > 0)
        except Exception as err:
            logger.error("Error in bulk")
            logger.error("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.error(err)
            return commons.launch_objects.BulkResponse(took=0, errors=True)

    def delete_logs(self, clean_index):
        """Delete logs from elasticsearch"""
        logger.info("Delete logs %s for the project %s",
                    clean_index.ids, clean_index.project)
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        t_start = time()
        if not self.index_exists(clean_index.project):
            return 0
        test_item_ids = set()
        try:
            search_query = self.es_query_builder.build_search_test_item_ids_query(
                clean_index.ids)
            for res in elasticsearch.helpers.scan(self.es_client,
                                                  query=search_query,
                                                  index=clean_index.project,
                                                  scroll="5m"):
                test_item_ids.add(res["_source"]["test_item"])
        except Exception as err:
            logger.error("Couldn't find test items for logs")
            logger.error(err)

        bodies = []
        for _id in clean_index.ids:
            bodies.append({
                "_op_type": "delete",
                "_id":      _id,
                "_index":   clean_index.project,
            })
        result = self._bulk_index(bodies)
        self._merge_logs(list(test_item_ids), clean_index.project)
        logger.info("Finished deleting logs %s for the project %s. It took %.2f sec",
                    clean_index.ids, clean_index.project, time() - t_start)
        return result.took

    def search_logs(self, search_req):
        """Get all logs similar to given logs"""
        similar_log_ids = set()
        logger.info("Started searching by request %s", search_req.json())
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        t_start = time()
        if not self.index_exists(str(search_req.projectId)):
            return []
        searched_logs = set()
        test_item_info = {}

        for message in search_req.logMessages:
            if not message.strip():
                continue

            queried_log = self._create_log_template()
            queried_log = self._fill_log_fields(queried_log,
                                                commons.launch_objects.Log(logId=0, message=message),
                                                search_req.logLines)

            msg_words = " ".join(utils.split_words(queried_log["_source"]["message"]))
            if not msg_words.strip() or msg_words in searched_logs:
                continue
            searched_logs.add(msg_words)
            query = self.es_query_builder.build_search_query(
                search_req, queried_log["_source"]["message"])
            res = self.es_client.search(index=str(search_req.projectId), body=query)
            for es_res in res["hits"]["hits"]:
                test_item_info[es_res["_id"]] = es_res["_source"]["test_item"]

            _similarity_calculator = similarity_calculator.SimilarityCalculator(
                {
                    "max_query_terms": self.search_cfg["MaxQueryTerms"],
                    "min_word_length": self.search_cfg["MinWordLength"],
                    "min_should_match": "90%",
                    "number_of_log_lines": search_req.logLines
                },
                weighted_similarity_calculator=self.weighted_log_similarity_calculator)
            _similarity_calculator.find_similarity([(queried_log, res)], ["message"])

            for group_id, similarity_obj in _similarity_calculator.similarity_dict["message"].items():
                log_id, _ = group_id
                similarity_percent = similarity_obj["similarity"]
                logger.debug("Log with id %s has %.3f similarity with the queried log '%s'",
                             log_id, similarity_percent, queried_log["_source"]["message"])
                if similarity_percent >= self.search_cfg["SearchLogsMinSimilarity"]:
                    similar_log_ids.add((utils.extract_real_id(log_id), int(test_item_info[log_id])))

        logger.info("Finished searching by request %s with %d results. It took %.2f sec.",
                    search_req.json(), len(similar_log_ids), time() - t_start)
        return [SearchLogInfo(logId=log_info[0],
                              testItemId=log_info[1]) for log_info in similar_log_ids]

    @staticmethod
    def build_more_like_this_query(max_query_terms,
                                   min_should_match, log_message,
                                   field_name="message", boost=1.0,
                                   override_min_should_match=None):
        """Build more like this query"""
        return {"more_like_this": {
            "fields":               [field_name],
            "like":                 log_message,
            "min_doc_freq":         1,
            "min_term_freq":        1,
            "minimum_should_match":
                ("5<" + min_should_match) if override_min_should_match is None else override_min_should_match,
            "max_query_terms":      max_query_terms,
            "boost": boost, }}

    def build_analyze_query(self, launch, unique_id, log, size=10):
        """Build analyze query"""
        min_should_match = "{}%".format(launch.analyzerConfig.minShouldMatch)\
            if launch.analyzerConfig.minShouldMatch \
            else self.search_cfg["MinShouldMatch"]

        query = {"size": size,
                 "sort": ["_score",
                          {"start_time": "desc"}, ],
                 "query": {
                     "bool": {
                         "filter": [
                             {"range": {"log_level": {"gte": ERROR_LOGGING_LEVEL}}},
                             {"exists": {"field": "issue_type"}},
                         ],
                         "must_not": [
                             {"wildcard": {"issue_type": "TI*"}},
                             {"wildcard": {"issue_type": "ti*"}},
                             {"wildcard": {"issue_type": "nd*"}},
                             {"wildcard": {"issue_type": "ND*"}},
                             {"term": {"test_item": log["_source"]["test_item"]}}
                         ],
                         "must": [],
                         "should": [
                             {"term": {"unique_id": {
                                 "value": unique_id,
                                 "boost": abs(self.search_cfg["BoostUniqueID"])}}},
                             {"term": {"test_case_hash": {
                                 "value": log["_source"]["test_case_hash"],
                                 "boost": abs(self.search_cfg["BoostUniqueID"])}}},
                             {"term": {"is_auto_analyzed": {
                                 "value": str(self.search_cfg["BoostAA"] < 0).lower(),
                                 "boost": abs(self.search_cfg["BoostAA"]), }}},
                         ]}}}

        if launch.analyzerConfig.analyzerMode in ["LAUNCH_NAME"]:
            query["query"]["bool"]["must"].append(
                {"term": {
                    "launch_name": {
                        "value": launch.launchName}}})
        elif launch.analyzerConfig.analyzerMode in ["CURRENT_LAUNCH"]:
            query["query"]["bool"]["must"].append(
                {"term": {
                    "launch_id": {
                        "value": launch.launchId}}})
        else:
            query["query"]["bool"]["should"].append(
                {"term": {
                    "launch_name": {
                        "value": launch.launchName,
                        "boost": abs(self.search_cfg["BoostLaunch"])}}})
        if log["_source"]["message"].strip():
            log_lines = launch.analyzerConfig.numberOfLogLines
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            if log_lines == -1:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                    min_should_match,
                                                    log["_source"]["detected_message"],
                                                    field_name="detected_message",
                                                    boost=4.0))
                if log["_source"]["stacktrace"].strip() != "":
                    query["query"]["bool"]["must"].append(
                        self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                        min_should_match,
                                                        log["_source"]["stacktrace"],
                                                        field_name="stacktrace",
                                                        boost=2.0))
                else:
                    query["query"]["bool"]["must_not"].append({"wildcard": {"stacktrace": "*"}})
            else:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                    min_should_match,
                                                    log["_source"]["message"],
                                                    field_name="message",
                                                    boost=4.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                    "80%",
                                                    log["_source"]["detected_message"],
                                                    field_name="detected_message",
                                                    boost=2.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                    "60%",
                                                    log["_source"]["stacktrace"],
                                                    field_name="stacktrace", boost=1.0))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                "80%",
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=0.5))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                "1",
                                                log["_source"]["only_numbers"],
                                                field_name="only_numbers",
                                                boost=4.0,
                                                override_min_should_match="1"))
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                min_should_match,
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=2.0))
        return query

    def leave_only_unique_logs(self, logs):
        unique_logs = set()
        all_logs = []
        for log in logs:
            if log.message.strip() not in unique_logs:
                all_logs.append(log)
                unique_logs.add(log.message.strip())
        return all_logs

    def prepare_query_for_batches(self, launches):
        all_queries = []
        all_query_logs = []
        launch_test_id_dict = {}
        index_log_id = 0
        for launch in launches:
            if not self.index_exists(str(launch.project)):
                continue
            for test_item in launch.testItems:
                unique_logs = self.leave_only_unique_logs(test_item.logs)
                prepared_logs = [self._prepare_log(launch, test_item, log)
                                 for log in unique_logs if log.logLevel >= ERROR_LOGGING_LEVEL]
                results = self.decompose_logs_merged_and_without_duplicates(prepared_logs)

                for log in results:
                    message = log["_source"]["message"].strip()
                    merged_logs = log["_source"]["merged_small_logs"].strip()
                    if log["_source"]["log_level"] < ERROR_LOGGING_LEVEL or\
                            (message == "" and merged_logs == ""):
                        continue

                    query = self.build_analyze_query(launch, test_item.uniqueId, log,
                                                     launch.analyzerConfig.numberOfLogLines)
                    full_query = "{}\n{}".format(json.dumps({"index": launch.project}), json.dumps(query))
                    all_queries.append(full_query)
                    all_query_logs.append(log)
                    if (launch.launchId, test_item.testItemId) not in launch_test_id_dict:
                        launch_test_id_dict[(launch.launchId, test_item.testItemId)] = []
                    launch_test_id_dict[(launch.launchId, test_item.testItemId)].append(index_log_id)
                    index_log_id += 1
        return all_queries, all_query_logs, launch_test_id_dict

    def query_elasticsearch_by_batches(self, all_queries, batch_size=50, max_workers=3):
        partial_batches = []
        for i in range(int(len(all_queries) / batch_size) + 1):
            part_batch = all_queries[i * batch_size: (i + 1) * batch_size]
            if not part_batch:
                continue
            partial_batches.append("\n".join(part_batch) + "\n")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self.es_client.msearch, partial_batches)
        results_all = []
        for r in results:
            results_all.extend(r["responses"])
        return results_all

    def get_bulk_search_results(self, launches):
        all_queries, all_query_logs, launch_test_id_dict = self.prepare_query_for_batches(launches)
        results_all = self.query_elasticsearch_by_batches(all_queries)
        es_results = []
        for launch in launches:
            for test_item in launch.testItems:
                if (launch.launchId, test_item.testItemId) not in launch_test_id_dict:
                    continue
                log_results_part = []
                for idx in launch_test_id_dict[(launch.launchId, test_item.testItemId)]:
                    log_results_part.append((all_query_logs[idx], results_all[idx]))
                es_results.append((launch.analyzerConfig, test_item.testItemId, log_results_part))
        return es_results

    def prepare_features_for_analysis(self, es_results, batch_size=100):
        num_chunks = int(len(es_results) / batch_size) + 1

        es_results_to_process = []
        for i in range(num_chunks):
            partial_result = es_results[i * batch_size: (i + 1) * batch_size]
            if not partial_result:
                continue
            es_results_to_process.append(partial_result)

        config = {
            "max_query_terms": self.search_cfg["MaxQueryTerms"],
            "min_should_match": self.search_cfg["MinShouldMatch"],
            "min_word_length": self.search_cfg["MinWordLength"],
            "filter_min_should_match": self.search_cfg["FilterMinShouldMatch"]
        }

        process_results, map_with_process_results = [], {}
        parallel_analysis = self.search_cfg["AllowParallelAnalysis"]\
            if "AllowParallelAnalysis" in self.search_cfg else False
        if parallel_analysis and len(es_results_to_process) >= 2:
            process_results, map_with_process_results = self.run_features_calculation_parallel(
                es_results_to_process, config)
        process_results = self.run_features_calculation_sequentially(es_results_to_process, config,
                                                                     process_results,
                                                                     map_with_process_results)
        return es_results_to_process, process_results

    def run_features_calculation_parallel(self, es_results_to_process, config):
        process_results = []
        try:
            with Pool(processes=2) as pool:
                all_line_features = self.get_decision_maker(-1).get_feature_ids()
                not_all_line_features = self.get_decision_maker(2).get_feature_ids()
                process_results = pool.map(
                    calculate_features,
                    [(res, (all_line_features, not_all_line_features), i, config, True)
                     for i, res in enumerate(es_results_to_process)])
        except Exception as e:
            logger.error("Couldn't process items in parallel. It will be processed sequentially.")
            logger.error(e)
        map_with_process_results = {}
        if len(process_results) != len(es_results_to_process):
            logger.error("Couldn't process items in parallel. It will be processed sequentially.")

        for i, result in enumerate(process_results):
            map_with_process_results[result[0]] = i
        return process_results, map_with_process_results

    def run_features_calculation_sequentially(self, es_results_to_process, config,
                                              old_results, map_with_process_results):
        process_results = []

        for i, res in enumerate(es_results_to_process):
            if i in map_with_process_results:
                process_results.append(old_results[map_with_process_results[i]])
            else:
                all_line_features = self.get_decision_maker(-1).get_feature_ids()
                not_all_line_features = self.get_decision_maker(2).get_feature_ids()
                process_results.extend(
                    calculate_features(
                        (res, (all_line_features, not_all_line_features), i, config, False)))
        return process_results

    def choose_issue_type(self, predicted_labels, predicted_labels_probability,
                          issue_type_names, boosting_data_gatherer):
        predicted_issue_type = ""
        max_val = 0.0
        max_val_start_time = None
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 1:
                issue_type = issue_type_names[i]
                chosen_type =\
                    boosting_data_gatherer.scores_by_issue_type[issue_type]
                start_time = chosen_type["mrHit"]["_source"]["start_time"]
                if (predicted_labels_probability[i][1] > max_val) or\
                        ((predicted_labels_probability[i][1] == max_val) and # noqa
                            (max_val_start_time is None or start_time > max_val_start_time)):
                    max_val = predicted_labels_probability[i][1]
                    predicted_issue_type = issue_type
                    max_val_start_time = start_time
        return predicted_issue_type

    def get_config_for_boosting(self, analyzer_config):
        min_should_match = analyzer_config.minShouldMatch / 100 if analyzer_config.minShouldMatch > 0 else\
            int(re.search(r"\d+", self.search_cfg["MinShouldMatch"]).group(0)) / 100
        return {
            "max_query_terms": self.search_cfg["MaxQueryTerms"],
            "min_should_match": min_should_match,
            "min_word_length": self.search_cfg["MinWordLength"],
            "filter_min_should_match": utils.choose_fields_to_filter(analyzer_config.numberOfLogLines),
            "number_of_log_lines": analyzer_config.numberOfLogLines
        }

    def _send_result_to_queue(self, test_item_dict, batches, batch_logs):
        partial_res = self.es_client.msearch("\n".join(batches) + "\n")["responses"]
        for test_item_id in test_item_dict:
            new_result = []
            for ind in test_item_dict[test_item_id]:
                all_info = batch_logs[ind]
                new_result.append((all_info[2], partial_res[ind]))
            self.queue.put((all_info[0], all_info[1], new_result))

    def _query_elasticsearch(self, launches, max_batch_size=30):
        t_start = time()
        batches = []
        batch_logs = []
        index_in_batch = 0
        test_item_dict = {}
        batch_size = 5
        n_first_blocks = 3
        try:
            for launch in launches:
                if not self.index_exists(str(launch.project)):
                    continue
                for test_item in launch.testItems:
                    unique_logs = utils.leave_only_unique_logs(test_item.logs)
                    prepared_logs = [self._prepare_log(launch, test_item, log)
                                     for log in unique_logs if log.logLevel >= ERROR_LOGGING_LEVEL]
                    results = LogMerger.decompose_logs_merged_and_without_duplicates(prepared_logs)

                    for log in results:
                        message = log["_source"]["message"].strip()
                        merged_logs = log["_source"]["merged_small_logs"].strip()
                        if log["_source"]["log_level"] < ERROR_LOGGING_LEVEL or\
                                (not message and not merged_logs):
                            continue

                        query = self.es_query_builder.build_analyze_query(
                            launch, log, launch.analyzerConfig.numberOfLogLines)
                        full_query = "{}\n{}".format(json.dumps({"index": launch.project}), json.dumps(query))
                        batches.append(full_query)
                        batch_logs.append((launch.analyzerConfig, test_item.testItemId, log))
                        if test_item.testItemId not in test_item_dict:
                            test_item_dict[test_item.testItemId] = []
                        test_item_dict[test_item.testItemId].append(index_in_batch)
                        index_in_batch += 1
                    if n_first_blocks <= 0:
                        batch_size = max_batch_size
                    if len(batches) >= batch_size:
                        n_first_blocks -= 1
                        self._send_result_to_queue(test_item_dict, batches, batch_logs)
                        batches = []
                        batch_logs = []
                        test_item_dict = {}
                        index_in_batch = 0
            if len(batches) > 0:
                self._send_result_to_queue(test_item_dict, batches, batch_logs)
        except Exception as err:
            logger.error("Error in ES query")
            logger.error(err)
        self.finished_queue.put("Finished")
        logger.info("Es queries finished %.2f s.", time() - t_start)

    @utils.ignore_warnings
    def analyze_logs(self, launches):
        logger.info("Started analysis for %d launches", len(launches))
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        self.queue = Queue()
        self.finished_queue = Queue()

        es_query_thread = Thread(target=self._query_elasticsearch, args=(launches, ))
        es_query_thread.daemon = True
        es_query_thread.start()
        results = []
        t_start = time()
        cnt_items_to_process = 0
        while self.finished_queue.empty() or not self.queue.empty():
            if self.queue.empty():
                sleep(0.1)
                continue
            else:
                item_to_process = self.queue.get()
            analyzer_config, test_item_id, searched_res = item_to_process
            cnt_items_to_process += 1

            boosting_data_gatherer = boosting_featurizer.BoostingFeaturizer(
                searched_res,
                self.get_config_for_boosting(analyzer_config),
                feature_ids=self.boosting_decision_maker.get_feature_ids(),
                weighted_log_similarity_calculator=self.weighted_log_similarity_calculator)
            feature_data, issue_type_names = boosting_data_gatherer.gather_features_info()

            if feature_data:

                predicted_labels, predicted_labels_probability =\
                    self.boosting_decision_maker.predict(feature_data)
                scores_by_issue_type = boosting_data_gatherer.scores_by_issue_type

                for i in range(len(issue_type_names)):
                    logger.debug("Most relevant item with issue type %s has id %s",
                                 issue_type_names[i],
                                 boosting_data_gatherer.
                                 scores_by_issue_type[issue_type_names[i]]["mrHit"]["_id"])
                    logger.debug("Most relevant item with issue type %s with info %s",
                                 issue_type_names[i],
                                 boosting_data_gatherer.
                                 scores_by_issue_type[issue_type_names[i]]["mrHit"]["_source"])
                    logger.debug("Issue type %s has label %d and probability %.3f for features %s",
                                 issue_type_names[i],
                                 predicted_labels[i],
                                 predicted_labels_probability[i][1],
                                 feature_data[i])

                predicted_issue_type = self.choose_issue_type(predicted_labels,
                                                              predicted_labels_probability,
                                                              issue_type_names,
                                                              boosting_data_gatherer)

                if predicted_issue_type:
                    chosen_type = scores_by_issue_type[predicted_issue_type]
                    relevant_item = chosen_type["mrHit"]["_source"]["test_item"]
                    analysis_result = AnalysisResult(testItem=test_item_id,
                                                     issueType=predicted_issue_type,
                                                     relevantItem=relevant_item)
                    results.append(analysis_result)
                    logger.debug(analysis_result)
                else:
                    logger.debug("Test item %s has no relevant items", test_item_id)
            else:
                logger.debug("There are no results for test item %s", test_item_id)
        logger.info("Processed %d test items. It took %.2f sec.", cnt_items_to_process, time() - t_start)
        logger.info("Finished analysis for %d launches with %d results.", len(launches), len(results))
        return results

    def get_config_for_boosting_suggests(self, analyzerConfig):
        return {
            "max_query_terms": self.search_cfg["MaxQueryTerms"],
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg["MinWordLength"],
            "filter_min_should_match": [],
            "filter_min_should_match_any": [
                "detected_message_extended",
                "detected_message_without_params_extended"] if analyzerConfig.numberOfLogLines == -1 else [
                "message_extended", "message_without_params_extended"],
            "number_of_log_lines": analyzerConfig.numberOfLogLines}

    def query_es_for_suggested_items(self, test_item_info, logs):
        full_results = []

        for log in logs:
            message = log["_source"]["message"].strip()
            merged_small_logs = log["_source"]["merged_small_logs"].strip()
            if log["_source"]["log_level"] < ERROR_LOGGING_LEVEL or\
                    (not message and not merged_small_logs):
                continue

            query = self.es_query_builder.build_suggest_query(
                test_item_info, log,
                message_field="message_extended",
                det_mes_field="detected_message_extended",
                stacktrace_field="stacktrace_extended")
            es_res = self.es_client.search(index=str(test_item_info.project), body=query)
            full_results.append((log, es_res))

            query = self.es_query_builder.build_suggest_query(
                test_item_info, log,
                message_field="message_without_params_extended",
                det_mes_field="detected_message_without_params_extended",
                stacktrace_field="stacktrace_extended")
            es_res = self.es_client.search(index=str(test_item_info.project), body=query)
            full_results.append((log, es_res))
        return full_results

    def deduplicate_results(self, gathered_results, scores_by_test_items, test_item_ids):
        _similarity_calculator = similarity_calculator.SimilarityCalculator(
            {
                "max_query_terms": self.search_cfg["MaxQueryTerms"],
                "min_word_length": self.search_cfg["MinWordLength"],
                "min_should_match": "98%",
                "number_of_log_lines": -1
            },
            weighted_similarity_calculator=self.weighted_log_similarity_calculator)
        all_pairs_to_check = []
        for i in range(len(gathered_results)):
            for j in range(i + 1, len(gathered_results)):
                test_item_id_first = test_item_ids[gathered_results[i][0]]
                test_item_id_second = test_item_ids[gathered_results[j][0]]
                issue_type1 = scores_by_test_items[test_item_id_first]["mrHit"]["_source"]["issue_type"]
                issue_type2 = scores_by_test_items[test_item_id_second]["mrHit"]["_source"]["issue_type"]
                if issue_type1 != issue_type2:
                    continue
                items_to_compare = {"hits": {"hits": [scores_by_test_items[test_item_id_first]["mrHit"]]}}
                all_pairs_to_check.append((scores_by_test_items[test_item_id_second]["mrHit"],
                                           items_to_compare))
        _similarity_calculator.find_similarity(
            all_pairs_to_check,
            ["detected_message_with_numbers", "stacktrace", "merged_small_logs"])
        filtered_results = []
        deleted_indices = set()
        for i in range(len(gathered_results)):
            if i in deleted_indices:
                continue
            for j in range(i + 1, len(gathered_results)):
                test_item_id_first = test_item_ids[gathered_results[i][0]]
                test_item_id_second = test_item_ids[gathered_results[j][0]]
                group_id = (scores_by_test_items[test_item_id_first]["mrHit"]["_id"],
                            scores_by_test_items[test_item_id_second]["mrHit"]["_id"])
                if group_id not in _similarity_calculator.similarity_dict["detected_message_with_numbers"]:
                    continue
                det_message = _similarity_calculator.similarity_dict["detected_message_with_numbers"]
                detected_message_sim = det_message[group_id]
                stacktrace_sim = _similarity_calculator.similarity_dict["stacktrace"][group_id]
                merged_logs_sim = _similarity_calculator.similarity_dict["merged_small_logs"][group_id]
                if detected_message_sim["similarity"] >= 0.98 and\
                        stacktrace_sim["similarity"] >= 0.98 and merged_logs_sim["similarity"] >= 0.98:
                    deleted_indices.add(j)
            filtered_results.append(gathered_results[i])
        return filtered_results

    def sort_results(self, scores_by_test_items, test_item_ids, predicted_labels_probability):
        gathered_results = []
        for idx, prob in enumerate(predicted_labels_probability):
            test_item_id = test_item_ids[idx]
            gathered_results.append(
                (idx,
                 round(prob[1], 2),
                 scores_by_test_items[test_item_id]["mrHit"]["_source"]["start_time"]))

        gathered_results = sorted(gathered_results, key=lambda x: (x[1], x[2]), reverse=True)
        return self.deduplicate_results(gathered_results, scores_by_test_items, test_item_ids)

    @utils.ignore_warnings
    def suggest_items(self, test_item_info, num_items=5):
        logger.info("Started suggesting test items")
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        if not self.index_exists(str(test_item_info.project)):
            logger.info("Project %d doesn't exist", test_item_info.project)
            logger.info("Finished suggesting for test item with 0 results.")
            return []
        t_start = time()
        results = []
        unique_logs = utils.leave_only_unique_logs(test_item_info.logs)
        prepared_logs = [self._prepare_log_for_suggests(test_item_info, log)
                         for log in unique_logs if log.logLevel >= ERROR_LOGGING_LEVEL]
        logs = LogMerger.decompose_logs_merged_and_without_duplicates(prepared_logs)
        searched_res = self.query_es_for_suggested_items(test_item_info, logs)

        _boosting_data_gatherer = SuggestBoostingFeaturizer(
            searched_res,
            self.get_config_for_boosting_suggests(test_item_info.analyzerConfig),
            feature_ids=self.suggest_decision_maker.get_feature_ids(),
            weighted_log_similarity_calculator=self.weighted_log_similarity_calculator)
        feature_data, test_item_ids = _boosting_data_gatherer.gather_features_info()
        scores_by_test_items = _boosting_data_gatherer.scores_by_issue_type

        if feature_data:
            predicted_labels, predicted_labels_probability = self.suggest_decision_maker.predict(feature_data)
            sorted_results = self.sort_results(
                scores_by_test_items, test_item_ids, predicted_labels_probability)

            logger.debug("Found %d results for test items ", len(sorted_results))
            for idx, prob, _ in sorted_results:
                test_item_id = test_item_ids[idx]
                issue_type = scores_by_test_items[test_item_id]["mrHit"]["_source"]["issue_type"]
                logger.debug("Test item id %d with issue type %s has probability %.2f",
                             test_item_id, issue_type, prob)

            for idx, prob, _ in sorted_results[:num_items]:
                if prob >= self.suggest_threshold:
                    test_item_id = test_item_ids[idx]
                    issue_type = scores_by_test_items[test_item_id]["mrHit"]["_source"]["issue_type"]
                    relevant_log_id = utils.extract_real_id(
                        scores_by_test_items[test_item_id]["mrHit"]["_id"])
                    analysis_result = SuggestAnalysisResult(testItem=test_item_info.testItemId,
                                                            issueType=issue_type,
                                                            relevantItem=test_item_id,
                                                            relevantLogId=relevant_log_id,
                                                            matchScore=round(prob * 100, 2))
                    results.append(analysis_result)
                    logger.debug(analysis_result)
        else:
            logger.debug("There are no results for test item %s", test_item_info.testItemId)
        logger.info("Processed the test item. It took %.2f sec.", time() - t_start)
        logger.info("Finished suggesting for test item with %d results.", len(results))
        return results
