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
import copy
import requests
import elasticsearch
import elasticsearch.helpers
from scipy import spatial
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import commons.launch_objects
from commons.launch_objects import AnalysisResult
import utils.utils as utils
from boosting_decision_making import boosting_featurizer
from time import time
from datetime import datetime
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

ERROR_LOGGING_LEVEL = 40000

DEFAULT_INDEX_SETTINGS = {
    'number_of_shards': 1,
    'analysis': {
        "analyzer": {
            "standard_english_analyzer": {
                "type": "standard",
                "stopwords": "_english_",
            }
        }
    }
}

DEFAULT_MAPPING_SETTINGS = {
    "properties": {
        "test_item": {
            "type": "keyword",
        },
        "issue_type": {
            "type": "keyword",
        },
        "message": {
            "type":     "text",
            "analyzer": "standard_english_analyzer"
        },
        "merged_small_logs": {
            "type":     "text",
            "analyzer": "standard_english_analyzer"
        },
        "detected_message": {
            "type":     "text",
            "analyzer": "standard_english_analyzer",
        },
        "detected_message_with_numbers": {
            "type":     "text",
            "index":    False,
        },
        "only_numbers": {
            "type":     "text",
            "analyzer": "standard_english_analyzer",
        },
        "stacktrace": {
            "type":     "text",
            "analyzer": "standard_english_analyzer",
        },
        "original_message_lines": {
            "type": "integer",
        },
        "original_message_words_number": {
            "type": "integer",
        },
        "log_level": {
            "type": "integer",
        },
        "test_case_hash": {
            "type": "integer",
        },
        "launch_name": {
            "type": "keyword",
        },
        "unique_id": {
            "type": "keyword",
        },
        "is_auto_analyzed": {
            "type": "keyword",
        },
        "is_merged": {
            "type": "boolean"
        },
        "start_time": {
            "type":   "date",
            "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd"
        }
    }
}


def calculate_features(params):
    es_results_to_process, feature_ids, idx, config, parallel = params
    features_gathered = []
    for analyzer_config, test_item, searched_res in es_results_to_process:
        min_should_match = analyzer_config.minShouldMatch / 100 if analyzer_config.minShouldMatch > 0 else\
            int(re.search(r"\d+", config["min_should_match"]).group(0)) / 100
        _boosting_data_gatherer = boosting_featurizer.BoostingFeaturizer(
            searched_res,
            {
                "max_query_terms": config["max_query_terms"],
                "min_should_match": min_should_match,
                "min_word_length": config["min_word_length"],
                "filter_min_should_match": utils.choose_fields_to_filter(config["filter_min_should_match"],
                                                                         analyzer_config.numberOfLogLines),
            },
            feature_ids=feature_ids)
        part_train_calc_data, issue_type_names = _boosting_data_gatherer.gather_features_info()
        features_gathered.append((part_train_calc_data, issue_type_names, _boosting_data_gatherer))
    return (idx, features_gathered) if parallel else [(idx, features_gathered)]


logger = logging.getLogger("analyzerApp.esclient")


class EsClient:
    """Elasticsearch client implementation"""
    def __init__(self, host="http://localhost:9200", search_cfg={}):
        self.host = host
        self.search_cfg = search_cfg
        self.es_client = elasticsearch.Elasticsearch([host], timeout=30,
                                                     max_retries=5, retry_on_timeout=True)
        self.boosting_decision_maker = None

    def set_boosting_decision_maker(self, boosting_decision_maker):
        self.boosting_decision_maker = boosting_decision_maker

    @staticmethod
    def compress(text):
        return " ".join(utils.split_words(text, only_unique=True))

    def create_index(self, index_name):
        """Create index in elasticsearch"""
        logger.debug("Creating '%s' Elasticsearch index", str(index_name))
        logger.info("ES Url %s", self.host)
        try:
            response = self.es_client.indices.create(index=str(index_name), body={
                'settings': DEFAULT_INDEX_SETTINGS,
                'mappings': DEFAULT_MAPPING_SETTINGS,
            })
            logger.debug("Created '%s' Elasticsearch index", str(index_name))
            return commons.launch_objects.Response(**response)
        except Exception as err:
            logger.error("Couldn't create index")
            logger.error("ES Url %s", self.host)
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
            logger.info("ES Url %s", self.host)
            logger.debug("Deleted index %s", str(index_name))
            return 1
        except Exception as err:
            logger.error("Not found %s for deleting", str(index_name))
            logger.error("ES Url %s", self.host)
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
        logger.info("ES Url %s", self.host)
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

                    if log.logLevel < ERROR_LOGGING_LEVEL or log.message.strip() == "":
                        continue

                    bodies.append(self._prepare_log(launch, test_item, log))
                    logs_added = True
                if logs_added:
                    test_item_ids.append(str(test_item.testItemId))
        result = self._bulk_index(bodies)
        result = self._merge_logs(test_item_ids, project)
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
        return message

    def _prepare_log(self, launch, test_item, log):
        cleaned_message = self.clean_message(log.message)

        message = utils.leave_only_unique_lines(utils.sanitize_text(
            utils.first_lines(cleaned_message,
                              launch.analyzerConfig.numberOfLogLines)))

        detected_message, stacktrace = utils.detect_log_description_and_stacktrace(
            cleaned_message,
            default_log_number=1)

        detected_message_with_numbers = utils.remove_starting_datetime(detected_message)
        detected_message = utils.sanitize_text(detected_message)
        stacktrace = utils.sanitize_text(stacktrace)

        stacktrace = utils.leave_only_unique_lines(stacktrace)
        detected_message = utils.leave_only_unique_lines(detected_message)
        detected_message_with_numbers = utils.leave_only_unique_lines(detected_message_with_numbers)

        detected_message_only_numbers = utils.find_only_numbers(detected_message_with_numbers)

        return {
            "_id":    log.logId,
            "_index": launch.project,
            "_source": {
                "launch_id":        launch.launchId,
                "launch_name":      launch.launchName,
                "test_item":        test_item.testItemId,
                "unique_id":        test_item.uniqueId,
                "test_case_hash":   test_item.testCaseHash,
                "is_auto_analyzed": test_item.isAutoAnalyzed,
                "issue_type":       test_item.issueType,
                "log_level":        log.logLevel,
                "original_message_lines": utils.calculate_line_number(cleaned_message),
                "original_message_words_number": len(
                    utils.split_words(cleaned_message, split_urls=False)),
                "message":          message,
                "is_merged":        False,
                "start_time":       datetime(*test_item.startTime[:6]).strftime("%Y-%m-%d %H:%M:%S"),
                "merged_small_logs":  "",
                "detected_message": detected_message,
                "detected_message_with_numbers": detected_message_with_numbers,
                "stacktrace":                    stacktrace,
                "only_numbers":                  detected_message_only_numbers}}

    def _merge_logs(self, test_item_ids, project):
        bodies = []
        batch_size = 1000
        self._delete_merged_logs(test_item_ids, project)
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            test_items = test_item_ids[i * batch_size: (i + 1) * batch_size]
            if len(test_items) == 0:
                continue
            test_items_dict = {}
            for r in elasticsearch.helpers.scan(self.es_client,
                                                query=EsClient.get_test_item_query(test_items, False),
                                                index=project):
                test_item_id = r["_source"]["test_item"]
                if test_item_id not in test_items_dict:
                    test_items_dict[test_item_id] = []
                test_items_dict[test_item_id].append(r)
            for test_item_id in test_items_dict:
                merged_logs = EsClient.decompose_logs_merged_and_without_duplicates(
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
            if len(test_item_ids) == 0:
                continue
            for log in elasticsearch.helpers.scan(self.es_client,
                                                  query=EsClient.get_test_item_query(test_item_ids, True),
                                                  index=project):
                bodies.append({
                    "_op_type": "delete",
                    "_id": log["_id"],
                    "_index": project
                })
        if len(bodies) > 0:
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
            if log["_source"]["message"].strip() == "":
                continue
            log_level = log["_source"]["log_level"]

            if log["_id"] in log_level_ids_to_add[log_level]:
                merged_small_logs = EsClient.compress(
                    log_level_messages[log["_source"]["log_level"]])
                new_logs.append(EsClient.prepare_new_log(
                    log, log["_id"], False, merged_small_logs))

        for log_level in log_level_messages:

            if len(log_level_ids_to_add[log_level]) == 0 and\
               log_level_messages[log_level].strip() != "":
                log = log_level_ids_merged[log_level]
                new_logs.append(EsClient.prepare_new_log(
                    log, str(log["_id"]) + "_m", True,
                    EsClient.compress(log_level_messages[log_level]),
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
            if log["_source"]["message"].strip() == "":
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
        if len(bodies) == 0:
            return commons.launch_objects.BulkResponse(took=0, errors=False)
        logger.debug("Indexing %d logs...", len(bodies))
        try:
            success_count, errors = elasticsearch.helpers.bulk(self.es_client,
                                                               bodies,
                                                               chunk_size=1000,
                                                               request_timeout=30,
                                                               refresh=refresh)

            logger.debug("Processed %d logs", success_count)
            if len(errors) > 0:
                logger.debug("Occured errors %s", errors)
            return commons.launch_objects.BulkResponse(took=success_count, errors=len(errors) > 0)
        except Exception as err:
            logger.error("Error in bulk")
            logger.error("ES Url %s", self.host)
            logger.error(err)
            return commons.launch_objects.BulkResponse(took=0, errors=True)

    def delete_logs(self, clean_index):
        """Delete logs from elasticsearch"""
        logger.info("Delete logs %s for the project %s",
                    clean_index.ids, clean_index.project)
        logger.info("ES Url %s", self.host)
        t_start = time()
        if not self.index_exists(clean_index.project):
            return 0
        test_item_ids = set()
        try:
            for res in elasticsearch.helpers.scan(self.es_client,
                                                  query=EsClient.build_search_test_item_ids_query(
                                                      clean_index.ids),
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

    @staticmethod
    def build_search_test_item_ids_query(log_ids):
        """Build search test item ids query"""
        return {"size": 10000,
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"log_level": {"gte": ERROR_LOGGING_LEVEL}}},
                            {"exists": {"field": "issue_type"}},
                            {"term": {"is_merged": False}},
                            {"terms": {"_id": [str(log_id) for log_id in log_ids]}},
                        ]
                    }
                }, }

    def build_search_query(self, search_req, message):
        """Build search query"""
        return {
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}},
                    ],
                    "must_not": {
                        "term": {"test_item": {"value": search_req.itemId, "boost": 1.0}}
                    },
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"wildcard": {"issue_type": "TI*"}},
                                    {"wildcard": {"issue_type": "ti*"}},
                                ]
                            }
                        },
                        {"terms": {"launch_id": search_req.filteredLaunchIds}},
                        EsClient.
                        build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                   self.search_cfg["SearchLogsMinShouldMatch"],
                                                   message),
                    ],
                    "should": [
                        {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                    ]}}}

    def search_logs(self, search_req):
        """Get all logs similar to given logs"""
        similar_log_ids = set()
        logger.info("Started searching by request %s", search_req.json())
        logger.info("ES Url %s", self.host)
        t_start = time()
        if not self.index_exists(str(search_req.projectId)):
            return []
        searched_logs = set()
        for message in search_req.logMessages:
            if message.strip() == "":
                continue
            cleaned_message = self.clean_message(message)
            sanitized_msg = utils.leave_only_unique_lines(utils.sanitize_text(
                utils.first_lines(cleaned_message, search_req.logLines)))

            msg_words = " ".join(utils.split_words(sanitized_msg))
            if msg_words in searched_logs:
                continue
            searched_logs.add(msg_words)
            query = self.build_search_query(search_req, sanitized_msg)
            res = self.es_client.search(index=str(search_req.projectId), body=query)
            similar_log_ids = similar_log_ids.union(
                self.find_similar_logs_by_cosine_similarity(msg_words, message, res))

        logger.info("Finished searching by request %s with %d results. It took %.2f sec.",
                    search_req.json(), len(similar_log_ids), time() - t_start)
        return list(similar_log_ids)

    def find_similar_logs_by_cosine_similarity(self, msg_words, message, res):
        similar_log_ids = set()
        messages_by_ids = {}
        message_index_id = 1
        all_messages = [msg_words]

        for result in res["hits"]["hits"]:
            try:
                log_id = int(re.search(r"\d+", result["_id"]).group(0))
                if log_id not in messages_by_ids:
                    log_query_words = " ".join(utils.split_words(result["_source"]["message"]))
                    all_messages.append(log_query_words)
                    messages_by_ids[log_id] = message_index_id
                    message_index_id += 1
            except Exception as err:
                logger.error("Id %s is not integer", result["_id"])
                logger.error(err)
        if len(all_messages) > 1:
            vectorizer = CountVectorizer(binary=True,
                                         analyzer="word",
                                         token_pattern="[^ ]+")
            count_vector_matrix = vectorizer.fit_transform(all_messages)
            for log_id in messages_by_ids:
                similarity_percent = round(1 - spatial.distance.cosine(
                    np.asarray(count_vector_matrix[0].toarray()),
                    np.asarray(count_vector_matrix[messages_by_ids[log_id]].toarray())), 3)
                logger.debug("Log with id %s has %.3f similarity with the log '%s'",
                             log_id, similarity_percent, message)
                if similarity_percent >= self.search_cfg["SearchLogsMinSimilarity"]:
                    similar_log_ids.add(log_id)
        return similar_log_ids

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
            if launch.analyzerConfig.minShouldMatch > 0\
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
        if log["_source"]["message"].strip() != "":
            log_lines = launch.analyzerConfig.numberOfLogLines
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                min_should_match,
                                                log["_source"]["message"],
                                                field_name="message",
                                                boost=(4.0 if log_lines != -1 else 2.0)))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                "80%",
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=0.5))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                "80%",
                                                log["_source"]["detected_message"],
                                                field_name="detected_message",
                                                boost=(4.0 if log_lines == -1 else 2.0)))
            if log_lines != -1:
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query(self.search_cfg["MaxQueryTerms"],
                                                    "60%",
                                                    log["_source"]["stacktrace"],
                                                    field_name="stacktrace", boost=1.0))
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
            if len(part_batch) == 0:
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
            if len(partial_result) == 0:
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
                process_results = pool.map(
                    calculate_features,
                    [(res, self.boosting_decision_maker.get_feature_ids(), i, config, True)
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
                process_results.extend(
                    calculate_features(
                        (res, self.boosting_decision_maker.get_feature_ids(), i, config, False)))
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

    @utils.ignore_warnings
    def analyze_logs(self, launches):
        logger.info("Started analysis for %d launches", len(launches))
        logger.info("ES Url %s", self.host)
        results = []

        t_start = time()
        es_results = self.get_bulk_search_results(launches)
        logger.debug("Searched ES for all test items for %.2f sec.", time() - t_start)

        t = time()
        es_results_to_process, process_results = self.prepare_features_for_analysis(es_results)
        logger.debug("Prepared features for all test items for %.2f sec.", time() - t)

        for idx, features_gathered in process_results:
            for i in range(len(features_gathered)):
                analyzer_config, test_item_id, searched_res = es_results_to_process[idx][i]
                feature_data, issue_type_names, boosting_data_gatherer = features_gathered[i]

                if len(feature_data) > 0:

                    predicted_labels, predicted_labels_probability =\
                        self.boosting_decision_maker.predict(feature_data)

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

                    if predicted_issue_type != "":
                        chosen_type =\
                            boosting_data_gatherer.scores_by_issue_type[predicted_issue_type]
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
        logger.info("Processed %d test items. It took %.2f sec.", len(es_results), time() - t_start)
        logger.info("Finished analysis for %d launches with %d results.", len(launches), len(results))
        return results
