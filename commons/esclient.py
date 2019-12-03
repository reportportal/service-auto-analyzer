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

import requests
import json
import traceback
import re
import json
import os
import elasticsearch
import elasticsearch.helpers
import commons.launch_objects
import utils.utils as utils
import logging
import sys
import copy

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
        "log_level": {
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
    }
}

logger = logging.getLogger("analyzerApp.esclient")

class EsClient:

    def __init__(self, host = "http://localhost:9200", search_cfg = {}):
        self.host = host
        self.search_cfg = search_cfg
        self.es = elasticsearch.Elasticsearch([host],timeout=30,\
                                               max_retries=5, retry_on_timeout=True)

    def create_index(self, index_name):
        logger.debug("Creating '%s' Elasticsearch index" % str(index_name))

        try:
            response = self.es.indices.create(index=str(index_name), body={
                'settings': DEFAULT_INDEX_SETTINGS,
                'mappings': DEFAULT_MAPPING_SETTINGS,
            })
            logger.debug("Created '%s' Elasticsearch index" % str(index_name))
            return commons.launch_objects.Response(**response)
        except Exception as err:
            logger.error("Couldn't create index")
            logger.error(err)
            return commons.launch_objects.Response()

    def send_request(self, url, method):
        try:
            rs = requests.get(url) if method == "GET" else {}
            data = rs._content.decode("utf-8")
            content = json.loads(data, strict=False)
            return content
        except Exception as err:
            logger.error("Error with loading url: %s"%url)
            logger.error(err)
        return []

    def is_healthy(self):
        try:
            url = utils.build_url(self.host, ["_cluster/health"])
            res = self.send_request(url, "GET")
            return res["status"] in ["green","yellow"]
        except Exception as err:
            logger.error("Elasticsearch is not healthy")
            return False

    def list_indices(self):
        url = utils.build_url(self.host, ["_cat", "indices?format=json"])
        res = self.send_request(url, "GET")
        return res

    def index_exists(self, index_name):
        try:
            index = self.es.indices.get(index=str(index_name))
            return True
        except Exception as err:
            logger.error("Index %s was not found"%str(index_name))
            logger.error(err)
            return False

    def delete_index(self, index_name):
        try:
            resp = self.es.indices.delete(index=str(index_name))

            logger.debug("Deleted index %s"%str(index_name))
            return commons.launch_objects.Response(**resp)
        except Exception as err:
            exc_info = sys.exc_info()
            error_info = ''.join(traceback.format_exception(*exc_info))
            logger.error("Not found %s for deleting"%str(index_name))
            logger.error(err)
            return commons.launch_objects.Response(**{"acknowledged": False, "error": error_info})

    def create_index_if_not_exists(self, index_name):
        if not self.index_exists(index_name):
            return self.create_index(index_name)

    def index_logs(self, launches):
        logger.debug("Indexing logs for %d launches"% len(launches))
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

                    message = utils.sanitize_text(utils.first_lines(log.message, launch.analyzerConfig.numberOfLogLines))

                    body = {
                        "_id":    log.logId,
                        "_index": launch.project,
                        "_source": {
                            "launch_id":        launch.launchId,
                            "launch_name":      launch.launchName,
                            "test_item":        test_item.testItemId,
                            "unique_id":        test_item.uniqueId,
                            "is_auto_analyzed": test_item.isAutoAnalyzed,
                            "issue_type":       test_item.issueType,
                            "log_level":        log.logLevel,
                            "original_message": log.message,
                            "message":          message,
                            "is_merged":        False
                    }}

                    bodies.append(body)
                    logs_added = True
                if logs_added:
                    test_item_ids.append(str(test_item.testItemId))
        result = self.bulk_index(bodies)
        result = self.merge_logs(test_item_ids, project)
        logger.debug("Finished indexing logs for %d launches"% len(launches))
        return result

    def merge_logs(self, test_item_ids, project):
        bodies = []
        self.delete_merged_logs(test_item_ids, project)
        for test_item_id in test_item_ids:
            res = self.es.search(index = project, body = self.get_test_item_query(test_item_id, False))
            merged_logs = self.decompose_logs_merged_and_without_duplicates(res["hits"]["hits"])
            bodies.extend(merged_logs)
        return self.bulk_index(bodies)

    def delete_merged_logs(self, test_items_to_delete, project):
        logger.debug("Delete merged logs for %d test items"% len(test_items_to_delete))
        bodies = []
        for test_item_id  in test_items_to_delete:
            res = self.es.search(index=project, body = self.get_test_item_query(test_item_id, True))
            for log in res["hits"]["hits"]:
                bodies.append({
                    "_op_type":"delete",
                    "_id":log["_id"],
                    "_index":project
                })
        if len(bodies) > 0:
            self.bulk_index(bodies)

    def get_test_item_query(self, test_item_id, is_merged):
        return {"size":10000,
                "query": {
                    "bool": {
                        "must":[
                            {"term" : { "test_item": test_item_id}},
                            {"term" : { "is_merged": is_merged}}
                        ]
                    }
                }}

    def decompose_logs_merged_and_without_duplicates(self, logs):
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

            if utils.calculate_line_number(log["_source"]["original_message"]) <= 2:
                if log_level not in log_level_ids_merged:
                    log_level_ids_merged[log_level] = log
                message = log["_source"]["message"]
                normalized_msg = " ".join(message.strip().lower().split())
                if normalized_msg not in logs_unique_log_level[log_level]:
                    logs_unique_log_level[log_level].add(normalized_msg)
                    log_level_messages[log_level] = log_level_messages[log_level] + message + "\r\n"
            else:
                log_level_ids_to_add[log_level].append(log["_id"])

        new_logs = []
        for log in logs:
            if log["_source"]["message"].strip() == "":
                continue
            log_level = log["_source"]["log_level"]

            if log["_id"] in log_level_ids_to_add[log_level]:
                normalized_message = log["_source"]["message"]

                if log_level_messages[log_level].strip() != "":
                    new_logs.append(self.prepare_new_log(log, str(log["_id"]) + "_m",
                        normalized_message + "\r\n" + log_level_messages[log["_source"]["log_level"]]))
                new_logs.append(self.prepare_new_log(log, str(log["_id"]) + "_big", normalized_message))

        for log_level in log_level_messages:

            if len(log_level_ids_to_add[log_level]) == 0:
                log = log_level_ids_merged[log_level]
                new_logs.append(self.prepare_new_log(log, str(log["_id"]) + "_m", log_level_messages[log_level]))
        return new_logs

    def prepare_new_log(self, old_log, new_id, message):
        merged_log = copy.deepcopy(old_log)
        merged_log["_source"]["is_merged"] = True
        merged_log["_id"] = new_id
        merged_log["_source"]["message"] = message
        return merged_log

    def bulk_index(self, bodies):
        logger.debug('Indexing %d logs...' % len(bodies))
        try:
            success_count, errors = elasticsearch.helpers.bulk(self.es, bodies, chunk_size=1000, request_timeout=30, refresh=True)

            logger.debug("Processed %d logs"%success_count)
            if len(errors) > 0:
                logger.debug("Occured errors ", errors)
            return commons.launch_objects.BulkResponse(took = success_count, errors = len(errors) > 0) # check how to set status and items
        except Exception as err:
            logger.error("Error in bulk")
            logger.error(err)
            return commons.launch_objects.BulkResponse(took = 0, errors = True) # check how to set status and items

    def delete_logs(self, clean_index):
        logger.debug("Delete logs {} for the project {}".format(clean_index.ids, clean_index.project))
        test_item_ids = set()
        try:
            all_logs = self.es.search(index= clean_index.project, body = self.build_search_test_item_ids_query(clean_index.ids))
            for res in all_logs["hits"]["hits"]:
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
        result = self.bulk_index(bodies)
        result_merge = self.merge_logs(list(test_item_ids), clean_index.project)
        logger.debug("Finished deleting logs {} for the project {}".format(clean_index.ids, clean_index.project))
        return result

    def build_search_test_item_ids_query(self, log_ids):
        return {"size": 10000,
                "query": {
                    "bool": {
                        "must": [
                            {"range":{"log_level":{"gte": ERROR_LOGGING_LEVEL}}},
                            {"exists":{"field":"issue_type"}},
                            {"term":{"is_merged":False}},
                            {"terms": {"_id": log_ids}},
                        ]
                    }
                },
            }

    def build_search_query(self, searchReq, message):
        return {"query": {
                    "bool": {
                        "must_not":{
                            "term" : { "test_item": {"value": searchReq.itemId, "boost":1.0}}
                        },
                        "must": [
                            {"range":{"log_level":{"gte": ERROR_LOGGING_LEVEL}}},
                            {"exists":{"field":"issue_type"}},
                            {"term":{"is_merged":True}},
                            {
                                "bool": {
                                    "should": [
                                        {"wildcard":{"issue_type":"TI*"}},
                                        {"wildcard":{"issue_type":"ti*"}},
                                    ]
                                }
                            },
                            {"terms": {"launch_id": searchReq.filteredLaunchIds}},
                            self.build_more_like_this_query(1, 1, self.search_cfg["MaxQueryTerms"],
                                self.search_cfg["SearchLogsMinShouldMatch"], message),
                        ],
                        "should" : [
                            {"term": {"is_auto_analyzed": {"value":"false", "boost": 1.0}}},
                        ]
                    }
                }}

    def search_logs(self, searchReq):
        keys = set()
        logger.debug("Started searching by request %s"%searchReq.json())
        for message in searchReq.logMessages:
            sanitizedMsg = utils.sanitize_text(utils.first_lines(message, searchReq.logLines))
            query = self.build_search_query(searchReq, sanitizedMsg)
            res = self.es.search(index=str(searchReq.projectId), body = query)

            for rs in res["hits"]["hits"]:
                try:
                    logId = int(re.search("\d+",rs["_id"]).group(0))
                    keys.add(logId)
                except:
                    logger.error("Id %s is not integer"%rs["_id"])
        logger.debug("Finished searching by request %s with %d results"%(searchReq.json(), len(keys)))
        return list(keys)

    def build_more_like_this_query(self, minDocFreq, minTermFreq, maxQueryTerms, minShouldMatch, logMessage):
        return {"more_like_this":{
                    "fields":               ["message"],
                    "like":                 logMessage,
                    "min_doc_freq":         minDocFreq,
                    "min_term_freq":        minTermFreq,
                    "minimum_should_match": "5<"+minShouldMatch,
                    "max_query_terms":      maxQueryTerms,
                }}

    def build_analyze_query(self, launch, uniqueId, message, size = 10):
        minDocFreq = launch.analyzerConfig.minDocFreq if launch.analyzerConfig.minDocFreq > 0 else self.search_cfg["MinDocFreq"]
        minTermFreq = launch.analyzerConfig.minTermFreq if launch.analyzerConfig.minTermFreq > 0 else self.search_cfg["MinTermFreq"]
        minShouldMatch = "{}%".format(launch.analyzerConfig.minShouldMatch) if launch.analyzerConfig.minShouldMatch > 0\
                                                                            else self.search_cfg["MinShouldMatch"]

        query =  {  "size":size,
                    "query": {
                        "bool": {
                            "must_not": [
                                {"wildcard":{"issue_type":"TI*"}},
                                {"wildcard":{"issue_type":"ti*"}},
                            ], 
                            "must": [
                                {"range":{"log_level":{"gte": ERROR_LOGGING_LEVEL}}},
                                {"exists":{"field":"issue_type"}},
                                {"term":{"is_merged":True}},
                            ],
                            "should" : [
                                { "term": {"unique_id": {"value": uniqueId, "boost": abs(self.search_cfg["BoostUniqueID"])}}},
                                {"term": {"is_auto_analyzed": {"value":str(self.search_cfg["BoostAA"] < 0).lower(),
                                                               "boost": abs(self.search_cfg["BoostAA"])}}},
                            ]
                        }
                }}

        if launch.analyzerConfig.analyzerMode in ["LAUNCH_NAME"]:
            query["query"]["bool"]["must"].append({"term": {"launch_name": {"value": launch.launchName}}})
            query["query"]["bool"]["must"].append(self.build_more_like_this_query(minDocFreq, minTermFreq,
                                                                        self.search_cfg["MaxQueryTerms"], minShouldMatch, message))
        elif launch.analyzerConfig.analyzerMode in ["CURRENT_LAUNCH"]:
            query["query"]["bool"]["must"].append({"term": {"launch_id": {"value": launch.launchId}}})
            query["query"]["bool"]["must"].append(self.build_more_like_this_query(1, minTermFreq,
                                                                        self.search_cfg["MaxQueryTerms"], minShouldMatch, message))
        else:
            query["query"]["bool"]["should"].append({"term": {"launch_name": {"value": launch.launchName,
                                                                              "boost": abs(self.search_cfg["BoostLaunch"])}}})
            query["query"]["bool"]["must"].append(self.build_more_like_this_query(minDocFreq, minTermFreq,
                                                                        self.search_cfg["MaxQueryTerms"], minShouldMatch, message))
        return query

    def get_elasticsearch_results_for_test_items(self, launch, test_item):
        full_results = []
        prepared_logs = [{"_id":log.logId, 
                          "_source":{ 
                                "message":          utils.sanitize_text(utils.first_lines(log.message, 
                                                                                          launch.analyzerConfig.numberOfLogLines)),
                                "original_message": log.message, 
                                "log_level":        log.logLevel,
                         }} for log in test_item.logs]
        for log in self.decompose_logs_merged_and_without_duplicates(prepared_logs):

            if log["_source"]["log_level"] < ERROR_LOGGING_LEVEL and log["_source"]["message"].strip() != "":
                continue

            query = self.build_analyze_query(launch, test_item.uniqueId, log["_source"]["message"])

            res = self.es.search(index=str(launch.project), body = query)
            full_results.append((log["_source"]["message"], res))
        return full_results

    def analyze_logs(self, launches):
        logger.debug("Started analysis for %d launches"%len(launches))
        results = []

        for launch in launches:
            for test_item in launch.testItems:
                issue_types = {}

                for log_message, res in self.get_elasticsearch_results_for_test_items(launch, test_item):
                    issue_types = self.calculate_scores(res, 10, issue_types)

                predicted_issue_type = ""
                if len(issue_types) > 0:
                    max_val = 0.0
                    for key in issue_types:
                        if issue_types[key]["score"] > max_val:
                            max_val = issue_types[key]["score"]
                            predicted_issue_type = key

                if predicted_issue_type != "":
                    relevant_item = issue_types[predicted_issue_type]["mrHit"]["_source"]["test_item"]
                    results.append(commons.launch_objects.AnalysisResult(testItem = test_item.testItemId,
                                                                         issueType = predicted_issue_type,
                                                                         relevantItem = relevant_item))
        logger.debug("Finished analysis for %d launches with %d results"%(len(launches), len(results)))
        return results

    def calculate_scores(self, res, k, issue_types):
        if res["hits"]["total"]["value"] > 0:
            total_score = 0
            hits = res["hits"]["hits"][:k]

            for hit in hits:
                total_score += hit["_score"]

                if hit["_source"]["issue_type"] in issue_types:
                    issue_type_item = issue_types[hit["_source"]["issue_type"]]
                    if hit["_score"] > issue_type_item["mrHit"]["_score"]:
                        issue_types[hit["_source"]["issue_type"]]["mrHit"] = hit
                else:
                    issue_types[hit["_source"]["issue_type"]] = {"mrHit": hit, "score":0}

            for hit in hits:
                curr_score = hit["_score"] / total_score
                issue_types[hit["_source"]["issue_type"]]["score"] += curr_score
        return issue_types