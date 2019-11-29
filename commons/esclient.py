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
        logger.info("Creating '%s' Elasticsearch index" % str(index_name))

        try:
            response = self.es.indices.create(index=str(index_name), body={
                'settings': DEFAULT_INDEX_SETTINGS,
                'mappings': DEFAULT_MAPPING_SETTINGS,
            })

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

            logger.info("Deleted index %s"%str(index_name))
            return commons.launch_objects.Response(**resp)
        except Exception as err:
            exc_info = sys.exc_info()
            error_info = ''.join(traceback.format_exception(*exc_info))
            logger.error("Not found %s"%str(index_name))
            logger.error(err)
            return commons.launch_objects.Response(**{"acknowledged": False, "error": error_info})

    def create_index_if_not_exists(self, index_name):
        if not self.index_exists(index_name):
            return self.create_index(index_name)

    def index_logs(self, launches):
        logger.info("Indexing logs for %d launches"% len(launches))
        bodies = []
        for launch in launches:
            self.create_index_if_not_exists(str(launch.project))

            for test_item in launch.testItems:
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
                            "message":          message,
                    }}

                    bodies.append(body)
        return self.bulk_index(bodies)

    def bulk_index(self, bodies):
        logger.info('Indexing %d logs...' % len(bodies))
        try:
            success_count, errors = elasticsearch.helpers.bulk(self.es, bodies, chunk_size=1000, request_timeout=30, refresh=True)

            logger.error("Processed %d logs"%success_count)
            if len(errors) > 0:
                logger.info("Occured errors ", errors)
            return commons.launch_objects.BulkResponse(took = success_count, errors = len(errors) > 0) # check how to set status and items
        except Exception as err:
            logger.error("Error in bulk")
            logger.error(err)
            return commons.launch_objects.BulkResponse(took = 0, errors = True) # check how to set status and items

    def delete_logs(self, clean_index):
        logger.info("Delete logs {}".format(clean_index.ids))

        bodies = []
        for _id in clean_index.ids:
            bodies.append({
                "_op_type": "delete",
                "_id":      _id,
                "_index":   clean_index.project,
            })
        return self.bulk_index(bodies)

    def build_search_query(self, searchReq, message):
        return {"query": {
                    "bool": {
                        "must_not":{
                            "term" : { "test_item": {"value": searchReq.itemId, "boost":1.0}}
                        },
                        "must": [
                            {"range":{"log_level":{"gte": ERROR_LOGGING_LEVEL}}},
                            {"exists":{"field":"issue_type"}},
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
        for message in searchReq.logMessages:
            sanitizedMsg = utils.sanitize_text(utils.first_lines(message, searchReq.logLines))
            query = self.build_search_query(searchReq, sanitizedMsg)
            res = self.es.search(index=str(searchReq.projectId), body = query)

            for rs in res["hits"]["hits"]:
                try:
                    logId = int(rs["_id"])
                except:
                    logger.error("Id %s is not integer"%rs["_id"])
                    logId = rs["_id"]
                keys.add(logId)
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

    def analyze_logs(self, launches):
        logger.info("Started analysis for %d launches"%len(launches))
        results = []

        for launch in launches:
            for test_item in launch.testItems:
                issue_types = {}

                for log in test_item.logs:
                    if log.logLevel < ERROR_LOGGING_LEVEL or log.message.strip() == "":
                        continue

                    message = utils.sanitize_text(utils.first_lines(log.message, launch.analyzerConfig.numberOfLogLines))

                    query = self.build_analyze_query(launch, test_item.uniqueId, message)

                    res = self.es.search(index=str(launch.project), body = query)

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