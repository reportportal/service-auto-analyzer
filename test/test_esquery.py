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

import unittest
import sure # noqa
import logging

import commons.launch_objects as launch_objects
import commons.esclient as esclient
from utils import utils


class TestEsQuery(unittest.TestCase):
    """Tests building analyze query"""
    @utils.ignore_warnings
    def setUp(self):
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def test_build_analyze_query(self):
        """Tests building analyze query"""
        search_cfg = {
            "MinShouldMatch": "80%",
            "BoostAA":        10,
            "BoostLaunch":    5,
            "BoostUniqueID":  3,
            "MaxQueryTerms":  50,
        }
        error_logging_level = 40000

        es_client = esclient.EsClient(search_cfg=search_cfg)
        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "SearchModeAll"},
            "launchId": 123,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "message":          "hello world",
                "merged_small_logs":  "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "",
                "only_numbers": "1"}}
        query_from_esclient = es_client.build_analyze_query(launch, "unique", log)
        demo_query = TestEsQuery.build_demo_query(search_cfg, "Launch name",
                                                  "unique", log,
                                                  error_logging_level)

        query_from_esclient.should.equal(demo_query)

    @staticmethod
    @utils.ignore_warnings
    def build_demo_query(search_cfg, launch_name,
                         unique_id, log, error_logging_level):
        """Build demo analyze query"""
        return {
            "size": 10,
            "sort": ["_score",
                     {"start_time": "desc"}, ],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": error_logging_level}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": True}},
                    ],
                    "must_not": [
                        {"wildcard": {"issue_type": "TI*"}},
                        {"wildcard": {"issue_type": "ti*"}},
                    ],
                    "must": [
                        {"more_like_this": {
                            "fields":               ["message"],
                            "like":                 log["_source"]["message"],
                            "min_doc_freq":         1,
                            "min_term_freq":        1,
                            "minimum_should_match": "5<" + search_cfg["MinShouldMatch"],
                            "max_query_terms":      search_cfg["MaxQueryTerms"],
                            "boost":                2.0,
                        }, },
                    ],
                    "should": [
                        {"term": {
                            "unique_id": {
                                "value": unique_id,
                                "boost": abs(search_cfg["BoostUniqueID"]),
                            },
                        }},
                        {"term": {
                            "test_case_hash": {
                                "value": log["_source"]["test_case_hash"],
                                "boost": abs(search_cfg["BoostUniqueID"]),
                            },
                        }},
                        {"term": {
                            "is_auto_analyzed": {
                                "value": str(search_cfg["BoostAA"] < 0).lower(),
                                "boost": abs(search_cfg["BoostAA"]),
                            },
                        }},
                        {"term": {
                            "launch_name": {
                                "value": launch_name,
                                "boost": abs(search_cfg["BoostLaunch"]),
                            },
                        }},
                        {"more_like_this": {
                            "fields":               ["merged_small_logs"],
                            "like":                 log["_source"]["merged_small_logs"],
                            "min_doc_freq":         1,
                            "min_term_freq":        1,
                            "minimum_should_match": "5<80%",
                            "max_query_terms":      search_cfg["MaxQueryTerms"],
                            "boost":                0.5,
                        }},
                        {"more_like_this": {
                            "fields":               ["detected_message"],
                            "like":                 log["_source"]["detected_message"],
                            "min_doc_freq":         1,
                            "min_term_freq":        1,
                            "minimum_should_match": "5<80%",
                            "max_query_terms":      search_cfg["MaxQueryTerms"],
                            "boost":                4.0,
                        }},
                        {"more_like_this": {
                            "fields":               ["only_numbers"],
                            "like":                 log["_source"]["only_numbers"],
                            "min_doc_freq":         1,
                            "min_term_freq":        1,
                            "minimum_should_match": "1",
                            "max_query_terms":      search_cfg["MaxQueryTerms"],
                            "boost":                4.0,
                        }},
                    ],
                },
            },
        }

    @utils.ignore_warnings
    def test_build_search_query(self):
        """Tests building analyze query"""
        search_cfg = {
            "MinShouldMatch": "80%",
            "BoostAA":        10,
            "BoostLaunch":    5,
            "BoostUniqueID":  3,
            "MaxQueryTerms":  50,
            "SearchLogsMinShouldMatch": "90%",
            "SearchLogsMinSimilarity":  0.9,
        }
        error_logging_level = 40000

        es_client = esclient.EsClient(search_cfg=search_cfg)
        search_req = launch_objects.SearchLogs(**{
            "launchId": 1,
            "launchName": "launch 1",
            "itemId": 2,
            "projectId": 3,
            "filteredLaunchIds": [1, 2, 3],
            "logMessages": ["log message 1"],
            "logLines": -1})
        query_from_esclient = es_client.build_search_query(search_req, "log message 1")
        demo_query = TestEsQuery.build_demo_search_query(search_cfg, search_req, "log message 1",
                                                         error_logging_level)

        query_from_esclient.should.equal(demo_query)

    @staticmethod
    @utils.ignore_warnings
    def build_demo_search_query(search_cfg, search_req, message, error_logging_level):
        """Build search query"""
        return {"query": {
            "bool": {
                "filter": [
                    {"range": {"log_level": {"gte": error_logging_level}}},
                    {"exists": {"field": "issue_type"}},
                    {"term": {"is_merged": True}},
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
                    {"more_like_this": {
                        "fields":               ["message"],
                        "like":                 message,
                        "min_doc_freq":         1,
                        "min_term_freq":        1,
                        "minimum_should_match": "5<" + search_cfg["SearchLogsMinShouldMatch"],
                        "max_query_terms":      search_cfg["MaxQueryTerms"],
                        "boost":                1.0,
                    }},
                ],
                "should": [
                    {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                ]}}}


if __name__ == '__main__':
    unittest.main()
