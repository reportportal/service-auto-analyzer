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
import sure
import sys
import os

import commons.launch_objects as launch_objects
import commons.esclient as esclient

class TestEsQuery(unittest.TestCase):

    def test_build_analyze_query(self):
        search_cfg = {
            "MinShouldMatch": "80%",
            "MinTermFreq":    25,
            "MinDocFreq":     30,
            "BoostAA":        10,
            "BoostLaunch":    5,
            "BoostUniqueID":  3,
            "MaxQueryTerms":  50,
        }
        ERROR_LOGGING_LEVEL = 40000

        es_client = esclient.EsClient(search_cfg = search_cfg)
        launch = launch_objects.Launch(**{
                                    "analyzerConfig": {"analyzerMode": "SearchModeAll"}, 
                                    "launchId": 123, 
                                    "launchName": "Launch name",
                                    "project": 1
                                    })
        query_from_esclient = es_client.build_analyze_query(launch, "unique", "hello world")
        demo_query = self.build_demo_query(search_cfg, "ALL", "Launch name", "unique", "hello world", ERROR_LOGGING_LEVEL)

        query_from_esclient.should.equal(demo_query)

    def build_demo_query(self, search_cfg, mode, launch_name, unique_id, log_message, error_logging_level):
        return {
            "size": 10,
            "query": {
                "bool": {
                    "must_not": [
                        {"wildcard":{"issue_type":"TI*"}},
                        {"wildcard":{"issue_type":"ti*"}},
                    ],
                    "must": [
                        {
                            "range": {
                                "log_level": {
                                    "gte": error_logging_level,
                                },
                            },
                        },
                        {
                            "exists": {
                                "field": "issue_type",
                            },
                        },
                        {   "term": {
                                "is_merged":True,
                            }
                        },
                        {
                            "more_like_this": {
                                "fields":               ["message"],
                                "like":                 log_message,
                                "min_doc_freq":         search_cfg["MinDocFreq"],
                                "min_term_freq":        search_cfg["MinTermFreq"],
                                "minimum_should_match": "5<" + search_cfg["MinShouldMatch"],
                                "max_query_terms":      search_cfg["MaxQueryTerms"],
                            },
                        },
                    ],
                    "should": [
                        {"term": {
                            "unique_id": {
                                "value": unique_id,
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
                    ],
                },
            },
        }

if __name__ == '__main__':
    unittest.main()