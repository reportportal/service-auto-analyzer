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
import logging
import os
import json
import sure # noqa
from boosting_decision_making.boosting_featurizer import BoostingFeaturizer
from boosting_decision_making.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from utils import utils


class TestBoostingFeaturizer(unittest.TestCase):
    """Tests boosting feature creation functionality"""
    @utils.ignore_warnings
    def setUp(self):
        self.one_hit_search_rs_explained = "one_hit_search_rs_explained.json"
        self.two_hits_search_rs_explained = "two_hits_search_rs_explained.json"
        self.log_message = "log_message.json"
        self.log_message_wo_stacktrace = "log_message_wo_stacktrace.json"
        self.one_hit_search_rs_explained_wo_stacktrace =\
            "one_hit_search_rs_explained_wo_stacktrace.json"
        self.log_message_only_small_logs = "log_message_only_small_logs.json"
        self.one_hit_search_rs_small_logs = "one_hit_search_rs_small_logs.json"
        self.two_hits_search_rs_small_logs = "two_hits_search_rs_small_logs.json"
        self.three_hits_search_rs_explained = "three_hits_search_rs_explained.json"
        self.epsilon = 0.0001
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @staticmethod
    @utils.ignore_warnings
    def get_default_config(filter_fields=["detected_message", "stacktrace"]):
        """Get default config"""
        return {
            "max_query_terms":  50,
            "min_should_match": 0.8,
            "min_word_length":  0,
            "filter_min_should_match": filter_fields,
            "similarity_weights_folder": os.getenv("SIMILARITY_WEIGHTS_FOLDER", ""),
            "number_of_log_lines": -1
        }

    @staticmethod
    @utils.ignore_warnings
    def get_fixture(fixture_name, jsonify=True):
        """Read fixture from file"""
        with open(os.path.join("fixtures", fixture_name), "r") as file:
            return file.read() if not jsonify else json.loads(file.read())

    @utils.ignore_warnings
    def test_normalize_results(self):
        tests = [
            {
                "elastic_results": [],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          [],
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          [[{"_score": 158.08437,
                                      "normalized_score": 1.0, }]],
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          [[{"_score": 158.08437,
                                      "normalized_score": 1.0,
                                      },
                                     {"_score": 77.53298,
                                      "normalized_score": 0.4904,
                                      }, ]],
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                _boosting_featurizer = BoostingFeaturizer(test["elastic_results"],
                                                          test["config"],
                                                          [])
                _boosting_featurizer.all_results.should.have.length_of(len(test["result"]))
                for i in range(len(test["result"])):
                    for j in range(len(test["result"][i])):
                        for field in test["result"][i][j]:
                            elastic_res = _boosting_featurizer.all_results[i][1][j]
                            elastic_res[field].should.equal(test["result"][i][j][field],
                                                            epsilon=self.epsilon)

    @utils.ignore_warnings
    def test_find_most_relevant_by_type(self):
        tests = [
            {
                "elastic_results": [],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {},
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": {"mrHit": {"_score": 158.08437,
                                                        "_id": "1"},
                                              "compared_log": self.get_fixture(self.log_message),
                                              "score": 1.0, },
                                    }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": {"mrHit": {"_score": 158.08437,
                                                        "_id": "1"},
                                              "compared_log": self.get_fixture(self.log_message),
                                              "score": 0.6709, },
                                    "PB001": {"mrHit": {"_score": 77.53298,
                                                        "_id": "2"},
                                              "compared_log": self.get_fixture(self.log_message),
                                              "score": 0.3291, },
                                    }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained)),
                                    (self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": {"mrHit": {"_score": 158.08437,
                                                        "_id": "1"},
                                              "compared_log": self.get_fixture(self.log_message),
                                              "score": 0.8031, },
                                    "PB001": {"mrHit": {"_score": 77.53298,
                                                        "_id": "2"},
                                              "compared_log": self.get_fixture(self.log_message),
                                              "score": 0.1969, },
                                    }
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                _boosting_featurizer = BoostingFeaturizer(test["elastic_results"],
                                                          test["config"],
                                                          [])
                scores_by_issue_type = _boosting_featurizer.find_most_relevant_by_type()
                scores_by_issue_type.should.have.length_of(len(test["result"]))
                for issue_type in test["result"]:
                    scores_by_issue_type.keys().should.contain(issue_type)
                    elastic_res = scores_by_issue_type[issue_type]
                    for field in test["result"][issue_type]:
                        if type(test["result"][issue_type][field]) != dict:
                            elastic_res[field].should.equal(test["result"][issue_type][field],
                                                            epsilon=self.epsilon)
                        else:
                            for field_dict in test["result"][issue_type][field]:
                                result_field_dict = test["result"][issue_type][field][field_dict]
                                elastic_res[field][field_dict].should.equal(result_field_dict,
                                                                            epsilon=self.epsilon)

    @utils.ignore_warnings
    def test_filter_by_min_should_match(self):
        tests = [
            {
                "elastic_results": [],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[]),
                "result":          [],
            },
            {
                "elastic_results": [],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[
                    "detected_message", "stacktrace"]),
                "result":          [],
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[
                    "detected_message", "stacktrace"]),
                "result":          [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[
                    "message"]),
                "result":          [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained)),
                                    (self.get_fixture(self.log_message_wo_stacktrace),
                                     self.get_fixture(self.one_hit_search_rs_explained_wo_stacktrace))],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[
                    "message"]),
                "result":          [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained)),
                                    (self.get_fixture(self.log_message_wo_stacktrace),
                                     self.get_fixture(self.one_hit_search_rs_explained_wo_stacktrace))]
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained)),
                                    (self.get_fixture(self.log_message_wo_stacktrace),
                                     self.get_fixture(self.one_hit_search_rs_explained_wo_stacktrace))],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[
                    "detected_message", "stacktrace"]),
                "result":          [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained)),
                                    (self.get_fixture(self.log_message_wo_stacktrace),
                                     self.get_fixture(self.one_hit_search_rs_explained_wo_stacktrace))]
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message_only_small_logs),
                                     self.get_fixture(self.one_hit_search_rs_small_logs))],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[
                    "detected_message", "stacktrace"]),
                "result":          []
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message_only_small_logs),
                                     self.get_fixture(self.two_hits_search_rs_small_logs))],
                "config":           TestBoostingFeaturizer.get_default_config(filter_fields=[
                    "detected_message", "stacktrace"]),
                "result":          [(self.get_fixture(self.log_message_only_small_logs),
                                     self.get_fixture(self.two_hits_search_rs_small_logs))]
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                _boosting_featurizer = BoostingFeaturizer(test["elastic_results"], test["config"], [])
                all_results = test["elastic_results"]
                for field in test["config"]["filter_min_should_match"]:
                    all_results = _boosting_featurizer.filter_by_min_should_match(all_results, field=field)
                all_results.should.have.length_of(len(test["result"]))
                for idx, (log, hits) in enumerate(all_results):
                    log["_id"].should.equal(test["result"][idx][0]["_id"])
                    for i, hit in enumerate(hits["hits"]["hits"]):
                        hit["_id"].should.equal(test["result"][idx][1]["hits"]["hits"][i]["_id"])

    @utils.ignore_warnings
    def test_find_most_relevant_by_type_for_suggests(self):
        tests = [
            {
                "elastic_results": [],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {},
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {1: {"mrHit": {"_score": 158.08437,
                                                  "_id": "1"},
                                        "compared_log": self.get_fixture(self.log_message),
                                        "score": 1.0, },
                                    }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {1: {"mrHit": {"_score": 158.08437,
                                                  "_id": "1"},
                                        "compared_log": self.get_fixture(self.log_message),
                                        "score": 1.0, },
                                    2: {"mrHit": {"_score": 77.53298,
                                                  "_id": "2"},
                                        "compared_log": self.get_fixture(self.log_message),
                                        "score": 0.4905, },
                                    }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained)),
                                    (self.get_fixture(self.log_message),
                                     self.get_fixture(self.three_hits_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {1: {"mrHit": {"_score": 158.08437,
                                                  "_id": "1"},
                                        "compared_log": self.get_fixture(self.log_message),
                                        "score": 0.9392, },
                                    2: {"mrHit": {"_score": 168.31,
                                                  "_id": "2"},
                                        "compared_log": self.get_fixture(self.log_message),
                                        "score": 1.0, },
                                    3: {"mrHit": {"_score": 85.345,
                                                  "_id": "3"},
                                        "compared_log": self.get_fixture(self.log_message),
                                        "score": 0.507, },
                                    }
            },
        ]
        for idx, test in enumerate(tests[3:]):
            with sure.ensure('Error in the test case number: {0}', idx):
                _boosting_featurizer = SuggestBoostingFeaturizer(test["elastic_results"],
                                                                 test["config"],
                                                                 [])
                scores_by_issue_type = _boosting_featurizer.find_most_relevant_by_type()
                scores_by_issue_type.should.have.length_of(len(test["result"]))
                for issue_type in test["result"]:
                    scores_by_issue_type.keys().should.contain(issue_type)
                    elastic_res = scores_by_issue_type[issue_type]
                    for field in test["result"][issue_type]:
                        if type(test["result"][issue_type][field]) != dict:
                            elastic_res[field].should.equal(test["result"][issue_type][field],
                                                            epsilon=self.epsilon)
                        else:
                            for field_dict in test["result"][issue_type][field]:
                                result_field_dict = test["result"][issue_type][field][field_dict]
                                elastic_res[field][field_dict].should.equal(result_field_dict,
                                                                            epsilon=self.epsilon)
