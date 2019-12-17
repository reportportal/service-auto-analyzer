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
import warnings
import os
import json
import sure # noqa
from boosting_decision_making.boosting_featurizer import BoostingFeaturizer


def ignore_warnings(method):
    """Decorator for ignoring warnings"""
    def _inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method(*args, **kwargs)
        return result
    return _inner


class TestBoostingFeaturizer(unittest.TestCase):
    """Tests boosting feature creation functionality"""

    def setUp(self):
        self.one_hit_search_rs_explained = "one_hit_search_rs_explained.json"
        self.two_hits_search_rs_explained = "two_hits_search_rs_explained.json"
        self.log_message = "log_message.json"
        self.epsilon = 0.0001
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.DEBUG)

    @staticmethod
    def get_default_config():
        """Get default config"""
        return {
            "max_query_terms":  50,
            "min_should_match": 0.8,
            "min_word_length":  0,
        }

    @staticmethod
    def get_fixture(fixture_name, jsonify=True):
        """Read fixture from file"""
        with open(os.path.join("fixtures", fixture_name), "r") as file:
            return file.read() if not jsonify else json.loads(file.read())

    @ignore_warnings
    def test_normalize_results(self):
        tests = [
            {
                "elastic_results": [],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          [],
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          [[{"_score": 158.08437,
                                      "normalized_score": 1.0, }]],
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          [[{"_score": 158.08437,
                                      "normalized_score": 0.6709,
                                      },
                                     {"_score": 77.53298,
                                      "normalized_score": 0.3291,
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

    @ignore_warnings
    def test_find_most_relevant_by_type(self):
        tests = [
            {
                "elastic_results": [],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {},
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": {"mrHit": {"_score": 158.08437,
                                                        "_id": "1"},
                                              "log_message": self.get_fixture(
                                                  self.log_message)["log_message"],
                                              "score": 1.0, },
                                    }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": {"mrHit": {"_score": 158.08437,
                                                        "_id": "1"},
                                              "log_message": self.get_fixture(
                                                  self.log_message)["log_message"],
                                              "score": 0.6709, },
                                    "PB001": {"mrHit": {"_score": 77.53298,
                                                        "_id": "2"},
                                              "log_message": self.get_fixture(
                                                  self.log_message)["log_message"],
                                              "score": 0.3291, },
                                    }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.two_hits_search_rs_explained)),
                                    (self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": {"mrHit": {"_score": 158.08437,
                                                        "_id": "1"},
                                              "log_message": self.get_fixture(
                                                  self.log_message)["log_message"],
                                              "score": 0.8355, },
                                    "PB001": {"mrHit": {"_score": 77.53298,
                                                        "_id": "2"},
                                              "log_message": self.get_fixture(
                                                  self.log_message)["log_message"],
                                              "score": 0.1645, },
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

    @ignore_warnings
    def test_calculate_features(self):
        all_features = list(BoostingFeaturizer([], [], []).feature_functions.keys())
        tests = [
            {
                "elastic_results": [],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {},
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": [1.0] * 12, }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": [0.6709, 1.0, 0.6709,
                                              1.0, 0.6709, 1.0, 0.6709,
                                              0.5, 1.0, 0.5, 1.0, 1.0],
                                    "PB001": [0.3291, 0.5, 0.3291, 0.5,
                                              0.3291, 0.5, 0.3291, 0.5,
                                              0.4905, 0.5, 0.86, 0.8835], }
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.two_hits_search_rs_explained)),
                                    (self.get_fixture(self.log_message)["log_message"],
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingFeaturizer.get_default_config(),
                "result":          {"AB001": [0.8355, 1.0, 1.0, 1.0,
                                              0.6709, 1.0, 0.8355, 0.6667,
                                              1.0, 0.5, 1.0, 1.0],
                                    "PB001": [0.1645, 0.5, 0.3291,
                                              0.5, 0.3291, 0.5, 0.3291,
                                              0.3334, 0.3291, 0.5, 0.86, 0.8835], }
            },
        ]
        for idx, test in enumerate(tests):
            _boosting_featurizer = BoostingFeaturizer(test["elastic_results"],
                                                      test["config"],
                                                      all_features)
            for feature in _boosting_featurizer.feature_functions:
                func, args = _boosting_featurizer.feature_functions[feature]
                result = func(**args)
                with sure.ensure('Error in the calculating feature: {0}, test case number: {1}',
                                 feature, idx):
                    result.should.have.length_of(len(test["result"]))
                    for issue_type in result:
                        result[issue_type].should.equal(test["result"][issue_type][feature],
                                                        epsilon=self.epsilon)
            with sure.ensure('Error in the test case number: {0}', idx):
                gathered_data, issue_type_names = _boosting_featurizer.gather_features_info()
                for i in range(len(issue_type_names)):
                    gathered_data[i].should.equal(test["result"][issue_type_names[i]],
                                                  epsilon=self.epsilon)
