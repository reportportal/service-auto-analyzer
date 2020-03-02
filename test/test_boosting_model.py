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
import numpy as np
from boosting_decision_making.boosting_featurizer import BoostingFeaturizer
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from utils import utils


class TestBoostingModel(unittest.TestCase):
    """Tests boosting model prediction functionality"""
    @utils.ignore_warnings
    def setUp(self):
        self.one_hit_search_rs_explained = "one_hit_search_rs_explained.json"
        self.two_hits_search_rs_explained = "two_hits_search_rs_explained.json"
        self.two_hits_search_rs_small_logs = "two_hits_search_rs_small_logs.json"
        self.log_message = "log_message.json"
        self.log_message_only_small_logs = "log_message_only_small_logs.json"
        self.boost_model_results = "boost_model_results.json"
        self.epsilon = 0.0001
        self.boost_model_folder = os.getenv("BOOST_MODEL_FOLDER")
        self.weights_folder = os.getenv("SIMILARITY_WEIGHTS_FOLDER", "")
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def get_default_config(self):
        """Get default config"""
        return {
            "max_query_terms":  50,
            "min_should_match": 0.8,
            "min_word_length":  0,
            "filter_min_should_match": ["detected_message", "message"],
            "similarity_weights_folder": self.weights_folder,
            "number_of_log_lines": -1
        }

    @staticmethod
    @utils.ignore_warnings
    def get_fixture(fixture_name, jsonify=True):
        """Read fixture from file"""
        with open(os.path.join("fixtures", fixture_name), "r") as file:
            return file.read() if not jsonify else json.loads(file.read())

    @utils.ignore_warnings
    def test_random_run(self):
        print("Boost model folder: ", self.boost_model_folder)
        print("Weights model folder: ", self.weights_folder)
        decision_maker = BoostingDecisionMaker(self.boost_model_folder)
        test_data_size = 5
        random_data = np.random.rand(test_data_size, len(decision_maker.get_feature_ids()))
        result, result_probability = decision_maker.predict(random_data)
        result.should.have.length_of(test_data_size)
        result_probability.should.have.length_of(test_data_size)

    @utils.ignore_warnings
    def test_full_data_check(self):
        print("Boost model folder: ", self.boost_model_folder)
        print("Weights model folder: ", self.weights_folder)
        decision_maker = BoostingDecisionMaker(self.boost_model_folder)
        boost_model_results = self.get_fixture(self.boost_model_results)
        tests = [
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          self.get_default_config(),
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          self.get_default_config(),
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained)),
                                    (self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          self.get_default_config(),
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message_only_small_logs),
                                     self.get_fixture(self.two_hits_search_rs_small_logs))],
                "config":          self.get_default_config(),
            },
        ]
        for idx, test in enumerate(tests):
            _boosting_featurizer = BoostingFeaturizer(test["elastic_results"],
                                                      test["config"],
                                                      decision_maker.get_feature_ids())
            with sure.ensure('Error in the test case number: {0}', idx):
                gathered_data, issue_type_names = _boosting_featurizer.gather_features_info()
                gathered_data.should.equal(boost_model_results[str(idx)][0],
                                           epsilon=self.epsilon)
                predict_label, predict_probability = decision_maker.predict(gathered_data)
                predict_label.tolist().should.equal(boost_model_results[str(idx)][1],
                                                    epsilon=self.epsilon)
                predict_probability.tolist().should.equal(boost_model_results[str(idx)][2],
                                                          epsilon=self.epsilon)
