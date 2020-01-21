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
import numpy as np
from boosting_decision_making.boosting_featurizer import BoostingFeaturizer
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker


def ignore_warnings(method):
    """Decorator for ignoring warnings"""
    def _inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method(*args, **kwargs)
        return result
    return _inner


class TestBoostingModel(unittest.TestCase):
    """Tests boosting model prediction functionality"""

    def setUp(self):
        self.one_hit_search_rs_explained = "one_hit_search_rs_explained.json"
        self.two_hits_search_rs_explained = "two_hits_search_rs_explained.json"
        self.log_message = "log_message.json"
        self.boost_model_results = "boost_model_results.json"
        self.epsilon = 0.0001
        self.boost_model_folder = os.getenv("BOOST_MODEL_FOLDER")
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
            "filter_min_should_match": ["detected_message", "message"]
        }

    @staticmethod
    def get_fixture(fixture_name, jsonify=True):
        """Read fixture from file"""
        with open(os.path.join("fixtures", fixture_name), "r") as file:
            return file.read() if not jsonify else json.loads(file.read())

    @ignore_warnings
    def test_random_run(self):
        print("Boost model folder: ", self.boost_model_folder)
        decision_maker = BoostingDecisionMaker(self.boost_model_folder)
        test_data_size = 5
        random_data = np.random.rand(test_data_size, len(decision_maker.get_feature_ids()))
        result, result_probability = decision_maker.predict(random_data)
        result.should.have.length_of(test_data_size)
        result_probability.should.have.length_of(test_data_size)

    @ignore_warnings
    def test_full_data_check(self):
        print("Boost model folder: ", self.boost_model_folder)
        decision_maker = BoostingDecisionMaker(self.boost_model_folder)
        boost_model_results = self.get_fixture(self.boost_model_results)
        tests = [
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingModel.get_default_config(),
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained))],
                "config":          TestBoostingModel.get_default_config(),
            },
            {
                "elastic_results": [(self.get_fixture(self.log_message),
                                     self.get_fixture(self.two_hits_search_rs_explained)),
                                    (self.get_fixture(self.log_message),
                                     self.get_fixture(self.one_hit_search_rs_explained))],
                "config":          TestBoostingModel.get_default_config(),
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
