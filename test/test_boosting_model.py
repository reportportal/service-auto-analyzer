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
from boosting_decision_making.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from boosting_decision_making import weighted_similarity_calculator
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
        self.log_message_suggest = "log_message_suggest.json"
        self.boost_model_results = "boost_model_results.json"
        self.suggest_boost_model_results = "suggest_boost_model_results.json"
        self.epsilon = 0.0001
        model_settings = utils.read_json_file("", "model_settings.json", to_json=True)
        self.boost_model_folder = model_settings["BOOST_MODEL_FOLDER"]
        self.suggest_boost_model_folder =\
            model_settings["SUGGEST_BOOST_MODEL_FOLDER"]
        self.weights_folder = model_settings["SIMILARITY_WEIGHTS_FOLDER"]
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @staticmethod
    def get_default_config(
            number_of_log_lines,
            filter_fields=["detected_message", "stacktrace"],
            filter_fields_any=[],
            min_should_match=0.8):
        """Get default config"""
        return {
            "max_query_terms":  50,
            "min_should_match": min_should_match,
            "min_word_length":  0,
            "filter_min_should_match": filter_fields,
            "filter_min_should_match_any": filter_fields_any,
            "number_of_log_lines": number_of_log_lines
        }

    @staticmethod
    @utils.ignore_warnings
    def get_fixture(fixture_name, jsonify=True):
        """Read fixture from file"""
        with open(os.path.join("fixtures", fixture_name), "r") as file:
            return file.read() if not jsonify else json.loads(file.read())

    @utils.ignore_warnings
    def test_random_run(self):
        print("Weights model folder: ", self.weights_folder)
        for folder in [self.boost_model_folder,
                       self.suggest_boost_model_folder]:
            print("Boost model folder ", folder)
            decision_maker = BoostingDecisionMaker(folder)
            test_data_size = 5
            random_data = np.random.rand(test_data_size, len(decision_maker.get_feature_ids()))
            result, result_probability = decision_maker.predict(random_data)
            result.should.have.length_of(test_data_size)
            result_probability.should.have.length_of(test_data_size)

    @utils.ignore_warnings
    def test_full_data_check(self):
        print("Boost model folder all lines: ", self.boost_model_folder_all_lines)
        print("Boost model folder not all lines: ", self.boost_model_folder_not_all_lines)
        print("Weights model folder: ", self.weights_folder)
        decision_maker_all_lines = BoostingDecisionMaker(self.boost_model_folder_all_lines)
        decision_maker_not_all_lines = BoostingDecisionMaker(self.boost_model_folder_not_all_lines)
        boost_model_results = self.get_fixture(self.boost_model_results)
        tests = []
        for log_lines, filter_fields, _decision_maker in [
                (-1, ["detected_message", "stacktrace"], decision_maker),
                (2, ["message"], decision_maker)]:
            tests.extend([
                {
                    "elastic_results": [(self.get_fixture(self.log_message),
                                         self.get_fixture(self.one_hit_search_rs_explained))],
                    "config":          self.get_default_config(number_of_log_lines=log_lines,
                                                               filter_fields=filter_fields),
                    "decision_maker":  _decision_maker
                },
                {
                    "elastic_results": [(self.get_fixture(self.log_message),
                                         self.get_fixture(self.two_hits_search_rs_explained))],
                    "config":          self.get_default_config(number_of_log_lines=log_lines,
                                                               filter_fields=filter_fields),
                    "decision_maker":  _decision_maker
                },
                {
                    "elastic_results": [(self.get_fixture(self.log_message),
                                         self.get_fixture(self.two_hits_search_rs_explained)),
                                        (self.get_fixture(self.log_message),
                                         self.get_fixture(self.one_hit_search_rs_explained))],
                    "config":          self.get_default_config(number_of_log_lines=log_lines,
                                                               filter_fields=filter_fields),
                    "decision_maker":  _decision_maker
                },
                {
                    "elastic_results": [(self.get_fixture(self.log_message_only_small_logs),
                                         self.get_fixture(self.two_hits_search_rs_small_logs))],
                    "config":          self.get_default_config(number_of_log_lines=log_lines,
                                                               filter_fields=filter_fields),
                    "decision_maker":  _decision_maker
                },
            ])

        for idx, test in enumerate(tests):
            feature_ids = test["decision_maker"].get_feature_ids()
            weight_log_sim = None
            if self.weights_folder.strip():
                weight_log_sim = weighted_similarity_calculator.\
                    WeightedSimilarityCalculator(folder=self.weights_folder)
            _boosting_featurizer = BoostingFeaturizer(test["elastic_results"],
                                                      test["config"],
                                                      feature_ids,
                                                      weighted_log_similarity_calculator=weight_log_sim)

            with sure.ensure('Error in the test case number: {0}', idx):
                gathered_data, issue_type_names = _boosting_featurizer.gather_features_info()
                gathered_data.should.equal(boost_model_results[str(idx)][0],
                                           epsilon=self.epsilon)
                predict_label, predict_probability = test["decision_maker"].predict(
                    gathered_data)
                predict_label.tolist().should.equal(boost_model_results[str(idx)][1],
                                                    epsilon=self.epsilon)
                predict_probability.tolist().should.equal(boost_model_results[str(idx)][2],
                                                          epsilon=self.epsilon)

    @utils.ignore_warnings
    def test_full_data_check_suggests(self):
        print("Boost model folder suggests: ", self.suggest_boost_model_folder)
        print("Weights model folder suggests: ", self.weights_folder)
        decision_maker = BoostingDecisionMaker(folder=self.suggest_boost_model_folder)
        boost_model_results = self.get_fixture(self.suggest_boost_model_results)
        tests = []
        all_configs = [(-1,
                        ["detected_message_extended", "detected_message_without_params_extended"],
                        decision_maker),
                       (2,
                        ["message_extended", "message_without_params_extended"],
                        decision_maker)]
        for log_lines, filter_fields_any, _decision_maker in all_configs:
            tests.extend([
                {
                    "elastic_results": [(self.get_fixture(self.log_message_suggest),
                                         self.get_fixture(self.one_hit_search_rs_explained))],
                    "config":          TestBoostingModel.get_default_config(
                        number_of_log_lines=log_lines,
                        filter_fields=[],
                        filter_fields_any=filter_fields_any,
                        min_should_match=0.4),
                    "decision_maker":  _decision_maker
                },
                {
                    "elastic_results": [(self.get_fixture(self.log_message_suggest),
                                         self.get_fixture(self.two_hits_search_rs_explained))],
                    "config":          TestBoostingModel.get_default_config(
                        number_of_log_lines=log_lines,
                        filter_fields=[],
                        filter_fields_any=filter_fields_any,
                        min_should_match=0.4),
                    "decision_maker":  _decision_maker
                },
                {
                    "elastic_results": [(self.get_fixture(self.log_message_suggest),
                                         self.get_fixture(self.two_hits_search_rs_explained)),
                                        (self.get_fixture(self.log_message_suggest),
                                         self.get_fixture(self.one_hit_search_rs_explained))],
                    "config":          TestBoostingModel.get_default_config(
                        number_of_log_lines=log_lines,
                        filter_fields=[],
                        filter_fields_any=filter_fields_any,
                        min_should_match=0.4),
                    "decision_maker":  _decision_maker
                },
                {
                    "elastic_results": [(self.get_fixture(self.log_message_only_small_logs),
                                         self.get_fixture(self.two_hits_search_rs_small_logs))],
                    "config":          TestBoostingModel.get_default_config(
                        number_of_log_lines=log_lines,
                        filter_fields=[],
                        filter_fields_any=filter_fields_any,
                        min_should_match=0.4),
                    "decision_maker":  _decision_maker
                },
            ])
        for idx, test in enumerate(tests):
            feature_ids = test["decision_maker"].get_feature_ids()
            weight_log_sim = None
            if self.weights_folder.strip():
                weight_log_sim = weighted_similarity_calculator.\
                    WeightedSimilarityCalculator(folder=self.weights_folder)
            _boosting_featurizer = SuggestBoostingFeaturizer(
                test["elastic_results"],
                test["config"],
                feature_ids,
                weighted_log_similarity_calculator=weight_log_sim)
            with sure.ensure('Error in the test case number: {0}', idx):
                gathered_data, test_item_ids = _boosting_featurizer.gather_features_info()
                predict_label, predict_probability = test["decision_maker"].predict(gathered_data)
                gathered_data.should.equal(boost_model_results[str(idx)][0],
                                           epsilon=self.epsilon)

                predict_label.tolist().should.equal(boost_model_results[str(idx)][1],
                                                    epsilon=self.epsilon)
                predict_probability.tolist().should.equal(boost_model_results[str(idx)][2],
                                                          epsilon=self.epsilon)
