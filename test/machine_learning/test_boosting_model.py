#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import unittest

import numpy as np

from app.commons.object_saving import create_filesystem
from app.machine_learning.boosting_featurizer import BoostingFeaturizer
from app.machine_learning.models import DefectTypeModel, WeightedSimilarityCalculator
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker
from app.machine_learning.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.utils import utils
from test import get_fixture


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
        model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
        self.boost_model_folder = model_settings["BOOST_MODEL_FOLDER"]
        self.suggest_boost_model_folder = model_settings["SUGGEST_BOOST_MODEL_FOLDER"]
        self.weights_folder = model_settings["SIMILARITY_WEIGHTS_FOLDER"]
        self.global_defect_type_model_folder = model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @staticmethod
    def get_default_config(
        number_of_log_lines,
        filter_fields=["detected_message", "stacktrace"],
        filter_fields_any=[],
        min_should_match=0.0,
    ):
        """Get default config"""
        return {
            "max_query_terms": 50,
            "min_should_match": min_should_match,
            "min_word_length": 0,
            "filter_min_should_match": filter_fields,
            "filter_min_should_match_any": filter_fields_any,
            "number_of_log_lines": number_of_log_lines,
            "chosen_namespaces": {"tv.horizon": 25, "sun.reflect": 10},
            "boosting_model": utils.read_json_file("res", "model_settings.json", to_json=True)["BOOST_MODEL_FOLDER"],
            "time_weight_decay": 0.95,
        }

    @utils.ignore_warnings
    def test_random_run(self):
        print("Weights model folder: ", self.weights_folder)
        for folder in [self.boost_model_folder, self.suggest_boost_model_folder]:
            print("Boost model folder ", folder)
            decision_maker = BoostingDecisionMaker(create_filesystem(folder))
            decision_maker.load_model()
            test_data_size = 5
            random_data = (
                np.random.Generator(np.random.PCG64(1337))
                .random(size=(test_data_size, len(decision_maker.feature_ids)))
                .tolist()
            )
            result, result_probability = decision_maker.predict(random_data)
            assert len(result) == test_data_size
            assert len(result_probability) == test_data_size

    @utils.ignore_warnings
    def test_full_data_check(self):
        print("Boost model folder : ", self.boost_model_folder)
        print("Weights model folder : ", self.weights_folder)
        print("Global defect type model folder : ", self.global_defect_type_model_folder)
        decision_maker = BoostingDecisionMaker(create_filesystem(self.boost_model_folder))
        decision_maker.load_model()
        boost_model_results = get_fixture(self.boost_model_results, to_json=True)
        tests = []
        for log_lines, filter_fields, _decision_maker in [
            (-1, ["detected_message", "stacktrace"], decision_maker),
            (2, ["message"], decision_maker),
        ]:
            tests.extend(
                [
                    {
                        "elastic_results": [
                            (
                                get_fixture(self.log_message, to_json=True),
                                get_fixture(self.one_hit_search_rs_explained, to_json=True),
                            )
                        ],
                        "config": self.get_default_config(number_of_log_lines=log_lines, filter_fields=filter_fields),
                        "decision_maker": _decision_maker,
                    },
                    {
                        "elastic_results": [
                            (
                                get_fixture(self.log_message, to_json=True),
                                get_fixture(self.two_hits_search_rs_explained, to_json=True),
                            )
                        ],
                        "config": self.get_default_config(number_of_log_lines=log_lines, filter_fields=filter_fields),
                        "decision_maker": _decision_maker,
                    },
                    {
                        "elastic_results": [
                            (
                                get_fixture(self.log_message, to_json=True),
                                get_fixture(self.two_hits_search_rs_explained, to_json=True),
                            ),
                            (
                                get_fixture(self.log_message, to_json=True),
                                get_fixture(self.one_hit_search_rs_explained, to_json=True),
                            ),
                        ],
                        "config": self.get_default_config(number_of_log_lines=log_lines, filter_fields=filter_fields),
                        "decision_maker": _decision_maker,
                    },
                ]
            )

        for idx, test in enumerate(tests):
            print(f"Running test {idx}")
            feature_ids = test["decision_maker"].feature_ids
            weight_log_sim = None
            if self.weights_folder.strip():
                weight_log_sim = WeightedSimilarityCalculator(create_filesystem(self.weights_folder))
                weight_log_sim.load_model()
            _boosting_featurizer = BoostingFeaturizer(
                test["elastic_results"], test["config"], feature_ids, weighted_log_similarity_calculator=weight_log_sim
            )
            if self.global_defect_type_model_folder.strip():
                model = DefectTypeModel(create_filesystem(self.global_defect_type_model_folder))
                model.load_model()
                _boosting_featurizer.set_defect_type_model(model)
            gathered_data, _ = _boosting_featurizer.gather_features_info()
            predict_label, predict_probability = test["decision_maker"].predict(gathered_data)
            assert gathered_data == boost_model_results[str(idx)][0]
            assert predict_label == boost_model_results[str(idx)][1]
            assert predict_probability == boost_model_results[str(idx)][2]

    @utils.ignore_warnings
    def test_full_data_check_suggests(self):
        print("Boost model folder suggests: ", self.suggest_boost_model_folder)
        print("Weights model folder suggests: ", self.weights_folder)
        print("Global defect type model folder : ", self.global_defect_type_model_folder)
        decision_maker = BoostingDecisionMaker(create_filesystem(self.suggest_boost_model_folder))
        decision_maker.load_model()
        boost_model_results = get_fixture(self.suggest_boost_model_results, to_json=True)
        tests = []
        all_configs = [
            (
                -1,
                [
                    "detected_message_extended",
                    "detected_message_without_params_extended",
                    "detected_message_without_params_and_brackets",
                ],
                decision_maker,
            ),
            (
                2,
                ["message_extended", "message_without_params_extended", "message_without_params_and_brackets"],
                decision_maker,
            ),
        ]
        for log_lines, filter_fields_any, _decision_maker in all_configs:
            tests.extend(
                [
                    {
                        "elastic_results": [
                            (
                                get_fixture(self.log_message_suggest, to_json=True),
                                get_fixture(self.one_hit_search_rs_explained, to_json=True),
                            )
                        ],
                        "config": TestBoostingModel.get_default_config(
                            number_of_log_lines=log_lines,
                            filter_fields=[],
                            filter_fields_any=filter_fields_any,
                            min_should_match=0.4,
                        ),
                        "decision_maker": _decision_maker,
                    },
                    {
                        "elastic_results": [
                            (
                                get_fixture(self.log_message_suggest, to_json=True),
                                get_fixture(self.two_hits_search_rs_explained, to_json=True),
                            )
                        ],
                        "config": TestBoostingModel.get_default_config(
                            number_of_log_lines=log_lines,
                            filter_fields=[],
                            filter_fields_any=filter_fields_any,
                            min_should_match=0.4,
                        ),
                        "decision_maker": _decision_maker,
                    },
                    {
                        "elastic_results": [
                            (
                                get_fixture(self.log_message_suggest, to_json=True),
                                get_fixture(self.two_hits_search_rs_explained, to_json=True),
                            ),
                            (
                                get_fixture(self.log_message_suggest, to_json=True),
                                get_fixture(self.one_hit_search_rs_explained, to_json=True),
                            ),
                        ],
                        "config": TestBoostingModel.get_default_config(
                            number_of_log_lines=log_lines,
                            filter_fields=[],
                            filter_fields_any=filter_fields_any,
                            min_should_match=0.4,
                        ),
                        "decision_maker": _decision_maker,
                    },
                    {
                        "elastic_results": [
                            (
                                get_fixture(self.log_message_only_small_logs, to_json=True),
                                get_fixture(self.two_hits_search_rs_small_logs, to_json=True),
                            )
                        ],
                        "config": TestBoostingModel.get_default_config(
                            number_of_log_lines=log_lines,
                            filter_fields=[],
                            filter_fields_any=filter_fields_any,
                            min_should_match=0.0,
                        ),
                        "decision_maker": _decision_maker,
                    },
                ]
            )
        for idx, test in enumerate(tests):
            print(f"Running test {idx}")
            feature_ids = test["decision_maker"].feature_ids
            weight_log_sim = None
            if self.weights_folder.strip():
                weight_log_sim = WeightedSimilarityCalculator(create_filesystem(self.weights_folder))
                weight_log_sim.load_model()
            _boosting_featurizer = SuggestBoostingFeaturizer(
                test["elastic_results"], test["config"], feature_ids, weighted_log_similarity_calculator=weight_log_sim
            )
            if self.global_defect_type_model_folder.strip():
                model = DefectTypeModel(create_filesystem(self.global_defect_type_model_folder))
                model.load_model()
                _boosting_featurizer.set_defect_type_model(model)

            gathered_data, _ = _boosting_featurizer.gather_features_info()
            predict_label, predict_probability = test["decision_maker"].predict(gathered_data)
            assert gathered_data == boost_model_results[str(idx)][0]
            assert predict_label == boost_model_results[str(idx)][1]
            assert predict_probability == boost_model_results[str(idx)][2]
