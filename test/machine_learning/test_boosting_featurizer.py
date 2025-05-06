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

import unittest
from test import get_fixture

from app.commons.object_saving import create_filesystem
from app.machine_learning.boosting_featurizer import BoostingFeaturizer
from app.machine_learning.models.weighted_similarity_calculator import WeightedSimilarityCalculator
from app.machine_learning.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.utils import utils


class TestBoostingFeaturizer(unittest.TestCase):
    """Tests boosting feature creation functionality"""

    @utils.ignore_warnings
    def setUp(self):
        self.one_hit_search_rs_explained = "one_hit_search_rs_explained.json"
        self.two_hits_search_rs_explained = "two_hits_search_rs_explained.json"
        self.log_message = "log_message.json"
        self.log_message_wo_stacktrace = "log_message_wo_stacktrace.json"
        self.one_hit_search_rs_explained_wo_stacktrace = "one_hit_search_rs_explained_wo_stacktrace.json"
        self.log_message_only_small_logs = "log_message_only_small_logs.json"
        self.one_hit_search_rs_small_logs = "one_hit_search_rs_small_logs.json"
        self.two_hits_search_rs_small_logs = "two_hits_search_rs_small_logs.json"
        self.three_hits_search_rs_explained = "three_hits_search_rs_explained.json"
        self.one_hit_search_rs_explained_wo_params = "one_hit_search_rs_explained_wo_params.json"
        self.epsilon = 0.0001
        model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
        self.weights_folder = model_settings["SIMILARITY_WEIGHTS_FOLDER"]

    @staticmethod
    @utils.ignore_warnings
    def get_default_config(filter_fields=None, filter_fields_any=None):
        """Get default config"""
        if filter_fields is None:
            filter_fields = ["detected_message", "stacktrace"]
        if filter_fields_any is None:
            filter_fields_any = []
        return {
            "max_query_terms": 50,
            "min_should_match": 0.41,
            "min_word_length": 0,
            "filter_min_should_match": filter_fields,
            "filter_min_should_match_any": filter_fields_any,
            "number_of_log_lines": -1,
            "boosting_model": "",
            "time_weight_decay": 0.95,
        }

    @utils.ignore_warnings
    def test_normalize_results(self):
        tests = [
            {
                "elastic_results": [],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": [],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": [
                    [
                        {
                            "_score": 158.08437,
                            "normalized_score": 1.0,
                        }
                    ]
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.two_hits_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": [
                    [
                        {
                            "_score": 158.08437,
                            "normalized_score": 1.0,
                        },
                        {
                            "_score": 77.53298,
                            "normalized_score": 0.4904,
                        },
                    ]
                ],
            },
        ]
        weight_log_sim = WeightedSimilarityCalculator(create_filesystem(self.weights_folder))
        weight_log_sim.load_model()
        for idx, test in enumerate(tests):
            print(f"Test index: {idx}")
            _boosting_featurizer = BoostingFeaturizer(
                test["elastic_results"], test["config"], [], weighted_log_similarity_calculator=weight_log_sim
            )
            assert len(_boosting_featurizer.all_results) == len(test["result"])
            for i in range(len(test["result"])):
                for j in range(len(test["result"][i])):
                    for field in test["result"][i][j]:
                        elastic_res = _boosting_featurizer.all_results[i][1][j]
                        assert abs(elastic_res[field] - test["result"][i][j][field]) <= self.epsilon

    def assert_scores_by_issue_type(self, boosting_featurizer, test):
        scores_by_issue_type = boosting_featurizer.find_most_relevant_by_type()
        assert scores_by_issue_type.keys() == test["result"].keys()
        for issue_type in test["result"]:
            elastic_res = scores_by_issue_type[issue_type]
            for field in test["result"][issue_type]:
                if not isinstance(test["result"][issue_type][field], dict):
                    assert abs(elastic_res[field] - test["result"][issue_type][field]) <= self.epsilon
                else:
                    for field_dict in test["result"][issue_type][field]:
                        result_field_dict = test["result"][issue_type][field][field_dict]
                        assert elastic_res[field][field_dict] == result_field_dict

    @utils.ignore_warnings
    def test_find_most_relevant_by_type(self):
        tests = [
            {
                "elastic_results": [],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {},
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {
                    "AB001": {
                        "mrHit": {"_score": 158.08437, "_id": "1"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 1.0,
                    },
                },
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.two_hits_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {
                    "AB001": {
                        "mrHit": {"_score": 158.08437, "_id": "1"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 0.6709,
                    },
                    "PB001": {
                        "mrHit": {"_score": 77.53298, "_id": "2"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 0.3291,
                    },
                },
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
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {
                    "AB001": {
                        "mrHit": {"_score": 158.08437, "_id": "1"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 0.8031,
                    },
                    "PB001": {
                        "mrHit": {"_score": 77.53298, "_id": "2"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 0.1969,
                    },
                },
            },
        ]
        weight_log_sim = WeightedSimilarityCalculator(create_filesystem(self.weights_folder))
        weight_log_sim.load_model()
        for idx, test in enumerate(tests):
            print(f"Test index: {idx}")
            _boosting_featurizer = BoostingFeaturizer(
                test["elastic_results"], test["config"], [], weighted_log_similarity_calculator=weight_log_sim
            )
            self.assert_scores_by_issue_type(_boosting_featurizer, test)

    def assert_elastic_results(self, results, test):
        assert len(results) == len(test["result"])
        for idx_res, (log, hits) in enumerate(results):
            assert log["_id"] == test["result"][idx_res][0]["_id"]
            for i, hit in enumerate(hits["hits"]["hits"]):
                assert hit["_id"] == hits["hits"]["hits"][i]["_id"]

    @utils.ignore_warnings
    def test_filter_by_min_should_match(self):
        tests = [
            {
                "elastic_results": [],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=[]),
                "result": [],
            },
            {
                "elastic_results": [],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=["detected_message", "stacktrace"]),
                "result": [],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=["detected_message", "stacktrace"]),
                "result": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=["message"]),
                "result": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message_wo_stacktrace, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained_wo_stacktrace, to_json=True),
                    ),
                ],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=["message"]),
                "result": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message_wo_stacktrace, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained_wo_stacktrace, to_json=True),
                    ),
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message_wo_stacktrace, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained_wo_stacktrace, to_json=True),
                    ),
                ],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=["detected_message", "stacktrace"]),
                "result": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message_wo_stacktrace, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained_wo_stacktrace, to_json=True),
                    ),
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message_only_small_logs, to_json=True),
                        get_fixture(self.one_hit_search_rs_small_logs, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=["detected_message", "stacktrace"]),
                "result": [(get_fixture(self.log_message_only_small_logs, to_json=True), {"hits": {"hits": []}})],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message_only_small_logs, to_json=True),
                        get_fixture(self.two_hits_search_rs_small_logs, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=["detected_message", "stacktrace"]),
                "result": [
                    (
                        get_fixture(self.log_message_only_small_logs, to_json=True),
                        get_fixture(self.two_hits_search_rs_small_logs, to_json=True),
                    )
                ],
            },
        ]
        weight_log_sim = WeightedSimilarityCalculator(create_filesystem(self.weights_folder))
        weight_log_sim.load_model()
        for idx, test in enumerate(tests):
            try:
                _boosting_featurizer = BoostingFeaturizer(
                    test["elastic_results"], test["config"], [], weighted_log_similarity_calculator=weight_log_sim
                )
                all_results = test["elastic_results"]
                for field in test["config"]["filter_min_should_match"]:
                    all_results = _boosting_featurizer.filter_by_min_should_match(all_results, field=field)
                self.assert_elastic_results(all_results, test)
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)

    @utils.ignore_warnings
    def test_find_most_relevant_by_type_for_suggests(self):
        tests = [
            {
                "elastic_results": [],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {},
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {
                    "1": {
                        "mrHit": {"_score": 158.08437, "_id": "1"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 1.0,
                    },
                },
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.two_hits_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {
                    "1": {
                        "mrHit": {"_score": 158.08437, "_id": "1"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 1.0,
                    },
                    "2": {
                        "mrHit": {"_score": 77.53298, "_id": "2"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 0.4905,
                    },
                },
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.two_hits_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.three_hits_search_rs_explained, to_json=True),
                    ),
                ],
                "config": TestBoostingFeaturizer.get_default_config(),
                "result": {
                    "1": {
                        "mrHit": {"_score": 158.08437, "_id": "1"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 0.9392,
                    },
                    "2": {
                        "mrHit": {"_score": 168.31, "_id": "2"},
                        "compared_log": get_fixture(self.log_message, to_json=True),
                        "score": 1.0,
                    },
                },
            },
        ]
        weight_log_sim = WeightedSimilarityCalculator(create_filesystem(self.weights_folder))
        weight_log_sim.load_model()
        for idx, test in enumerate(tests):
            print(f"Test index: {idx}")
            _boosting_featurizer = SuggestBoostingFeaturizer(
                test["elastic_results"], test["config"], [], weighted_log_similarity_calculator=weight_log_sim
            )
            self.assert_scores_by_issue_type(_boosting_featurizer, test)

    @utils.ignore_warnings
    def test_filter_by_min_should_match_any(self):
        tests = [
            {
                "elastic_results": [],
                "config": TestBoostingFeaturizer.get_default_config(filter_fields=[], filter_fields_any=[]),
                "result": [],
            },
            {
                "elastic_results": [],
                "config": TestBoostingFeaturizer.get_default_config(
                    filter_fields=[], filter_fields_any=["detected_message"]
                ),
                "result": [],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(
                    filter_fields=[],
                    filter_fields_any=["detected_message", "detected_message_without_params_extended"],
                ),
                "result": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    )
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained_wo_params, to_json=True),
                    ),
                ],
                "config": TestBoostingFeaturizer.get_default_config(
                    filter_fields=[],
                    filter_fields_any=["detected_message", "detected_message_without_params_extended"],
                ),
                "result": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained_wo_params, to_json=True),
                    ),
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained_wo_params, to_json=True),
                    ),
                ],
                "config": TestBoostingFeaturizer.get_default_config(
                    filter_fields=[], filter_fields_any=["detected_message"]
                ),
                "result": [
                    (
                        get_fixture(self.log_message, to_json=True),
                        get_fixture(self.one_hit_search_rs_explained, to_json=True),
                    ),
                    (get_fixture(self.log_message, to_json=True), {"hits": {"hits": []}}),
                ],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message_only_small_logs, to_json=True),
                        get_fixture(self.one_hit_search_rs_small_logs, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(
                    filter_fields=[],
                    filter_fields_any=["detected_message", "detected_message_without_params_extended"],
                ),
                "result": [(get_fixture(self.log_message_only_small_logs, to_json=True), {"hits": {"hits": []}})],
            },
            {
                "elastic_results": [
                    (
                        get_fixture(self.log_message_only_small_logs, to_json=True),
                        get_fixture(self.two_hits_search_rs_small_logs, to_json=True),
                    )
                ],
                "config": TestBoostingFeaturizer.get_default_config(
                    filter_fields=[],
                    filter_fields_any=["detected_message", "detected_message_without_params_extended"],
                ),
                "result": [
                    (
                        get_fixture(self.log_message_only_small_logs, to_json=True),
                        get_fixture(self.two_hits_search_rs_small_logs, to_json=True),
                    )
                ],
            },
        ]
        weight_log_sim = WeightedSimilarityCalculator(create_filesystem(self.weights_folder))
        weight_log_sim.load_model()
        for idx, test in enumerate(tests):
            print(f"Test index: {idx}")
            _boosting_featurizer = SuggestBoostingFeaturizer(
                test["elastic_results"], test["config"], [], weighted_log_similarity_calculator=weight_log_sim
            )
            all_results = test["elastic_results"]
            all_results = _boosting_featurizer.filter_by_min_should_match_any(
                all_results, fields=test["config"]["filter_min_should_match_any"]
            )
            self.assert_elastic_results(all_results, test)
