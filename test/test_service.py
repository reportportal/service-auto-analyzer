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
import json
import logging
import sure # noqa
import httpretty
from utils import utils


class TestService(unittest.TestCase):

    ERROR_LOGGING_LEVEL = 40000

    @utils.ignore_warnings
    def setUp(self):
        self.two_indices_rs = "two_indices_rs.json"
        self.index_created_rs = "index_created_rs.json"
        self.index_already_exists_rs = "index_already_exists_rs.json"
        self.index_deleted_rs = "index_deleted_rs.json"
        self.index_not_found_rs = "index_not_found_rs.json"
        self.launch_wo_test_items = "launch_wo_test_items.json"
        self.launch_w_test_items_wo_logs = "launch_w_test_items_wo_logs.json"
        self.launch_w_test_items_w_logs = "launch_w_test_items_w_logs.json"
        self.launch_w_test_items_w_empty_logs = "launch_w_test_items_w_empty_logs.json"
        self.launch_w_test_items_w_logs_to_be_merged =\
            "launch_w_test_items_w_logs_to_be_merged.json"
        self.index_logs_rq = "index_logs_rq.json"
        self.index_logs_rq_big_messages = "index_logs_rq_big_messages.json"
        self.index_logs_rs = "index_logs_rs.json"
        self.search_rq_first = "search_rq_first.json"
        self.search_rq_second = "search_rq_second.json"
        self.search_rq_third = "search_rq_third.json"
        self.search_rq_filtered = "search_rq_filtered.json"
        self.search_rq_another_log = "search_rq_another_log.json"
        self.search_rq_different_logs = "search_rq_different_logs.json"
        self.search_rq_to_be_merged = "search_rq_to_be_merged.json"
        self.no_hits_search_rs = "no_hits_search_rs.json"
        self.one_hit_search_rs = "one_hit_search_rs.json"
        self.one_hit_search_rs_search_logs = "one_hit_search_rs_search_logs.json"
        self.two_hits_search_rs = "two_hits_search_rs.json"
        self.two_hits_search_rs_search_logs = "two_hits_search_rs_search_logs.json"
        self.three_hits_search_rs = "three_hits_search_rs.json"
        self.launch_w_test_items_w_logs_different_log_level =\
            "launch_w_test_items_w_logs_different_log_level.json"
        self.index_logs_rq_different_log_level = "index_logs_rq_different_log_level.json"
        self.index_logs_rq_different_log_level_merged =\
            "index_logs_rq_different_log_level_merged.json"
        self.index_logs_rs_different_log_level = "index_logs_rs_different_log_level.json"
        self.delete_logs_rs = "delete_logs_rs.json"
        self.two_hits_search_with_big_messages_rs = "two_hits_search_with_big_messages_rs.json"
        self.search_not_merged_logs_for_delete = "search_not_merged_logs_for_delete.json"
        self.search_merged_logs = "search_merged_logs.json"
        self.search_not_merged_logs = "search_not_merged_logs.json"
        self.search_logs_rq = "search_logs_rq.json"
        self.search_logs_rq_not_found = "search_logs_rq_not_found.json"
        self.index_logs_rq_merged_logs = "index_logs_rq_merged_logs.json"
        self.suggest_test_item_info_w_logs = "suggest_test_item_info_w_logs.json"
        self.three_hits_search_rs_with_duplicate = "three_hits_search_rs_with_duplicate.json"
        self.one_hit_search_rs_merged = "one_hit_search_rs_merged.json"
        self.search_rq_merged_first = "search_rq_merged_first.json"
        self.search_rq_merged_second = "search_rq_merged_second.json"
        self.search_rq_merged_third = "search_rq_merged_third.json"
        self.suggest_test_item_info_w_merged_logs = "suggest_test_item_info_w_merged_logs.json"
        self.one_hit_search_rs_merged_wrong = "one_hit_search_rs_merged_wrong.json"
        self.three_hits_search_rs_with_one_unique_id = "three_hits_search_rs_with_one_unique_id.json"
        self.launch_w_items_clustering = "launch_w_items_clustering.json"
        self.cluster_update_all_the_same = "cluster_update_all_the_same.json"
        self.search_logs_rq_first_group = "search_logs_rq_first_group.json"
        self.search_logs_rq_second_group = "search_logs_rq_second_group.json"
        self.one_hit_search_rs_clustering = "one_hit_search_rs_clustering.json"
        self.search_logs_rq_first_group_2lines = "search_logs_rq_first_group_2lines.json"
        self.cluster_update_es_update = "cluster_update_es_update.json"
        self.cluster_update_all_the_same_es_update = "cluster_update_all_the_same_es_update.json"
        self.cluster_update = "cluster_update.json"
        self.app_config = {
            "esHost": "http://localhost:9200",
            "esVerifyCerts":     False,
            "esUseSsl":          False,
            "esSslShowWarn":     False,
            "turnOffSslVerification": True,
            "esCAcert":          "",
            "esClientCert":      "",
            "esClientKey":       "",
            "appVersion":        "",
            "minioRegion":       "",
            "minioBucketPrefix": "",
            "filesystemDefaultPath": "",
            "esChunkNumber":     1000,
            "minioHost":         "",
            "minioAccessKey":    "",
            "minioSecretKey":    ""
        }
        self.model_settings = utils.read_json_file("", "model_settings.json", to_json=True)
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def get_default_search_config(self):
        """Get default search config"""
        return {
            "MinShouldMatch": "80%",
            "MinTermFreq":    1,
            "MinDocFreq":     1,
            "BoostAA": -2,
            "BoostLaunch":    2,
            "BoostUniqueID":  2,
            "MaxQueryTerms":  50,
            "SearchLogsMinShouldMatch": "98%",
            "SearchLogsMinSimilarity": 0.9,
            "MinWordLength":  0,
            "PatternLabelMinPercentToSuggest": 0.5,
            "PatternLabelMinCountToSuggest":   5,
            "PatternMinCountToSuggest":        10,
            "BoostModelFolder":
                self.model_settings["BOOST_MODEL_FOLDER"],
            "SimilarityWeightsFolder":
                self.model_settings["SIMILARITY_WEIGHTS_FOLDER"],
            "SuggestBoostModelFolder":
                self.model_settings["SUGGEST_BOOST_MODEL_FOLDER"],
            "GlobalDefectTypeModelFolder":
                self.model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"],
            "ProbabilityForCustomModelSuggestions": 0.9
        }

    @utils.ignore_warnings
    def _start_server(self, test_calls):
        httpretty.reset()
        httpretty.enable(allow_net_connect=False)
        for test_info in test_calls:
            if "content_type" in test_info:
                httpretty.register_uri(
                    test_info["method"],
                    self.app_config["esHost"] + test_info["uri"],
                    body=test_info["rs"] if "rs" in test_info else "",
                    status=test_info["status"],
                    content_type=test_info["content_type"],
                )
            else:
                httpretty.register_uri(
                    test_info["method"],
                    self.app_config["esHost"] + test_info["uri"],
                    body=test_info["rs"] if "rs" in test_info else "",
                    status=test_info["status"],
                )

    @staticmethod
    @utils.ignore_warnings
    def shutdown_server(test_calls):
        """Shutdown server and test request calls"""
        httpretty.latest_requests().should.have.length_of(len(test_calls))
        for expected_test_call, test_call in zip(test_calls, httpretty.latest_requests()):
            expected_test_call["method"].should.equal(test_call.method)
            expected_test_call["uri"].should.equal(test_call.path)
            if "rq" in expected_test_call:
                expected_body = expected_test_call["rq"]
                real_body = test_call.parse_request_body(test_call.body)
                if type(expected_body) == str and type(real_body) != str:
                    expected_body = json.loads(expected_body)
                expected_body.should.equal(real_body)
        httpretty.disable()
        httpretty.reset()
