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
from commons.es_query_builder import EsQueryBuilder
from service.search_service import SearchService
from utils import utils
import os


class TestEsQuery(unittest.TestCase):
    """Tests building analyze query"""
    @utils.ignore_warnings
    def setUp(self):
        self.query_all_logs_empty_stacktrace = "query_all_logs_empty_stacktrace.json"
        self.query_two_log_lines = "query_two_log_lines.json"
        self.query_all_logs_nonempty_stacktrace = "query_all_logs_nonempty_stacktrace.json"
        self.query_merged_small_logs_search = "query_merged_small_logs_search.json"
        self.query_search_logs = "query_search_logs.json"
        self.query_two_log_lines_only_current_launch = "query_two_log_lines_only_current_launch.json"
        self.query_two_log_lines_only_current_launch_wo_exceptions =\
            "query_two_log_lines_only_current_launch_wo_exceptions.json"
        self.query_all_logs_nonempty_stacktrace_launches_with_the_same_name =\
            "query_all_logs_nonempty_stacktrace_launches_with_the_same_name.json"
        self.suggest_query_all_logs_empty_stacktrace = "suggest_query_all_logs_empty_stacktrace.json"
        self.suggest_query_two_log_lines = "suggest_query_two_log_lines.json"
        self.suggest_query_all_logs_nonempty_stacktrace =\
            "suggest_query_all_logs_nonempty_stacktrace.json"
        self.suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name =\
            "suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name.json"
        self.suggest_query_merged_small_logs_search = "suggest_query_merged_small_logs_search.json"
        self.app_config = {
            "esHost": "http://localhost:9200",
            "esVerifyCerts":     False,
            "esUseSsl":          False,
            "esSslShowWarn":     False,
            "esCAcert":          "",
            "esClientCert":      "",
            "esClientKey":       "",
            "appVersion":        ""
        }
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @staticmethod
    @utils.ignore_warnings
    def get_default_search_config():
        """Get default search config"""
        return {
            "MinShouldMatch": "80%",
            "MinTermFreq":    1,
            "MinDocFreq":     1,
            "BoostAA": -10,
            "BoostLaunch":    5,
            "BoostUniqueID":  3,
            "MaxQueryTerms":  50,
            "SearchLogsMinShouldMatch": "90%",
            "SearchLogsMinSimilarity": 0.9,
            "MinWordLength":  0,
            "BoostModelFolderAllLines":    os.getenv("BOOST_MODEL_FOLDER_ALL_LINES", ""),
            "BoostModelFolderNotAllLines": os.getenv("BOOST_MODEL_FOLDER_NOT_ALL_LINES", ""),
            "SimilarityWeightsFolder":     os.getenv("SIMILARITY_WEIGHTS_FOLDER", "")
        }

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_empty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world",
                "merged_small_logs":  "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": ""}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_analyze_query(launch, log)
        demo_query = utils.get_fixture(self.query_all_logs_empty_stacktrace, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world",
                "merged_small_logs":  "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": ""}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_analyze_query(launch, log)
        demo_query = utils.get_fixture(self.query_two_log_lines, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines_only_current_launch(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "CURRENT_LAUNCH", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world",
                "merged_small_logs":  "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": ""}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_analyze_query(launch, log)
        demo_query = utils.get_fixture(
            self.query_two_log_lines_only_current_launch, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines_only_current_launch_wo_exceptions(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "CURRENT_LAUNCH", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world",
                "merged_small_logs":  "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "",
                "potential_status_codes": ""}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_analyze_query(launch, log)
        demo_query = utils.get_fixture(
            self.query_two_log_lines_only_current_launch_wo_exceptions, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_nonempty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world",
                "merged_small_logs":  "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "invoke.method(arg)",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": ""}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_analyze_query(launch, log)
        demo_query = utils.get_fixture(self.query_all_logs_nonempty_stacktrace, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_nonempty_stacktrace_launches_with_the_same_name(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "LAUNCH_NAME", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world",
                "merged_small_logs":  "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "invoke.method(arg)",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "300 401"}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_analyze_query(launch, log)
        demo_query = utils.get_fixture(
            self.query_all_logs_nonempty_stacktrace_launches_with_the_same_name, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_analyze_query_merged_small_logs_search(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "",
                "merged_small_logs":  "hello world",
                "detected_message": "",
                "detected_message_with_numbers": "",
                "stacktrace": "",
                "only_numbers": "",
                "found_exceptions": "AssertionError",
                "potential_status_codes": ""}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_analyze_query(launch, log)
        demo_query = utils.get_fixture(self.query_merged_small_logs_search, to_json=True)

        query_from_esclient.should.equal(demo_query)

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
                        {"term": {"is_merged": False}},
                    ],
                    "must_not": [
                        {"wildcard": {"issue_type": "TI*"}},
                        {"wildcard": {"issue_type": "ti*"}},
                        {"wildcard": {"issue_type": "nd*"}},
                        {"wildcard": {"issue_type": "ND*"}},
                        {"term": {"test_item": log["_source"]["test_item"]}}
                    ],
                    "must": [
                        {"more_like_this": {
                            "fields":               ["detected_message"],
                            "like":                 log["_source"]["detected_message"],
                            "min_doc_freq":         1,
                            "min_term_freq":        1,
                            "minimum_should_match": "5<" + search_cfg["MinShouldMatch"],
                            "max_query_terms":      search_cfg["MaxQueryTerms"],
                            "boost":                4.0,
                        }, },
                        {"more_like_this": {
                            "fields":               ["stacktrace"],
                            "like":                 log["_source"]["stacktrace"],
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
                            "fields":               ["only_numbers"],
                            "like":                 log["_source"]["only_numbers"],
                            "min_doc_freq":         1,
                            "min_term_freq":        1,
                            "minimum_should_match": "1",
                            "max_query_terms":      search_cfg["MaxQueryTerms"],
                            "boost":                4.0,
                        }},
                        {"more_like_this": {
                            "fields":               ["potential_status_codes"],
                            "like":                 log["_source"]["potential_status_codes"],
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
        search_cfg = TestEsQuery.get_default_search_config()

        search_req = launch_objects.SearchLogs(**{
            "launchId": 1,
            "launchName": "launch 1",
            "itemId": 2,
            "projectId": 3,
            "filteredLaunchIds": [1, 2, 3],
            "logMessages": ["log message 1"],
            "logLines": -1})
        query_from_service = SearchService(self.app_config, search_cfg).build_search_query(
            search_req, "log message 1")
        demo_query = utils.get_fixture(self.query_search_logs, to_json=True)

        query_from_service.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_empty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world 'sdf'",
                "merged_small_logs":  "",
                "detected_message": "hello world 'sdf'",
                "detected_message_with_numbers": "hello world 1 'sdf'",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "found_exceptions_extended": "AssertionError",
                "message_params": "sdf",
                "urls": "",
                "paths": "",
                "message_without_params_extended": "hello world",
                "detected_message_without_params_extended": "hello world",
                "stacktrace_extended": "",
                "message_extended": "hello world 'sdf'",
                "detected_message_extended": "hello world 'sdf'",
                "potential_status_codes": ""
            }}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = utils.get_fixture(self.suggest_query_all_logs_empty_stacktrace, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_suggest_query_two_log_lines(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world 'sdf'",
                "merged_small_logs":  "",
                "detected_message": "hello world 'sdf'",
                "detected_message_with_numbers": "hello world 1 'sdf'",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "found_exceptions_extended": "AssertionError",
                "message_params": "sdf",
                "urls": "",
                "paths": "",
                "message_without_params_extended": "hello world",
                "detected_message_without_params_extended": "hello world",
                "stacktrace_extended": "",
                "message_extended": "hello world 'sdf'",
                "detected_message_extended": "hello world 'sdf'",
                "potential_status_codes": "400 200"
            }}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = utils.get_fixture(self.suggest_query_two_log_lines, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_nonempty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world 'sdf'",
                "merged_small_logs":  "",
                "detected_message": "hello world 'sdf'",
                "detected_message_with_numbers": "hello world 1 'sdf'",
                "stacktrace": "invoke.method(arg)",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "found_exceptions_extended": "AssertionError",
                "message_params": "sdf",
                "urls": "",
                "paths": "",
                "message_without_params_extended": "hello world",
                "detected_message_without_params_extended": "hello world",
                "stacktrace_extended": "invoke.method(arg)",
                "message_extended": "hello world 'sdf'",
                "detected_message_extended": "hello world 'sdf'",
                "potential_status_codes": ""
            }}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = utils.get_fixture(self.suggest_query_all_logs_nonempty_stacktrace, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "LAUNCH_NAME", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "hello world 'sdf'",
                "merged_small_logs":  "",
                "detected_message": "hello world 'sdf'",
                "detected_message_with_numbers": "hello world 1 'sdf'",
                "stacktrace": "invoke.method(arg)",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "found_exceptions_extended": "AssertionError",
                "message_params": "sdf",
                "urls": "",
                "paths": "",
                "message_without_params_extended": "hello world",
                "detected_message_without_params_extended": "hello world",
                "stacktrace_extended": "invoke.method(arg)",
                "message_extended": "hello world 'sdf'",
                "detected_message_extended": "hello world 'sdf'",
                "potential_status_codes": "200 401"
            }}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_suggest_query(
            test_item_info, log,
            message_field="message_without_params_extended",
            det_mes_field="detected_message_without_params_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = utils.get_fixture(
            self.suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name, to_json=True)

        query_from_esclient.should.equal(demo_query)

    @utils.ignore_warnings
    def test_build_suggest_query_merged_small_logs_search(self):
        """Tests building analyze query"""
        search_cfg = TestEsQuery.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id":    1,
            "_index": 1,
            "_source": {
                "unique_id":        "unique",
                "test_case_hash":   1,
                "test_item":        "123",
                "message":          "",
                "merged_small_logs":  "hello world",
                "detected_message": "",
                "detected_message_with_numbers": "",
                "stacktrace": "",
                "only_numbers": "",
                "found_exceptions": "AssertionError",
                "found_exceptions_extended": "AssertionError",
                "message_params": "",
                "urls": "",
                "paths": "",
                "message_without_params_extended": "",
                "detected_message_without_params_extended": "",
                "stacktrace_extended": "",
                "message_extended": "",
                "detected_message_extended": "",
                "potential_status_codes": "200 400"}}
        query_from_esclient = EsQueryBuilder(search_cfg, 40000).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = utils.get_fixture(self.suggest_query_merged_small_logs_search, to_json=True)

        query_from_esclient.should.equal(demo_query)


if __name__ == '__main__':
    unittest.main()
