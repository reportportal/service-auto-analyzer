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
from typing import Any

from app.commons import launch_objects
from app.commons import model_chooser
from app.commons.launch_objects import SearchConfig
from app.service import AutoAnalyzerService
from app.service import SearchService
from app.service import SuggestService
from app.utils import utils
from test import get_fixture


class TestEsQuery(unittest.TestCase):
    """Tests building analyze query"""
    model_settings: dict
    app_config: dict[str, Any]

    @utils.ignore_warnings
    def setUp(self):
        self.query_all_logs_empty_stacktrace = "query_all_logs_empty_stacktrace.json"
        self.query_two_log_lines = "query_two_log_lines.json"
        self.query_all_logs_nonempty_stacktrace = "query_all_logs_nonempty_stacktrace.json"
        self.query_merged_small_logs_search = "query_merged_small_logs_search.json"
        self.query_search_logs = "query_search_logs.json"
        self.query_two_log_lines_only_current_launch = "query_two_log_lines_only_current_launch.json"
        self.query_two_log_lines_only_current_launch_wo_exceptions = \
            "query_two_log_lines_only_current_launch_wo_exceptions.json"
        self.query_all_logs_nonempty_stacktrace_launches_with_the_same_name = \
            "query_all_logs_nonempty_stacktrace_launches_with_the_same_name.json"
        self.suggest_query_all_logs_empty_stacktrace = "suggest_query_all_logs_empty_stacktrace.json"
        self.suggest_query_two_log_lines = "suggest_query_two_log_lines.json"
        self.suggest_query_all_logs_nonempty_stacktrace = \
            "suggest_query_all_logs_nonempty_stacktrace.json"
        self.suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name = \
            "suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name.json"
        self.suggest_query_merged_small_logs_search = "suggest_query_merged_small_logs_search.json"
        self.query_analyze_items_including_no_defect = "query_analyze_items_including_no_defect.json"
        self.query_analyze_items_including_no_defect_small_logs = \
            "query_analyze_items_including_no_defect_small_logs.json"
        self.app_config = {
            "esHost": "http://localhost:9200",
            "esUser": "",
            "esPassword": "",
            "esVerifyCerts": False,
            "esUseSsl": False,
            "esSslShowWarn": False,
            "turnOffSslVerification": True,
            "esCAcert": "",
            "esClientCert": "",
            "esClientKey": "",
            "appVersion": "",
            "esChunkNumber": 1000,
            'binaryStoreType': 'filesystem',
            'filesystemDefaultPath': ''
        }
        model_settings = utils.read_json_file('res', 'model_settings.json', to_json=True)
        if model_settings and isinstance(model_settings, dict):
            self.model_settings = model_settings
        else:
            raise RuntimeError('Failed to read model settings')
        self.model_chooser = model_chooser.ModelChooser(self.app_config, self.get_default_search_config())
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def get_default_search_config(self) -> SearchConfig:
        """Get default search config"""
        return SearchConfig(
            MinShouldMatch='80%',
            BoostAA=-10,
            BoostLaunch=5,
            BoostTestCaseHash=3,
            MaxQueryTerms=50,
            SearchLogsMinSimilarity=0.9,
            MinWordLength=0,
            BoostModelFolder=self.model_settings['BOOST_MODEL_FOLDER'],
            SimilarityWeightsFolder=self.model_settings['SIMILARITY_WEIGHTS_FOLDER'],
            SuggestBoostModelFolder=self.model_settings['SUGGEST_BOOST_MODEL_FOLDER'],
            GlobalDefectTypeModelFolder=self.model_settings['GLOBAL_DEFECT_TYPE_MODEL_FOLDER'],
            TimeWeightDecay=0.95,
            RetrainSuggestBoostModelConfig=self.model_settings['RETRAIN_SUGGEST_BOOST_MODEL_CONFIG'],
            RetrainAutoBoostModelConfig=self.model_settings['RETRAIN_AUTO_BOOST_MODEL_CONFIG']
        )

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_empty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world",
                "merged_small_logs": "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_analyze_query(launch, log)
        demo_query = get_fixture(self.query_all_logs_empty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world",
                "merged_small_logs": "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_analyze_query(launch, log)
        demo_query = get_fixture(self.query_two_log_lines, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines_only_current_launch(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "CURRENT_LAUNCH", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world",
                "merged_small_logs": "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "",
                "found_tests_and_methods": "FindAllMessagesTest.findMessage"}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_analyze_query(launch, log)
        demo_query = get_fixture(
            self.query_two_log_lines_only_current_launch, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines_only_current_launch_wo_exceptions(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "CURRENT_LAUNCH", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world",
                "merged_small_logs": "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "",
                "potential_status_codes": "",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_analyze_query(launch, log)
        demo_query = get_fixture(
            self.query_two_log_lines_only_current_launch_wo_exceptions, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_nonempty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world",
                "merged_small_logs": "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "invoke.method(arg)",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_analyze_query(launch, log)
        demo_query = get_fixture(self.query_all_logs_nonempty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_nonempty_stacktrace_launches_with_the_same_name(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "LAUNCH_NAME", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world",
                "merged_small_logs": "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "invoke.method(arg)",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "300 401",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_analyze_query(launch, log)
        demo_query = get_fixture(
            self.query_all_logs_nonempty_stacktrace_launches_with_the_same_name, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_merged_small_logs_search(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "",
                "merged_small_logs": "hello world",
                "detected_message": "",
                "detected_message_with_numbers": "",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "",
                "only_numbers": "",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_analyze_query(launch, log)
        demo_query = get_fixture(self.query_merged_small_logs_search, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_search_query(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        search_req = launch_objects.SearchLogs(**{
            "launchId": 1,
            "launchName": "launch 1",
            "itemId": 2,
            "projectId": 3,
            "filteredLaunchIds": [1, 2, 3],
            "logMessages": ["log message 1"],
            "logLines": -1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world 'sdf'",
                "merged_small_logs": "",
                "detected_message": "hello world 'sdf'",
                "detected_message_with_numbers": "hello world 1 'sdf'",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "found_exceptions_extended": "AssertionError",
                "message_params": "sdf",
                "urls": "",
                "paths": "",
                "message_without_params_extended": "hello world",
                "stacktrace_extended": "",
                "message_extended": "hello world 'sdf'",
                "detected_message_extended": "hello world 'sdf'",
                "potential_status_codes": "300 500",
                "found_tests_and_methods": "FindAllMessagesTest.findMessage"
            }}
        query_from_service = SearchService(self.app_config, search_cfg).build_search_query(
            search_req, log)
        demo_query = get_fixture(self.query_search_logs, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_empty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "test_item_name": "test item Common Query",
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world 'sdf'",
                "merged_small_logs": "",
                "detected_message": "hello world 'sdf'",
                "detected_message_with_numbers": "hello world 1 'sdf'",
                "detected_message_without_params_extended": "hello world",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "found_exceptions_extended": "AssertionError",
                "message_params": "sdf",
                "urls": "",
                "paths": "",
                "message_without_params_extended": "hello world",
                "stacktrace_extended": "",
                "message_extended": "hello world 'sdf'",
                "detected_message_extended": "hello world 'sdf'",
                "potential_status_codes": "",
                "found_tests_and_methods": "FindAllMessagesTest.findMessage"
            }}
        query_from_service = SuggestService(
            self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = get_fixture(self.suggest_query_all_logs_empty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_two_log_lines(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": 2},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "test_item_name": "test item Common Query",
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world 'sdf'",
                "merged_small_logs": "",
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
                "potential_status_codes": "400 200",
                "found_tests_and_methods": ""
            }}
        query_from_service = SuggestService(
            self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = get_fixture(self.suggest_query_two_log_lines, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_nonempty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "test_item_name": "test item Common Query",
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world 'sdf'",
                "merged_small_logs": "",
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
                "potential_status_codes": "",
                "found_tests_and_methods": ""
            }}
        query_from_service = SuggestService(
            self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = get_fixture(self.suggest_query_all_logs_nonempty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "LAUNCH_NAME", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "test_item_name": "test item Common Query",
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world 'sdf'",
                "merged_small_logs": "",
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
                "potential_status_codes": "200 401",
                "found_tests_and_methods": ""
            }}
        query_from_service = SuggestService(
            self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info, log,
            message_field="message_without_params_extended",
            det_mes_field="detected_message_without_params_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = get_fixture(
            self.suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_merged_small_logs_search(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()

        test_item_info = launch_objects.TestItemInfo(**{
            "analyzerConfig": {"analyzerMode": "ALL", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1,
            "test_item_name": "test item Common Query",
            "testCaseHash": 1,
            "uniqueId": "unique",
            "testItemId": 2})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "",
                "merged_small_logs": "hello world",
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
                "detected_message_without_params_extended": "hello world",
                "stacktrace_extended": "",
                "message_extended": "",
                "detected_message_extended": "",
                "potential_status_codes": "200 400",
                "found_tests_and_methods": "FindAllMessagesTest.findMessage"}}
        query_from_service = SuggestService(
            self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info, log,
            message_field="message_extended", det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended")
        demo_query = get_fixture(self.suggest_query_merged_small_logs_search, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_query_with_no_defect(self):
        """Tests building analyze query with finding No defect"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "LAUNCH_NAME", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "hello world",
                "merged_small_logs": "",
                "detected_message": "hello world",
                "detected_message_with_numbers": "hello world 1",
                "stacktrace": "invoke.method(arg)",
                "only_numbers": "1",
                "found_exceptions": "AssertionError",
                "potential_status_codes": "300 401",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_query_with_no_defect(launch, log)
        demo_query = get_fixture(self.query_analyze_items_including_no_defect, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_query_with_no_defect_small_logs(self):
        """Tests building analyze query with finding No defect for small logs"""
        search_cfg = self.get_default_search_config()

        launch = launch_objects.Launch(**{
            "analyzerConfig": {"analyzerMode": "LAUNCH_NAME", "numberOfLogLines": -1},
            "launchId": 12,
            "launchName": "Launch name",
            "project": 1})
        log = {
            "_id": 1,
            "_index": 1,
            "_source": {
                "start_time": "2021-08-30 08:11:23",
                "unique_id": "unique",
                "test_case_hash": 1,
                "test_item": "123",
                "test_item_name": "test item Common Query",
                "message": "",
                "merged_small_logs": "hello world",
                "detected_message": "",
                "detected_message_with_numbers": "",
                "stacktrace": "",
                "only_numbers": "1",
                "found_exceptions": "",
                "potential_status_codes": "300 401",
                "found_tests_and_methods": ""}}
        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg).build_query_with_no_defect(launch, log)
        demo_query = get_fixture(
            self.query_analyze_items_including_no_defect_small_logs, to_json=True)

        assert query_from_service == demo_query


if __name__ == '__main__':
    unittest.main()
