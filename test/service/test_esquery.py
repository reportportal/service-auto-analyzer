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

from app.commons import model_chooser
from app.commons.model import launch_objects
from app.commons.model.launch_objects import SearchConfig
from app.service import AutoAnalyzerService, SearchService, SuggestService
from app.utils import utils
from test import get_fixture
from test.service import (
    get_base_log_dict,
    get_extended_log_dict,
    get_launch_object,
    get_search_logs_object,
    get_test_item_info_for_suggest,
)


class TestEsQuery(unittest.TestCase):
    """Tests building analyze query"""

    model_settings: dict
    app_config: launch_objects.ApplicationConfig

    @utils.ignore_warnings
    def setUp(self):
        self.query_all_logs_empty_stacktrace = "query_all_logs_empty_stacktrace.json"
        self.query_two_log_lines = "query_two_log_lines.json"
        self.query_all_logs_nonempty_stacktrace = "query_all_logs_nonempty_stacktrace.json"
        self.query_merged_small_logs_search = "query_merged_small_logs_search.json"
        self.query_search_logs = "query_search_logs.json"
        self.query_two_log_lines_only_current_launch = "query_two_log_lines_only_current_launch.json"
        self.query_two_log_lines_only_current_launch_wo_exceptions = (
            "query_two_log_lines_only_current_launch_wo_exceptions.json"
        )
        self.query_all_logs_nonempty_stacktrace_launches_with_the_same_name = (
            "query_all_logs_nonempty_stacktrace_launches_with_the_same_name.json"
        )
        self.suggest_query_all_logs_empty_stacktrace = "suggest_query_all_logs_empty_stacktrace.json"
        self.suggest_query_two_log_lines = "suggest_query_two_log_lines.json"
        self.suggest_query_all_logs_nonempty_stacktrace = "suggest_query_all_logs_nonempty_stacktrace.json"
        self.suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name = (
            "suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name.json"
        )
        self.suggest_query_merged_small_logs_search = "suggest_query_merged_small_logs_search.json"
        self.query_analyze_items_including_no_defect = "query_analyze_items_including_no_defect.json"
        self.query_analyze_items_including_no_defect_small_logs = (
            "query_analyze_items_including_no_defect_small_logs.json"
        )
        self.app_config = launch_objects.ApplicationConfig(
            esHost="http://localhost:9200",
            esUser="",
            esPassword="",
            esVerifyCerts=False,
            esUseSsl=False,
            esSslShowWarn=False,
            turnOffSslVerification=True,
            esCAcert="",
            esClientCert="",
            esClientKey="",
            appVersion="",
            esChunkNumber=1000,
            binaryStoreType="filesystem",
            filesystemDefaultPath="",
        )
        model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
        if model_settings and isinstance(model_settings, dict):
            self.model_settings = model_settings
        else:
            raise RuntimeError("Failed to read model settings")
        self.model_chooser = model_chooser.ModelChooser(self.app_config, self.get_default_search_config())
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def get_default_search_config(self) -> SearchConfig:
        """Get default search config"""
        return SearchConfig(
            MinShouldMatch="80%",
            BoostAA=-10,
            BoostLaunch=5,
            BoostTestCaseHash=3,
            MaxQueryTerms=50,
            SearchLogsMinSimilarity=0.9,
            MinWordLength=0,
            BoostModelFolder=self.model_settings["BOOST_MODEL_FOLDER"],
            SimilarityWeightsFolder=self.model_settings["SIMILARITY_WEIGHTS_FOLDER"],
            SuggestBoostModelFolder=self.model_settings["SUGGEST_BOOST_MODEL_FOLDER"],
            GlobalDefectTypeModelFolder=self.model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"],
            TimeWeightDecay=0.95,
        )

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_empty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object()
        log = get_base_log_dict()

        query_from_service = AutoAnalyzerService(self.model_chooser, self.app_config, search_cfg).build_analyze_query(
            launch, log
        )
        demo_query = get_fixture(self.query_all_logs_empty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object(number_of_log_lines=2)
        log = get_base_log_dict()

        query_from_service = AutoAnalyzerService(self.model_chooser, self.app_config, search_cfg).build_analyze_query(
            launch, log
        )
        demo_query = get_fixture(self.query_two_log_lines, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines_only_current_launch(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object(analyzer_mode="CURRENT_LAUNCH", number_of_log_lines=2)
        log = get_base_log_dict(found_tests_and_methods="FindAllMessagesTest.findMessage")

        query_from_service = AutoAnalyzerService(self.model_chooser, self.app_config, search_cfg).build_analyze_query(
            launch, log
        )
        demo_query = get_fixture(self.query_two_log_lines_only_current_launch, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_two_log_lines_only_current_launch_wo_exceptions(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object(analyzer_mode="CURRENT_LAUNCH", number_of_log_lines=2)
        log = get_base_log_dict(found_exceptions="")

        query_from_service = AutoAnalyzerService(self.model_chooser, self.app_config, search_cfg).build_analyze_query(
            launch, log
        )
        demo_query = get_fixture(self.query_two_log_lines_only_current_launch_wo_exceptions, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_nonempty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object()
        log = get_base_log_dict(stacktrace="invoke.method(arg)")

        query_from_service = AutoAnalyzerService(self.model_chooser, self.app_config, search_cfg).build_analyze_query(
            launch, log
        )
        demo_query = get_fixture(self.query_all_logs_nonempty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_all_logs_nonempty_stacktrace_launches_with_the_same_name(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object(analyzer_mode="LAUNCH_NAME")
        log = get_base_log_dict(stacktrace="invoke.method(arg)", potential_status_codes="300 401")

        query_from_service = AutoAnalyzerService(self.model_chooser, self.app_config, search_cfg).build_analyze_query(
            launch, log
        )
        demo_query = get_fixture(self.query_all_logs_nonempty_stacktrace_launches_with_the_same_name, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_analyze_query_merged_small_logs_search(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object()
        log = get_base_log_dict(
            message="",
            merged_small_logs="hello world",
            detected_message="",
            detected_message_with_numbers="",
            only_numbers="",
        )

        query_from_service = AutoAnalyzerService(self.model_chooser, self.app_config, search_cfg).build_analyze_query(
            launch, log
        )
        demo_query = get_fixture(self.query_merged_small_logs_search, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_search_query(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        search_req = get_search_logs_object()
        log = get_extended_log_dict(
            potential_status_codes="300 500", found_tests_and_methods="FindAllMessagesTest.findMessage"
        )

        query_from_service = SearchService(self.app_config, search_cfg).build_search_query(search_req, log)
        demo_query = get_fixture(self.query_search_logs, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_empty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        test_item_info = get_test_item_info_for_suggest()
        log = get_extended_log_dict(found_tests_and_methods="FindAllMessagesTest.findMessage")

        query_from_service = SuggestService(self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info,
            log,
            message_field="message_extended",
            det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended",
        )
        demo_query = get_fixture(self.suggest_query_all_logs_empty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_two_log_lines(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        test_item_info = get_test_item_info_for_suggest(number_of_log_lines=2)
        log = get_extended_log_dict(potential_status_codes="400 200")

        query_from_service = SuggestService(self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info,
            log,
            message_field="message_extended",
            det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended",
        )
        demo_query = get_fixture(self.suggest_query_two_log_lines, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_nonempty_stacktrace(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        test_item_info = get_test_item_info_for_suggest()
        log = get_extended_log_dict(stacktrace="invoke.method(arg)", stacktrace_extended="invoke.method(arg)")

        query_from_service = SuggestService(self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info,
            log,
            message_field="message_extended",
            det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended",
        )
        demo_query = get_fixture(self.suggest_query_all_logs_nonempty_stacktrace, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        test_item_info = get_test_item_info_for_suggest(analyzer_mode="LAUNCH_NAME")
        log = get_extended_log_dict(
            stacktrace="invoke.method(arg)", stacktrace_extended="invoke.method(arg)", potential_status_codes="200 401"
        )

        query_from_service = SuggestService(self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info,
            log,
            message_field="message_without_params_extended",
            det_mes_field="detected_message_without_params_extended",
            stacktrace_field="stacktrace_extended",
        )
        demo_query = get_fixture(
            self.suggest_query_all_logs_nonempty_stacktrace_launches_with_the_same_name, to_json=True
        )

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_suggest_query_merged_small_logs_search(self):
        """Tests building analyze query"""
        search_cfg = self.get_default_search_config()
        test_item_info = get_test_item_info_for_suggest()
        log = get_extended_log_dict(
            message="",
            merged_small_logs="hello world",
            detected_message="",
            detected_message_with_numbers="",
            message_params="",
            message_without_params_extended="",
            message_extended="",
            detected_message_extended="",
            potential_status_codes="200 400",
            found_tests_and_methods="FindAllMessagesTest.findMessage",
        )
        # Override only_numbers to be empty for this test
        log["_source"]["only_numbers"] = ""

        query_from_service = SuggestService(self.model_chooser, self.app_config, search_cfg).build_suggest_query(
            test_item_info,
            log,
            message_field="message_extended",
            det_mes_field="detected_message_extended",
            stacktrace_field="stacktrace_extended",
        )
        demo_query = get_fixture(self.suggest_query_merged_small_logs_search, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_query_with_no_defect(self):
        """Tests building analyze query with finding No defect"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object(analyzer_mode="LAUNCH_NAME")
        log = get_base_log_dict(stacktrace="invoke.method(arg)", potential_status_codes="300 401")

        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg
        ).build_query_with_no_defect(launch, log)
        demo_query = get_fixture(self.query_analyze_items_including_no_defect, to_json=True)

        assert query_from_service == demo_query

    @utils.ignore_warnings
    def test_build_query_with_no_defect_small_logs(self):
        """Tests building analyze query with finding No defect for small logs"""
        search_cfg = self.get_default_search_config()
        launch = get_launch_object(analyzer_mode="LAUNCH_NAME")
        log = get_base_log_dict(
            message="",
            merged_small_logs="hello world",
            detected_message="",
            detected_message_with_numbers="",
            found_exceptions="",
            potential_status_codes="300 401",
        )

        query_from_service = AutoAnalyzerService(
            self.model_chooser, self.app_config, search_cfg
        ).build_query_with_no_defect(launch, log)
        demo_query = get_fixture(self.query_analyze_items_including_no_defect_small_logs, to_json=True)

        assert query_from_service == demo_query


if __name__ == "__main__":
    unittest.main()
