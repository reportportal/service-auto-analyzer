#  Copyright 2025 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Unit tests for request_factory module."""

from app.commons.model.launch_objects import AnalyzerConf, Launch, Log, TestItem
from app.commons.request_factory import prepare_test_items


class TestPrepareTestItem:
    """Test suite for prepare_test_item function."""

    def test_prepare_test_item_basic(self):
        """Test basic item preparation with single log."""
        # Arrange
        log = Log(
            logId=1001,
            logLevel=40000,
            logTime=[2025, 1, 15, 22, 15, 31, 0],
            message="java.lang.NullPointerException: Cannot invoke method getUser()\n"
            "\tat com.example.UserService.getUser(UserService.java:42)",
        )

        test_item = TestItem(
            testItemId=12345,
            testItemName="test_user_login_with_valid_credentials",
            uniqueId="auto:abc123def456",
            testCaseHash=987654321,
            isAutoAnalyzed=False,
            issueType="PB001",
            startTime=[2025, 1, 15, 22, 15, 30, 0],
            logs=[log],
        )

        launch = Launch(
            launchId=100,
            project=1,
            launchName="nightly_regression",
            launchNumber=42,
            launchStartTime=[2025, 1, 15, 22, 0, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch)[0]

        # Assert - Test Item level fields
        assert result.test_item_id == "12345"
        assert result.test_item_name == "test user login valid credentials"
        assert result.unique_id == "auto:abc123def456"
        assert result.test_case_hash == 987654321
        assert result.launch_id == "100"
        assert result.launch_name == "nightly_regression"
        assert result.launch_number == "42"
        assert result.launch_start_time == "2025-01-15 22:00:00"
        assert result.is_auto_analyzed is False
        assert result.issue_type == "pb001"
        assert result.start_time == "2025-01-15 22:15:30"
        assert result.log_count == 1

        # Assert - Log level fields
        assert len(result.logs) == 1
        log_data = result.logs[0]
        assert log_data.log_id == "1001"
        assert log_data.log_order == 0
        assert log_data.log_time == "2025-01-15 22:15:31"
        assert log_data.log_level == 40000
        assert log_data.original_message == log.message
        assert "nullpointerexception" in log_data.message.lower()
        assert log_data.message_lines > 0
        assert log_data.message_words_number > 0

    def test_prepare_test_item_multiple_logs(self):
        """Test item preparation with multiple logs."""
        # Arrange
        logs = [
            Log(
                logId=1001,
                logLevel=40000,
                logTime=[2025, 1, 15, 22, 15, 31, 0],
                message="java.lang.NullPointerException: Cannot invoke method",
            ),
            Log(
                logId=1002,
                logLevel=40000,
                logTime=[2025, 1, 15, 22, 15, 32, 0],
                message="org.springframework.dao.DataAccessException: Connection refused",
            ),
            Log(
                logId=1003,
                logLevel=40000,
                logTime=[2025, 1, 15, 22, 15, 33, 0],
                message="java.net.ConnectException: Connection timeout",
            ),
        ]

        test_item = TestItem(
            testItemId=12345,
            testItemName="test_database_connection",
            uniqueId="auto:xyz789",
            testCaseHash=123456,
            isAutoAnalyzed=True,
            issueType="AB001",
            startTime=[2025, 1, 15, 22, 15, 30, 0],
            logs=logs,
        )

        launch = Launch(
            launchId=200,
            project=2,
            launchName="smoke_tests",
            launchNumber=10,
            launchStartTime=[2025, 1, 15, 22, 0, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch)[0]

        # Assert
        assert result.test_item_id == "12345"
        assert result.log_count == 3
        assert len(result.logs) == 3
        assert result.is_auto_analyzed is True
        assert result.issue_type == "ab001"

        # Verify log order
        for idx, log_data in enumerate(result.logs):
            assert log_data.log_order == idx
            assert log_data.log_id == str(logs[idx].logId)

    def test_prepare_test_item_with_cluster_info(self):
        """Test item preparation with clustered logs."""
        # Arrange
        log = Log(
            logId=2001,
            logLevel=40000,
            logTime=[2025, 1, 16, 10, 30, 0, 0],
            message="Error occurred in processing",
            clusterId=101,
            clusterMessage="Error occurred in processing",
        )

        test_item = TestItem(
            testItemId=54321,
            testItemName="test_with_cluster",
            uniqueId="auto:cluster123",
            testCaseHash=111222,
            isAutoAnalyzed=False,
            issueType="SI001",
            startTime=[2025, 1, 16, 10, 29, 0, 0],
            logs=[log],
        )

        launch = Launch(
            launchId=300,
            project=3,
            launchName="regression",
            launchNumber=5,
            launchStartTime=[2025, 1, 16, 10, 0, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch)[0]

        # Assert
        assert result.test_item_id == "54321"
        assert result.issue_type == "si001"
        log_data = result.logs[0]
        assert log_data.cluster_id == "101"
        assert log_data.cluster_message == "Error occurred in processing"
        assert isinstance(log_data.cluster_with_numbers, bool)

    def test_prepare_test_item_empty_logs(self):
        """Test item preparation with no logs."""
        # Arrange
        test_item = TestItem(
            testItemId=99999,
            testItemName="test_no_logs",
            uniqueId="auto:nologs",
            testCaseHash=999,
            isAutoAnalyzed=False,
            issueType="PB001",
            startTime=[2025, 1, 16, 12, 0, 0, 0],
            logs=[],
        )

        launch = Launch(
            launchId=400,
            project=4,
            launchName="unit_tests",
            launchNumber=1,
            launchStartTime=[2025, 1, 16, 11, 0, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch)[0]

        # Assert
        assert result.test_item_id == "99999"
        assert result.log_count == 0
        assert len(result.logs) == 0

    def test_prepare_test_item_text_processing(self):
        """Test that text processing is applied correctly."""
        # Arrange
        log = Log(
            logId=3001,
            logLevel=40000,
            logTime=[2025, 1, 17, 14, 0, 0, 0],
            message="java.lang.NullPointerException: NPE at line 123\n"
            "\tat com.example.TestClass.testMethod(TestClass.java:123)\n"
            "\tat com.example.Runner.run(Runner.java:456)",
        )

        test_item = TestItem(
            testItemId=77777,
            testItemName="Test_With_Underscores_And_CamelCase",
            uniqueId="auto:textproc",
            testCaseHash=777,
            isAutoAnalyzed=False,
            issueType="AB001",
            startTime=[2025, 1, 17, 13, 59, 0, 0],
            logs=[log],
        )

        launch = Launch(
            launchId=500,
            project=5,
            launchName="integration",
            launchNumber=20,
            launchStartTime=[2025, 1, 17, 13, 0, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=5),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch)[0]

        # Assert
        log_data = result.logs[0]
        # Verify test item name is preprocessed
        assert "with underscores and camel case" in result.test_item_name.lower()

        # Verify message fields are populated
        assert log_data.message != ""
        assert log_data.detected_message != ""
        assert log_data.stacktrace != ""
        assert log_data.found_exceptions != ""
        assert "nullpointerexception" in log_data.found_exceptions.lower()

        # Verify extended fields are populated
        assert log_data.message_extended != ""
        assert log_data.stacktrace_extended != ""
        assert log_data.detected_message_extended != ""

        # Verify whole_message combines exception and stacktrace
        assert log_data.whole_message != ""

    def test_prepare_test_item_to_dict(self):
        """Test that TestItemIndexData can be converted to dict for indexing."""
        # Arrange
        log = Log(
            logId=4001,
            logLevel=40000,
            logTime=[2025, 1, 18, 9, 0, 0, 0],
            message="Error message",
        )

        test_item = TestItem(
            testItemId=88888,
            testItemName="test_to_dict",
            uniqueId="auto:dict",
            testCaseHash=888,
            isAutoAnalyzed=True,
            issueType="PB001",
            startTime=[2025, 1, 18, 8, 59, 0, 0],
            logs=[log],
        )

        launch = Launch(
            launchId=600,
            project=6,
            launchName="daily",
            launchNumber=30,
            launchStartTime=[2025, 1, 18, 8, 0, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch)[0]
        result_dict = result.to_index_dict()

        # Assert
        assert isinstance(result_dict, dict)
        assert result_dict["test_item_id"] == "88888"
        assert result_dict["log_count"] == 1
        assert "logs" in result_dict
        assert isinstance(result_dict["logs"], list)
        assert len(result_dict["logs"]) == 1
        assert result_dict["logs"][0]["log_id"] == "4001"

    def test_prepare_multiple_test_items(self):
        """Test processing multiple test items within a single launch."""
        first_log = Log(
            logId=5001,
            logLevel=40000,
            logTime=[2025, 1, 19, 9, 0, 0, 0],
            message="First failure",
        )
        second_log = Log(
            logId=5002,
            logLevel=40000,
            logTime=[2025, 1, 19, 9, 5, 0, 0],
            message="Second failure",
        )

        first_test_item = TestItem(
            testItemId=11111,
            testItemName="first_item",
            uniqueId="auto:first",
            testCaseHash=111,
            isAutoAnalyzed=False,
            issueType="PB001",
            startTime=[2025, 1, 19, 9, 0, 0, 0],
            logs=[first_log],
        )
        second_test_item = TestItem(
            testItemId=22222,
            testItemName="second_item",
            uniqueId="auto:second",
            testCaseHash=222,
            isAutoAnalyzed=True,
            issueType="AB001",
            startTime=[2025, 1, 19, 9, 10, 0, 0],
            logs=[second_log],
        )

        launch = Launch(
            launchId=700,
            project=7,
            launchName="multi_items",
            launchNumber=2,
            launchStartTime=[2025, 1, 19, 8, 45, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[first_test_item, second_test_item],
        )

        # Act
        results = prepare_test_items(launch)

        # Assert
        assert len(results) == 2
        first_result, second_result = results
        assert first_result.test_item_id == "11111"
        assert first_result.log_count == 1
        assert second_result.test_item_id == "22222"
        assert second_result.log_count == 1

    def test_prepare_test_item_similarity_thresholds(self):
        """Test log deduplication with different similarity thresholds."""
        logs = [
            Log(
                logId=6001,
                logLevel=40000,
                logTime=[2025, 1, 20, 10, 0, 0, 0],
                message="Timeout error while connecting to database",
            ),
            Log(
                logId=6002,
                logLevel=40000,
                logTime=[2025, 1, 20, 10, 1, 0, 0],
                message="Timeout error while connecting to database again",
            ),
            Log(
                logId=6003,
                logLevel=40000,
                logTime=[2025, 1, 20, 10, 2, 0, 0],
                message="Different failure occurred during processing",
            ),
        ]
        test_item = TestItem(
            testItemId=33333,
            testItemName="deduplication_item",
            uniqueId="auto:dedup",
            testCaseHash=333,
            isAutoAnalyzed=False,
            issueType="PB001",
            startTime=[2025, 1, 20, 9, 59, 0, 0],
            logs=logs,
        )
        base_launch = Launch(
            launchId=800,
            project=8,
            launchName="dedup_launch",
            launchNumber=3,
            launchStartTime=[2025, 1, 20, 9, 45, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act - threshold 1.0 preserves all distinct logs
        all_logs_result = prepare_test_items(base_launch, similarity_threshold=1.0)[0]
        # Act - threshold 0.0 keeps only the last log
        last_log_only_result = prepare_test_items(base_launch, similarity_threshold=0.0)[0]
        # Act - threshold 0.5 merges similar first two logs, keeps different third
        merged_logs_result = prepare_test_items(base_launch, similarity_threshold=0.5)[0]

        # Assert
        assert all_logs_result.log_count == 3
        assert [log_data.log_id for log_data in all_logs_result.logs] == ["6001", "6002", "6003"]

        assert last_log_only_result.log_count == 1
        assert [log_data.log_id for log_data in last_log_only_result.logs] == ["6003"]

        assert merged_logs_result.log_count == 2
        assert [log_data.log_id for log_data in merged_logs_result.logs] == ["6002", "6003"]

    def test_prepare_test_item_maximum_log_number_to_take(self):
        """Test that only the configured number of logs are taken after deduplication."""
        logs = [
            Log(
                logId=7001,
                logLevel=40000,
                logTime=[2025, 1, 21, 11, 0, 0, 0],
                message="First unique error",
            ),
            Log(
                logId=7002,
                logLevel=40000,
                logTime=[2025, 1, 21, 11, 1, 0, 0],
                message="Second unique error",
            ),
            Log(
                logId=7003,
                logLevel=40000,
                logTime=[2025, 1, 21, 11, 2, 0, 0],
                message="Third unique error",
            ),
            Log(
                logId=7004,
                logLevel=40000,
                logTime=[2025, 1, 21, 11, 3, 0, 0],
                message="Fourth unique error",
            ),
        ]
        test_item = TestItem(
            testItemId=44444,
            testItemName="limit_logs_item",
            uniqueId="auto:limit",
            testCaseHash=444,
            isAutoAnalyzed=False,
            issueType="PB001",
            startTime=[2025, 1, 21, 10, 59, 0, 0],
            logs=logs,
        )
        launch = Launch(
            launchId=900,
            project=9,
            launchName="limit_launch",
            launchNumber=4,
            launchStartTime=[2025, 1, 21, 10, 45, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch, maximum_log_number_to_take=2, similarity_threshold=1.0)[0]

        # Assert - only the last two logs remain, log order reset
        assert result.log_count == 2
        assert [log_data.log_id for log_data in result.logs] == ["7003", "7004"]
        assert [log_data.log_order for log_data in result.logs] == [0, 1]

    def test_prepare_test_item_minimal_log_level_filter(self):
        """Test that logs below minimal_log_level are filtered out."""
        logs = [
            Log(
                logId=8001,
                logLevel=20000,
                logTime=[2025, 1, 22, 12, 0, 0, 0],
                message="Info level message",
            ),
            Log(
                logId=8002,
                logLevel=40000,
                logTime=[2025, 1, 22, 12, 1, 0, 0],
                message="Error level message",
            ),
        ]
        test_item = TestItem(
            testItemId=55555,
            testItemName="log_level_filter_item",
            uniqueId="auto:level",
            testCaseHash=555,
            isAutoAnalyzed=False,
            issueType="PB001",
            startTime=[2025, 1, 22, 11, 59, 0, 0],
            logs=logs,
        )
        launch = Launch(
            launchId=1000,
            project=10,
            launchName="log_level_launch",
            launchNumber=5,
            launchStartTime=[2025, 1, 22, 11, 45, 0, 0],
            analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
            testItems=[test_item],
        )

        # Act
        result = prepare_test_items(launch, minimal_log_level=30000, similarity_threshold=1.0)[0]

        # Assert
        assert result.log_count == 1
        assert len(result.logs) == 1
        assert result.logs[0].log_id == "8002"
