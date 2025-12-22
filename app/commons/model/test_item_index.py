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

"""Test Item-centric index data model for OpenSearch storage."""

from pydantic import BaseModel, Field


class LogData(BaseModel):
    """
    Nested log entry data for Test Item-centric OpenSearch indexing.

    Represents a single log entry within a Test Item, containing preprocessed
    message fields, stack trace information, and metadata for analysis.
    """

    log_id: str = Field(description="Unique identifier for the log entry")
    log_order: int = Field(description="Position of log within Test Item (0-based)")
    log_time: str = Field(description="Timestamp of the log entry")
    log_level: int = Field(description="Log level (e.g., 40000 for ERROR)")
    cluster_id: str = Field(default="", description="Cluster identifier if clustered")
    cluster_message: str = Field(default="", description="Cluster representative message")
    cluster_with_numbers: bool = Field(default=False, description="Whether cluster includes numbers")
    original_message: str = Field(description="Raw log message for display")
    message: str = Field(description="Cleaned message (garbage removed)")
    message_lines: int = Field(description="Number of lines in message")
    message_words_number: int = Field(description="Word count in message")
    message_extended: str = Field(description="Message with enriched class/method names")
    message_without_params_extended: str = Field(description="Message without params, enriched")
    message_without_params_and_brackets: str = Field(description="Message without params and brackets")
    detected_message: str = Field(description="Exception message without numbers")
    detected_message_with_numbers: str = Field(description="Exception message with numbers")
    detected_message_extended: str = Field(description="Enriched exception message")
    detected_message_without_params_extended: str = Field(description="Exception message without params, enriched")
    detected_message_without_params_and_brackets: str = Field(
        description="Exception message without params and brackets"
    )
    stacktrace: str = Field(description="Stack trace portion")
    stacktrace_extended: str = Field(description="Enriched stack trace")
    only_numbers: str = Field(description="Extracted numeric values")
    potential_status_codes: str = Field(description="Detected status/error codes")
    found_exceptions: str = Field(description="Exception type names found")
    found_exceptions_extended: str = Field(description="Enriched exception names")
    found_tests_and_methods: str = Field(description="Test method references")
    urls: str = Field(description="URLs found in message")
    paths: str = Field(description="File paths found in message")
    message_params: str = Field(description="Extracted parameters")
    whole_message: str = Field(description="Combined exception message and stacktrace")


class TestItemIndexData(BaseModel):
    """
    Test Item-centric document for OpenSearch indexing.

    Represents a complete Test Item with all associated logs as nested objects.
    This structure allows holistic analysis of test failures while preserving
    log order, frequency, and context.
    """

    test_item_id: str = Field(description="Unique identifier for the Test Item")
    test_item_name: str = Field(description="Name of the test (preprocessed)")
    unique_id: str = Field(description="Unique ID from ReportPortal")
    test_case_hash: int = Field(description="Hash for test case grouping")
    launch_id: str = Field(description="Parent launch identifier")
    launch_name: str = Field(description="Name of the launch")
    launch_number: str = Field(description="Launch execution number")
    launch_start_time: str = Field(description="When the launch started")
    is_auto_analyzed: bool = Field(description="Whether auto-analysis was applied")
    issue_type: str = Field(description="Assigned issue type (e.g., pb001, ab001)")
    start_time: str = Field(description="When the test item started")
    log_count: int = Field(description="Number of logs in this Test Item")
    logs: list[LogData] = Field(description="Nested log entries")

    def to_index_dict(self) -> dict:
        """
        Convert the model to a dictionary suitable for OpenSearch indexing.

        :return: Dictionary representation ready for indexing
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "TestItemIndexData":
        """
        Create TestItemIndexData instance from dictionary.

        :param data: Dictionary containing test item data
        :return: TestItemIndexData instance
        """
        return cls(**data)
