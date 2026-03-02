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
from typing import Any, Optional

from pydantic import BaseModel, Field


class LogData(BaseModel):
    """
    Nested log entry data for Test Item-centric OpenSearch indexing.

    Represents a single log entry within a Test Item, containing preprocessed
    message fields, stack trace information, and metadata for analysis.
    """

    log_id: str = Field(description="Unique identifier for the log entry")
    log_order: Optional[int] = Field(default=None, description="Position of log within Test Item (0-based)")
    log_time: Optional[str] = Field(default=None, description="Timestamp of the log entry")
    log_level: int = Field(description="Log level (e.g., 40000 for ERROR)")
    cluster_id: str = Field(default="", description="Cluster identifier if clustered")
    cluster_message: str = Field(default="", description="Cluster representative message")
    cluster_with_numbers: bool = Field(default=False, description="Whether cluster includes numbers")
    original_message: Optional[str] = Field(default=None, description="Raw log message for display")
    message: Optional[str] = Field(default=None, description="Cleaned message (garbage removed)")
    message_for_clustering: Optional[str] = Field(
        default=None, description="Cleaned message for clustering (less removed text)"
    )
    message_lines: Optional[int] = Field(default=None, description="Number of lines in message")
    message_words_number: Optional[int] = Field(default=None, description="Word count in message")
    message_extended: Optional[str] = Field(default=None, description="Message with enriched class/method names")
    message_without_params_extended: Optional[str] = Field(
        default=None, description="Message without params, enriched"
    )
    message_without_params_and_brackets: Optional[str] = Field(
        default=None, description="Message without params and brackets"
    )
    detected_message: Optional[str] = Field(default=None, description="Exception message without numbers")
    detected_message_with_numbers: Optional[str] = Field(default=None, description="Exception message with numbers")
    detected_message_extended: Optional[str] = Field(default=None, description="Enriched exception message")
    detected_message_without_params_extended: Optional[str] = Field(
        default=None, description="Exception message without params, enriched"
    )
    detected_message_without_params_and_brackets: Optional[str] = Field(
        default=None, description="Exception message without params and brackets"
    )
    stacktrace: Optional[str] = Field(default=None, description="Stack trace portion")
    stacktrace_extended: Optional[str] = Field(default=None, description="Enriched stack trace")
    only_numbers: Optional[str] = Field(default=None, description="Extracted numeric values")
    potential_status_codes: Optional[str] = Field(default=None, description="Detected status/error codes")
    found_exceptions: Optional[str] = Field(default=None, description="Exception type names found")
    # TODO: the following field is barely used, subject to remove
    found_exceptions_extended: Optional[str] = Field(default=None, description="Enriched exception names")
    found_tests_and_methods: Optional[str] = Field(default=None, description="Test method references")
    urls: Optional[str] = Field(default=None, description="URLs found in message")
    paths: Optional[str] = Field(default=None, description="File paths found in message")
    message_params: Optional[str] = Field(default=None, description="Extracted parameters")
    whole_message: Optional[str] = Field(default=None, description="Combined exception message and stacktrace")


class LogClusterData(BaseModel):
    """Payload for updating cluster data on a nested log."""

    log_id: str = Field(description="Identifier of the log entry to update")
    test_item_id: str = Field(description="Identifier of the parent Test Item document")
    cluster_id: str = Field(description="Cluster identifier to assign")
    cluster_message: str = Field(description="Cluster representative message")
    cluster_with_numbers: bool = Field(description="Whether cluster includes numbers")

    def to_update_params(self) -> dict[str, Any]:
        """
        Convert the update data into OpenSearch script parameters.

        :return: Dictionary of painless script params
        """
        return {
            "log_id": self.log_id,
            "cluster_id": self.cluster_id,
            "cluster_message": self.cluster_message,
            "cluster_with_numbers": self.cluster_with_numbers,
        }


class TestItemHistoryData(BaseModel):
    """Payload for updating issue history of a Test Item."""

    test_item_id: str = Field(description="Identifier of the Test Item to update")
    is_auto_analyzed: bool = Field(description="Whether assignment was made by auto-analysis")
    issue_type: str = Field(description="Assigned issue type (e.g., pb001, ab001)")
    timestamp: str = Field(description="Timestamp of the assignment")
    issue_comment: str = Field(default="", description="Optional comment for the assignment")

    def to_update_dict(self) -> dict[str, Any]:
        """
        Convert update data into issue_history entry shape used in OpenSearch script.

        :return: Dictionary for painless script params
        """
        return {
            "is_auto_analyzed": self.is_auto_analyzed,
            "issue_type": self.issue_type,
            "timestamp": self.timestamp,
            "issue_comment": self.issue_comment,
        }


class TestItemIndexData(BaseModel):
    """
    Test Item-centric document for OpenSearch indexing.

    Represents a complete Test Item with all associated logs as nested objects.
    This structure allows holistic analysis of test failures while preserving
    log order, frequency, and context.
    """

    test_item_id: str = Field(description="Unique identifier for the Test Item")
    test_item_name: Optional[str] = Field(default=None, description="Name of the test (preprocessed)")
    unique_id: Optional[str] = Field(default=None, description="Unique ID from ReportPortal")
    test_case_hash: Optional[int] = Field(default=None, description="Hash for test case grouping")
    launch_id: str = Field(description="Parent launch identifier")
    launch_name: Optional[str] = Field(default=None, description="Name of the launch")
    launch_number: Optional[str] = Field(default=None, description="Launch execution number")
    launch_start_time: Optional[str] = Field(default=None, description="When the launch started")
    is_auto_analyzed: Optional[bool] = Field(default=None, description="Whether the Test Item was auto-analyzed")
    issue_type: Optional[str] = Field(default=None, description="Assigned issue type (e.g., pb001, ab001)")
    start_time: Optional[str] = Field(default=None, description="When the test item started")
    log_count: Optional[int] = Field(default=None, description="Number of logs in this Test Item")
    logs: Optional[list[LogData]] = Field(default=None, description="Nested log entries")
    issue_history: Optional[list[TestItemHistoryData]] = Field(default=None, description="Nested issue type history")

    def to_index_dict(self) -> dict[str, Any]:
        """
        Convert the model to a dictionary suitable for OpenSearch indexing.

        :return: Dictionary representation ready for indexing
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestItemIndexData":
        """
        Create TestItemIndexData instance from dictionary.

        :param data: Dictionary containing test item data
        :return: TestItemIndexData instance
        """
        return cls.model_validate(data)
