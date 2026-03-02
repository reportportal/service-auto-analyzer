#  Copyright 2026 EPAM Systems
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

"""Log entry-centric index data model for OpenSearch storage."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class LogItemIndexData(BaseModel):
    """
    Log entry-centric document used during migration to Test Item-centric indexing.

    Contains both log-level fields and Test Item metadata required by ML featurizers.
    """

    log_id: str = Field(default="", description="Unique identifier for the log entry")
    test_item: int = Field(default=0, description="Test Item identifier")
    test_item_name: str = Field(default="", description="Preprocessed test item name")
    test_case_hash: int = Field(default=0, description="Test case hash")
    unique_id: str = Field(default="", description="Unique ID from ReportPortal")
    launch_id: int = Field(default=0, description="Launch identifier")
    launch_number: Optional[str] = Field(default=None, description="Launch execution number")
    launch_name: str = Field(default="", description="Launch name")
    issue_type: str = Field(default="", description="Issue type label")
    is_auto_analyzed: bool = Field(default=False, description="Whether assigned by auto-analysis")
    start_time: str = Field(default="", description="Test item start time")
    log_time: str = Field(default="", description="Log time")
    log_level: int = Field(default=0, description="Log level")
    is_merged: bool = Field(default=False, description="Whether log is a merged entry")
    merged_small_logs: str = Field(default="", description="Merged small logs text")
    message: str = Field(default="", description="Prepared log message")
    detected_message: str = Field(default="", description="Detected message without numbers")
    detected_message_with_numbers: str = Field(default="", description="Detected message with numbers")
    detected_message_extended: str = Field(default="", description="Detected message with enriched data")
    detected_message_without_params_extended: str = Field(
        default="", description="Detected message without params (enriched)"
    )
    detected_message_without_params_and_brackets: str = Field(
        default="", description="Detected message without params and brackets"
    )
    stacktrace: str = Field(default="", description="Stacktrace")
    stacktrace_extended: str = Field(default="", description="Enriched stacktrace")
    message_extended: str = Field(default="", description="Enriched message")
    message_without_params_extended: str = Field(default="", description="Message without params (enriched)")
    message_without_params_and_brackets: str = Field(default="", description="Message without params and brackets")
    message_params: str = Field(default="", description="Message parameters")
    only_numbers: str = Field(default="", description="Only numbers extracted from message")
    found_exceptions: str = Field(default="", description="Found exceptions")
    # TODO: the following field is barely used, subject to remove
    found_exceptions_extended: str = Field(default="", description="Found exceptions")
    found_tests_and_methods: str = Field(default="", description="Found tests and methods")
    potential_status_codes: str = Field(default="", description="Potential status codes")
    urls: str = Field(default="", description="URLs extracted from message")
    paths: Optional[str] = Field(default=None, description="File paths found in message")
    original_message: str = Field(default="", description="Original raw message")
    whole_message: str = Field(default="", description="Combined exception message and stacktrace")
    cluster_id: str = Field(default="", description="Cluster identifier")
    cluster_message: str = Field(default="", description="Cluster message")
    cluster_with_numbers: bool = Field(default=False, description="Cluster message with numbers flag")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogItemIndexData":
        """
        Create LogItemIndexData instance from dictionary.

        :param data: Dictionary containing log item data
        :return: LogItemIndexData instance
        """
        return cls.model_validate(data)
