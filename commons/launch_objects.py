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

from typing import List
from pydantic import BaseModel
from datetime import datetime


class AnalyzerConf(BaseModel):
    """Analyzer config object"""
    analyzerMode: str = "ALL"
    minShouldMatch: int = 0
    numberOfLogLines: int = -1
    isAutoAnalyzerEnabled: bool = True
    indexingRunning: bool = True


class Log(BaseModel):
    """Log object"""
    logId: int
    logLevel: int = 0
    message: str


class TestItem(BaseModel):
    """Test item object"""
    testItemId: int
    uniqueId: str
    isAutoAnalyzed: bool
    issueType: str = ""
    originalIssueType: str = ""
    startTime: List[int] = list(datetime.now().timetuple())[:7]
    testCaseHash: int = 0
    logs: List[Log] = []


class TestItemInfo(BaseModel):
    """Test item object"""
    testItemId: int
    uniqueId: str
    startTime: List[int] = list(datetime.now().timetuple())[:7]
    testCaseHash: int = 0
    launchId: int
    launchName: str = ""
    project: int
    analyzerConfig: AnalyzerConf = AnalyzerConf()


class Launch(BaseModel):
    """Launch object"""
    launchId: int
    project: int
    launchName: str = ""
    analyzerConfig: AnalyzerConf = AnalyzerConf()
    testItems: List[TestItem] = []


class AnalysisResult(BaseModel):
    """Analysis result object"""
    testItem: int
    issueType: str
    relevantItem: int


class CleanIndex(BaseModel):
    """Clean index object"""
    ids: List[int]
    project: int


class SearchLogs(BaseModel):
    """Search logs object"""
    launchId: int
    launchName: str
    itemId: int
    projectId: int
    filteredLaunchIds: List[int]
    logMessages: List[str]
    logLines: int


class Response(BaseModel):
    """Response object"""
    acknowledged: bool = False
    error: str = ""
    status: int = 0


class IndexResult(BaseModel):
    """Index result object"""
    _index: str
    _type: str
    _id: str
    _version: int
    result: str
    created: bool
    status: int


class Item(BaseModel):
    """Index item object"""
    index: IndexResult


class BulkResponse(BaseModel):
    """Bulk response object"""
    took: int
    errors: bool
    items: List[Item] = []
    status: int = 0
