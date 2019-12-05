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

from datetime import datetime
from typing import List
from pydantic import BaseModel

class AnalyzerConf(BaseModel):
    analyzerMode: str = "ALL"
    minDocFreq: float = 0
    minTermFreq: float = 0
    minShouldMatch: int = 0
    numberOfLogLines: int = -1
    isAutoAnalyzerEnabled: bool = True
    indexingRunning: bool = True

class Log(BaseModel):
    logId: int
    logLevel: int = 0
    message: str

class TestItem(BaseModel):
    testItemId: int
    uniqueId: str
    isAutoAnalyzed: bool
    issueType: str = ""
    originalIssueType: str = ""
    logs: List[Log] = []

class Launch(BaseModel):
    launchId: int
    project: int
    launchName: str = ""
    analyzerConfig: AnalyzerConf = AnalyzerConf()
    testItems: List[TestItem] = []

class AnalysisResult(BaseModel):
    testItem: int
    issueType: str
    relevantItem: int

class CleanIndex(BaseModel):
    ids: List[int]
    project: int

class SearchLogs(BaseModel):
    launchId: int
    launchName: str
    itemId: int
    projectId: int
    filteredLaunchIds: List[int]
    logMessages: List[str]
    logLines: int

class SearchLogConfig(BaseModel):
    searchMode: str
    numberOfLogLines: int

class Response(BaseModel):
    acknowledged: bool = False
    error: str = ""
    status: int = 0

class IndexResult(BaseModel):
    _index: str
    _type: str
    _id: str
    _version: int
    result: str
    created: bool
    status: int

class Item(BaseModel):
    index: IndexResult

class BulkResponse(BaseModel):
    took: int
    errors: bool
    items: List[Item] = []
    status: int = 0