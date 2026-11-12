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

"""Chain of Responsibility for auto-analysis.

Every handler of the chain gets the Test Items of one launch which are not analyzed yet, analyzes the ones it can and
passes the rest to the next handler. Handlers work on batches, not on single Test Items, so they can query OpenSearch
for a whole launch at once.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.commons.model.launch_objects import AnalysisResult, Launch, LaunchAnalysisStats
from app.commons.model.test_item_index import TestItemIndexData


@dataclass
class AnalysisContext:
    """State of the analysis of one launch, shared by all handlers of the chain."""

    launch: Launch
    stats: LaunchAnalysisStats
    # Chosen namespaces by project, shared by all launches of one request
    chosen_namespaces: dict[int, dict[str, int]]


@dataclass(frozen=True)
class HandlerResult:
    """Analysis result of a Test Item together with the name of the handler which produced it."""

    result: AnalysisResult
    handler: str


class AnalysisHandler(ABC):
    """A link of the auto-analysis chain."""

    name: str
    _next: Optional["AnalysisHandler"]

    def __init__(self, next_handler: Optional["AnalysisHandler"] = None) -> None:
        """Initialize the handler.

        :param next_handler: Handler to pass the Test Items to which this handler does not analyze
        """
        self._next = next_handler

    def handle(
        self, context: AnalysisContext, items: list[TestItemIndexData]
    ) -> tuple[list[HandlerResult], list[TestItemIndexData]]:
        """Analyze Test Items of a launch by this handler and the rest of the chain.

        :param context: Launch analysis context
        :param items: Prepared request Test Items, not analyzed yet
        :return: Results of this handler and of the rest of the chain, and Test Items no handler found results for
        """
        results, remaining = self._analyze(context, items)
        if remaining and self._next is not None:
            next_results, remaining = self._next.handle(context, remaining)
            results = results + next_results
        return results, remaining

    @abstractmethod
    def _analyze(
        self, context: AnalysisContext, items: list[TestItemIndexData]
    ) -> tuple[list[HandlerResult], list[TestItemIndexData]]:
        """Analyze the Test Items this handler can.

        A Test Item which failed with an error is neither in the results nor in the remaining Test Items, the error is
        recorded in the launch statistics instead.

        :param context: Launch analysis context
        :param items: Prepared request Test Items, not analyzed yet
        :return: Results for the Test Items this handler analyzed, and the Test Items to pass to the next handler
        """
        ...
