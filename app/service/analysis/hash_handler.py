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

"""Auto-analysis handlers which take the issue type of earlier Test Items failed with exactly the same error."""

from abc import abstractmethod
from time import time
from typing import Any, Optional

from app.commons import logging
from app.commons.model.db import Hit
from app.commons.model.launch_objects import AnalysisResult, Launch, SearchConfig
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.os_client import OsClient
from app.commons.query_builder import ErrorHashQueryBuilder
from app.service.analysis.handler import AnalysisContext, AnalysisHandler, HandlerResult
from app.utils import utils

LOGGER = logging.getLogger("analyzerApp.hashHandler")

# Queries per msearch request. Responses carry only a few small fields, so batches can be much larger than the ones of
# the "more like this" search.
LOOKUP_BATCH_SIZE = 200
SAME_TEST_MATCHES = 3
ANY_TEST_MATCHES = 20


def _unanimous_result(request_item: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> Optional[AnalysisResult]:
    """Take the issue type all the found Test Items agree on, the newest one is the relevant Test Item.

    :param request_item: Prepared request Test Item
    :param hits: Found Test Items, newest first
    :return: Analysis result, or None if there are no found Test Items or they disagree
    """
    issue_types = {hit.source.issue_type for hit in hits}
    if len(issue_types) != 1:
        return None
    issue_type = issue_types.pop()
    if not issue_type:
        return None
    return AnalysisResult(
        testItem=int(request_item.test_item_id),
        issueType=issue_type,
        relevantItem=int(hits[0].source.test_item_id),
    )


class ErrorHashHandler(AnalysisHandler):
    """Base handler which finds earlier Test Items with the same error hash and decides by their issue types."""

    search_cfg: SearchConfig
    os_client: OsClient
    query_builder: ErrorHashQueryBuilder
    lookup_batch_size: int

    def __init__(
        self,
        search_cfg: SearchConfig,
        os_client: OsClient,
        next_handler: Optional[AnalysisHandler] = None,
        lookup_batch_size: int = LOOKUP_BATCH_SIZE,
    ) -> None:
        """Initialize the handler.

        :param search_cfg: Search configuration
        :param os_client: OpenSearch client
        :param next_handler: Handler to pass the Test Items to which this handler does not decide on
        :param lookup_batch_size: Queries per `msearch` request
        """
        super().__init__(next_handler)
        self.search_cfg = search_cfg
        self.os_client = os_client
        self.query_builder = ErrorHashQueryBuilder(search_cfg)
        self.lookup_batch_size = lookup_batch_size

    @abstractmethod
    def _build_query(self, launch: Launch, request_item: TestItemIndexData) -> dict[str, Any]:
        """Build the query to find earlier Test Items to decide by.

        :param launch: Launch being analyzed
        :param request_item: Prepared request Test Item
        :return: OpenSearch query, or an empty dict if the handler cannot decide on the Test Item
        """
        ...

    @abstractmethod
    def _decide(self, request_item: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> Optional[AnalysisResult]:
        """Decide on the issue type of a request Test Item by the found Test Items.

        :param request_item: Prepared request Test Item
        :param hits: Found Test Items, newest first
        :return: Analysis result, or None if the found Test Items are not enough to decide
        """
        ...

    def _lookup(self, launch: Launch, items: list[TestItemIndexData]) -> list[list[Hit[TestItemIndexData]]]:
        """Find earlier Test Items for every request Test Item, batching queries across Test Items.

        :param launch: Launch being analyzed
        :param items: Prepared request Test Items
        :return: Found Test Items aligned with `items`, empty for Test Items without a query or a response
        """
        plan: list[int] = []
        queries: list[dict[str, Any]] = []
        for item_index, item in enumerate(items):
            query = self._build_query(launch, item)
            if not query:
                continue
            plan.append(item_index)
            queries.append({})
            queries.append(query)

        results: list[list[Hit[TestItemIndexData]]] = [[] for _ in items]
        # Each query contributes a header and a body to `queries`
        chunk = self.lookup_batch_size * 2
        for offset in range(0, len(queries), chunk):
            planned = plan[offset // 2 : (offset + chunk) // 2]
            consumed = 0
            for item_index, hits in zip(
                planned, self.os_client.msearch_grouped(launch.project, queries[offset : offset + chunk])
            ):
                consumed += 1
                results[item_index] = hits
            if consumed < len(planned):
                LOGGER.warning(
                    "Got %d responses for %d error hash queries in project %s, %d test items are passed on",
                    consumed,
                    len(planned),
                    str(launch.project),
                    len(planned) - consumed,
                )
        return results

    def _analyze(
        self, context: AnalysisContext, items: list[TestItemIndexData]
    ) -> tuple[list[HandlerResult], list[TestItemIndexData]]:
        t_start = time()
        results: list[HandlerResult] = []
        remaining: list[TestItemIndexData] = []
        for item, hits in zip(items, self._lookup(context.launch, items)):
            analysis_result = self._decide(item, hits)
            if analysis_result is None:
                remaining.append(item)
                continue
            LOGGER.debug(analysis_result)
            results.append(HandlerResult(result=analysis_result, handler=self.name))
        context.stats.processed_time += time() - t_start
        LOGGER.info(
            "Handler '%s' analyzed %d of %d test items of launch %s",
            self.name,
            len(results),
            len(items),
            str(context.launch.launchId),
        )
        return results, remaining


class SameTestHashHandler(ErrorHashHandler):
    """Take the issue type of the latest runs of the same test case which failed with exactly the same error.

    The latest `SAME_TEST_MATCHES` analyzed runs are taken, a single one is enough. The issue type is taken only if all
    of them agree on it, the newest run is the relevant Test Item.
    """

    name = "hash_same_test"

    def _build_query(self, launch: Launch, request_item: TestItemIndexData) -> dict[str, Any]:
        return self.query_builder.build(launch, request_item, size=SAME_TEST_MATCHES, same_test_case=True)

    def _decide(self, request_item: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> Optional[AnalysisResult]:
        return _unanimous_result(request_item, hits)


class AnyTestHashHandler(ErrorHashHandler):
    """Take the issue type of the latest Test Items of any test case which failed with exactly the same error.

    The latest `ANY_TEST_MATCHES` analyzed Test Items are taken. If the newest run of the same test case among them is
    "No Defect", it is the result, the same way as the "No Defect" candidate of the Gradient Boosting analysis.
    Otherwise "No Defect" Test Items are skipped and the issue type is taken only if all the rest agree on it.
    """

    name = "hash_any_test"

    def _build_query(self, launch: Launch, request_item: TestItemIndexData) -> dict[str, Any]:
        return self.query_builder.build(launch, request_item, size=ANY_TEST_MATCHES, same_test_case=False)

    def _decide(self, request_item: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> Optional[AnalysisResult]:
        if request_item.test_case_hash:
            same_test_case_hits = [hit for hit in hits if hit.source.test_case_hash == request_item.test_case_hash]
            if same_test_case_hits and utils.is_no_defect(same_test_case_hits[0].source.issue_type):
                return _unanimous_result(request_item, same_test_case_hits[:1])
        return _unanimous_result(request_item, [hit for hit in hits if not utils.is_no_defect(hit.source.issue_type)])
