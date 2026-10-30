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

"""Auto-analysis handler which searches for similar Test Items and chooses one by the Gradient Boosting model."""

from time import time
from typing import Any, Optional

from app.commons import logging
from app.commons.model.db import Hit
from app.commons.model.launch_objects import (
    AnalysisCandidate,
    AnalysisResult,
    AnalyzerConf,
    ApplicationConfig,
    Launch,
    SearchConfig,
)
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.os_client import OsClient
from app.commons.query_builder import (
    DEFAULT_ITEM_QUERY_SIZE,
    AutoAnalysisQueryBuilder,
    add_constraints_for_launches_into_query,
    add_start_time_decay,
)
from app.commons.similarity_filter import SimilarityFilter, choose_message_fields
from app.ml.predictor import AutoAnalysisPredictor, PredictionResult
from app.service.analysis.handler import AnalysisContext, AnalysisHandler, HandlerResult
from app.utils import utils
from app.utils.utils import normalize_issue_type

LOGGER = logging.getLogger("analyzerApp.boostingHandler")

NO_DEFECT_BASE_ISSUE_TYPE = "nd"
# "No Defect" Test Items are found along with the regular candidates and take their places in search results
ANALYSIS_QUERY_SIZE = DEFAULT_ITEM_QUERY_SIZE * 3 // 2

SearchResults = tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]


def choose_fields_to_filter_strict(number_of_log_lines: int, min_similarity: float) -> list[str]:
    """Choose log fields every one of which must be similar for a found log to be analyzed by.

    :param number_of_log_lines: Number of log lines to use, -1 means all lines
    :param min_similarity: Minimum similarity in the range [0.0, 1.0]
    :return: Log field names
    """
    fields = choose_message_fields(number_of_log_lines)
    if min_similarity > 0.99:
        fields.append("found_tests_and_methods")
    return fields


def to_analysis_result(candidate: AnalysisCandidate, result: PredictionResult) -> AnalysisResult:
    return AnalysisResult(
        testItem=candidate.testItemId,
        issueType=result.data.mrHit.source.issue_type or "",
        relevantItem=int(result.data.mrHit.source.test_item_id),
    )


def choose_best_prediction(prediction_results: list[PredictionResult]) -> Optional[PredictionResult]:
    """Choose the positive prediction with the highest probability, the first found one on ties.

    :param prediction_results: Predictions for found Test Items
    :return: The best positive prediction, or None if there are no positive predictions
    """
    positive_predictions = [result for result in prediction_results if result.label == 1]
    if not positive_predictions:
        return None
    return max(positive_predictions, key=lambda result: (result.probability[1], -result.original_position))


def is_no_defect(hit: Hit[TestItemIndexData]) -> bool:
    return normalize_issue_type(hit.source.issue_type).startswith(NO_DEFECT_BASE_ISSUE_TYPE)


def split_no_defect_candidates(search_results: SearchResults) -> tuple[SearchResults, Optional[SearchResults]]:
    """Split found Test Items into regular candidates and a "No Defect" candidate.

    Regular candidates never contain "No Defect" Test Items, the same way as the train data of the model. A request
    Test Item has a "No Defect" candidate only if the latest similar run of the same test case is "No Defect".

    :param search_results: Request Test Item and similar Test Items of any issue type except "To Investigate"
    :return: Request Test Item with regular candidates, and with the latest run of the same test case if it is a
             "No Defect" one, otherwise None
    """
    request_item, hits = search_results
    regular_results = (request_item, [hit for hit in hits if not is_no_defect(hit)])
    if not request_item.test_case_hash:
        return regular_results, None
    same_test_case_hits = [hit for hit in hits if hit.source.test_case_hash == request_item.test_case_hash]
    if not same_test_case_hits:
        return regular_results, None
    # Start time is stored in "%Y-%m-%d %H:%M:%S" format, so string order is chronological. On ties `max` keeps the
    # first, i.e. the best scored, Test Item.
    latest = max(same_test_case_hits, key=lambda hit: hit.source.start_time or "")
    if not is_no_defect(latest):
        return regular_results, None
    return regular_results, (request_item, [latest])


class BoostingHandler(AnalysisHandler):
    """Analyze Test Items by similar ones found with a "more like this" query, chosen by the boosting model."""

    name = "boosting"
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    launch_boost: float
    model_chooser: ModelChooser
    os_client: OsClient
    namespace_finder: NamespaceFinder
    query_builder: AutoAnalysisQueryBuilder

    def __init__(
        self,
        model_chooser: ModelChooser,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        os_client: OsClient,
        namespace_finder: NamespaceFinder,
        next_handler: Optional[AnalysisHandler] = None,
    ) -> None:
        """Initialize the handler.

        :param model_chooser: Chooser of the global or custom boosting model of a project
        :param app_config: Application configuration
        :param search_cfg: Search configuration
        :param os_client: OpenSearch client
        :param namespace_finder: Finder of the chosen namespaces of a project
        :param next_handler: Handler to pass the Test Items to which have no positive predictions
        """
        super().__init__(next_handler)
        self.search_cfg = search_cfg
        self.launch_boost = abs(search_cfg.BoostLaunch)
        self.model_chooser = model_chooser
        self.app_config = app_config
        self.os_client = os_client
        self.namespace_finder = namespace_finder
        self.query_builder = AutoAnalysisQueryBuilder(self.search_cfg)

    def _get_config_for_boosting(self, analyzer_config: AnalyzerConf) -> dict[str, Any]:
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": utils.find_min_should_match_threshold(analyzer_config, self.search_cfg) / 100,
            "min_word_length": self.search_cfg.MinWordLength,
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "boosting_model": self.search_cfg.BoostModelFolder,
            "time_weight_decay": self.search_cfg.TimeWeightDecay,
        }

    def _get_min_should_match_setting(self, launch: Launch) -> str:
        if launch.analyzerConfig.minShouldMatch > 0:
            return f"{launch.analyzerConfig.minShouldMatch}%"
        return self.search_cfg.MinShouldMatch

    def _build_item_query(self, launch: Launch, request_item: TestItemIndexData) -> dict[str, Any]:
        """Build a query to search for Test Items similar to the request one by all its logs.

        :param launch: Launch being analyzed
        :param request_item: Request Test Item
        :return: OpenSearch query, or an empty dict if the request Test Item has no logs to search by
        """
        query = self.query_builder.build(
            request_item,
            number_of_log_lines=launch.analyzerConfig.numberOfLogLines,
            min_should_match=self._get_min_should_match_setting(launch),
            min_logs_to_match="100%" if launch.analyzerConfig.allMessagesShouldMatch else "1",
            filter_no_defect=False,
            size=ANALYSIS_QUERY_SIZE,
        )
        if not query:
            return {}
        query = add_constraints_for_launches_into_query(query, launch, self.launch_boost)
        return add_start_time_decay(query, request_item.start_time, self.search_cfg.TimeWeightDecay)

    def _create_similarity_filter(self, analyzer_config: AnalyzerConf) -> SimilarityFilter:
        min_similarity = utils.find_min_should_match_threshold(analyzer_config, self.search_cfg) / 100
        return SimilarityFilter(
            choose_fields_to_filter_strict(analyzer_config.numberOfLogLines, min_similarity),
            min_similarity,
            all_fields=True,
        )

    def _filter_by_similarity(
        self,
        launch: Launch,
        similarity_filter: SimilarityFilter,
        search_results: SearchResults,
    ) -> SearchResults:
        """Remove found Test Items which logs are not similar enough to the request logs by every message field.

        :param launch: Launch being analyzed
        :param similarity_filter: Filter configured for the launch
        :param search_results: Request Test Item and found Test Items
        :return: Request Test Item and found Test Items with similar logs
        """
        request_item, hits = search_results
        required_log_indices = None
        if launch.analyzerConfig.allMessagesShouldMatch:
            required_log_indices = self.query_builder.select_log_indices(
                request_item, launch.analyzerConfig.numberOfLogLines
            )
        filtered_results = similarity_filter.filter(search_results, required_log_indices)
        removed = len(hits) - len(filtered_results[1])
        if removed:
            LOGGER.debug(
                f"Filtering by similarity removed {removed} of {len(hits)} found items for test item "
                f"{request_item.test_item_id}"
            )
        return filtered_results

    def _query_candidates_for_launch(
        self,
        launch: Launch,
        request_items: list[TestItemIndexData],
        max_batch_size: Optional[int] = None,
    ) -> list[SearchResults]:
        """Query candidates for a whole launch, one query per request Test Item, batching queries across Test Items.

        One `msearch` per test item leaves OpenSearch with only a handful of queries to run
        concurrently and pays a full round trip each time. Batching keeps the request count
        proportional to the query count instead of the test item count.

        Batch size is a memory trade, not just a round-trip one: every query in a batch holds its
        full response until the batch is consumed, and a Test Item response carries its matched
        logs. Past roughly 20 queries the round-trip saving flattens while the memory held per
        request keeps growing linearly, so `AnalysisQueryBatchSize` defaults there.

        :param launch: Launch being analyzed
        :param request_items: Prepared request Test Items
        :param max_batch_size: Queries per `msearch`; defaults to `AnalysisQueryBatchSize`
        :return: Request Test Items with the Test Items found for them, aligned with `request_items`
        """
        if max_batch_size is None:
            max_batch_size = self.search_cfg.AnalysisQueryBatchSize
        plan: list[int] = []
        all_queries: list[dict[str, Any]] = []
        for item_index, request_item in enumerate(request_items):
            query = self._build_item_query(launch, request_item)
            if not query:
                continue
            plan.append(item_index)
            all_queries.append({})
            all_queries.append(query)

        similarity_filter = self._create_similarity_filter(launch.analyzerConfig)
        results: list[SearchResults] = [(request_item, []) for request_item in request_items]

        # Each query contributes a header and a body to `all_queries`, so a chunk of `chunk`
        # entries corresponds to `chunk // 2` entries of `plan`.
        chunk = max_batch_size * 2
        for offset in range(0, len(all_queries), chunk):
            planned = plan[offset // 2 : (offset + chunk) // 2]
            consumed = 0
            for item_index, hits in zip(
                planned, self.os_client.msearch_grouped(launch.project, all_queries[offset : offset + chunk])
            ):
                consumed += 1
                results[item_index] = self._filter_by_similarity(
                    launch, similarity_filter, (request_items[item_index], hits)
                )
            if consumed < len(planned):
                LOGGER.warning(
                    "Got %d responses for %d queries in project %s, %d test items were not analyzed",
                    consumed,
                    len(planned),
                    str(launch.project),
                    len(planned) - consumed,
                )
        return results

    def _to_candidates(
        self, launch: Launch, results_per_test_item: list[SearchResults], time_processed: float
    ) -> list[AnalysisCandidate]:
        candidates = []
        for search_results in results_per_test_item:
            regular_results, no_defect_results = split_no_defect_candidates(search_results)
            candidates.append(
                AnalysisCandidate(
                    analyzerConfig=launch.analyzerConfig,
                    testItemId=int(search_results[0].test_item_id),
                    project=launch.project,
                    launchId=launch.launchId,
                    launchName=launch.launchName,
                    launchNumber=launch.launchNumber,
                    timeProcessed=time_processed,
                    candidates=regular_results,
                    candidatesWithNoDefect=no_defect_results,
                )
            )
        return candidates

    def _predict_best(
        self, context: AnalysisContext, predictor: AutoAnalysisPredictor, candidate: AnalysisCandidate
    ) -> Optional[PredictionResult]:
        """Choose the best positive prediction for a request Test Item.

        :param context: Launch analysis context
        :param predictor: Predictor of the launch
        :param candidate: Request Test Item with the Test Items found for it
        :return: The best positive prediction, or None if there are no positive predictions
        """
        # The "No Defect" candidate goes first: it exists only if the latest similar run of the same test case is
        # "No Defect", while regular candidates never contain such Test Items
        candidates_to_check: list[SearchResults] = []
        if candidate.candidatesWithNoDefect:
            candidates_to_check.append(candidate.candidatesWithNoDefect)
        candidates_to_check.append(candidate.candidates)

        for candidates in candidates_to_check:
            prediction_results = predictor.predict(candidates)
            if not prediction_results:
                continue
            context.stats.model_info.update(prediction_results[0].model_info_tags)
            best = choose_best_prediction(prediction_results)
            if best is not None:
                return best
        return None

    def _analyze(
        self, context: AnalysisContext, items: list[TestItemIndexData]
    ) -> tuple[list[HandlerResult], list[TestItemIndexData]]:
        launch = context.launch
        stats = context.stats
        t_start = time()
        results_per_test_item = self._query_candidates_for_launch(launch, items)
        mean_processing_time = (time() - t_start) / max(len(items), 1)
        candidates = self._to_candidates(launch, results_per_test_item, mean_processing_time)
        LOGGER.info("Collected %d candidates for analysis", len(candidates))
        LOGGER.info("Os queries finished %.2f s.", time() - t_start)

        boosting_config = self._get_config_for_boosting(launch.analyzerConfig)
        predictor = AutoAnalysisPredictor(
            model_chooser=self.model_chooser,
            project_id=launch.project,
            boosting_config=boosting_config,
            custom_model_prob=self.search_cfg.ProbabilityForCustomModelAutoAnalysis,
            hash_source=launch.launchId,
        )

        results: list[HandlerResult] = []
        remaining: list[TestItemIndexData] = []
        for item, candidate in zip(items, candidates):
            try:
                t_start_item = time()
                stats.processed_time += candidate.timeProcessed
                if launch.project not in context.chosen_namespaces:
                    context.chosen_namespaces[launch.project] = self.namespace_finder.get_chosen_namespaces(
                        launch.project
                    )
                boosting_config["chosen_namespaces"] = context.chosen_namespaces[launch.project]

                best = self._predict_best(context, predictor, candidate)
                stats.processed_time += time() - t_start_item
                if best is None:
                    LOGGER.debug(f"Test item {candidate.testItemId} has no positive predictions")
                    remaining.append(item)
                    continue

                analysis_result = to_analysis_result(candidate, best)
                LOGGER.debug(analysis_result)
                results.append(HandlerResult(result=analysis_result, handler=self.name))
            except Exception as exc:
                LOGGER.exception(f"Unable to process candidate for analysis {candidate.testItemId}", exc_info=exc)
                stats.errors.append(utils.extract_exception(exc))
        return results, remaining
