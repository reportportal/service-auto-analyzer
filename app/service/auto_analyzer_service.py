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

import json
from collections import Counter
from time import time
from typing import Optional

from app.amqp.amqp import AmqpClient
from app.commons import logging, request_factory
from app.commons.model.launch_objects import AnalysisResult, ApplicationConfig, Launch, SearchConfig
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.os_client import OsClient
from app.service.analysis import (
    AnalysisContext,
    AnalysisHandler,
    AnyTestHashHandler,
    BoostingHandler,
    LaunchAnalysisStats,
    SameTestHashHandler,
)
from app.utils import utils

LOGGER = logging.getLogger("analyzerApp.autoAnalyzerService")


def prepare_request_items_for_launch(launch: Launch) -> list[TestItemIndexData]:
    """Prepare request Test Items of the launch, skipping Test Items without logs to analyze by.

    :param launch: Launch to analyze
    :return: Request Test Items with logs
    """
    prepared = request_factory.prepare_test_items(
        launch,
        number_of_logs_to_index=launch.analyzerConfig.numberOfLogsToIndex,
        minimal_log_level=launch.analyzerConfig.minimumLogLevel,
        similarity_threshold_to_drop=launch.analyzerConfig.similarityThresholdToDrop,
    )
    return [prepared_item for prepared_item in prepared if prepared_item.logs]


class AutoAnalyzerService:
    """Analyze launches by a chain of handlers and publish the results.

    The service prepares request Test Items once per launch and passes them to the first handler of the chain. Every
    handler analyzes what it can and passes the rest on, the results of all handlers are published in one response.
    """

    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient
    namespace_finder: NamespaceFinder
    model_chooser: ModelChooser
    amqp_client: Optional[AmqpClient]
    chain: AnalysisHandler

    def __init__(
        self,
        model_chooser: ModelChooser,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        os_client: Optional[OsClient] = None,
        namespace_finder: Optional[NamespaceFinder] = None,
    ) -> None:
        self.search_cfg = search_cfg
        self.model_chooser = model_chooser
        self.app_config = app_config
        self.os_client = os_client or OsClient(app_config=self.app_config)
        self.namespace_finder = namespace_finder or NamespaceFinder(app_config)
        self.amqp_client = None
        if self.app_config.amqpUrl:
            self.amqp_client = AmqpClient(self.app_config)
        self.chain = self._build_chain()

    def _build_chain(self) -> AnalysisHandler:
        """Build the chain of handlers which analyze request Test Items, in the order they are asked.

        :return: The first handler of the chain
        """
        chain: AnalysisHandler = BoostingHandler(
            self.model_chooser, self.app_config, self.search_cfg, self.os_client, self.namespace_finder
        )
        if self.search_cfg.HashMatchAnyTest:
            chain = AnyTestHashHandler(self.search_cfg, self.os_client, next_handler=chain)
        if self.search_cfg.HashMatchSameTest:
            chain = SameTestHashHandler(self.search_cfg, self.os_client, next_handler=chain)
        return chain

    def _should_stop_processing(self, test_items_processed: int) -> bool:
        if test_items_processed >= self.search_cfg.MaxAutoAnalysisItemsToProcess:
            LOGGER.info("Only first %d test items were taken", self.search_cfg.MaxAutoAnalysisItemsToProcess)
            return True
        return False

    def _create_stats(self, launch: Launch) -> LaunchAnalysisStats:
        return LaunchAnalysisStats(
            launch_id=launch.launchId,
            launch_name=launch.launchName,
            project_id=launch.project,
            number_of_log_lines=launch.analyzerConfig.numberOfLogLines,
            min_should_match=utils.find_min_should_match_threshold(launch.analyzerConfig, self.search_cfg),
            module_version=[self.app_config.appVersion],
        )

    def analyze_logs(self, launches: list[Launch]) -> None:
        """Analyze Test Items of the launches and publish the results.

        :param launches: Launches with request Test Items
        """
        LOGGER.info(f"Started analysis for {len(launches)} launches")
        t_start = time()
        results: list[AnalysisResult] = []
        stats_by_launch: dict[int, LaunchAnalysisStats] = {}
        chosen_namespaces: dict[int, dict[str, int]] = {}
        processed_items = 0

        try:
            for launch in launches:
                preparation_start = time()
                request_items = prepare_request_items_for_launch(launch)
                LOGGER.info(
                    "Finished preparing request test items for launch '%s', took: %.2f s.",
                    str(launch.launchId),
                    time() - preparation_start,
                )
                if self._should_stop_processing(processed_items):
                    break
                request_items = request_items[: self.search_cfg.MaxAutoAnalysisItemsToProcess - processed_items]
                processed_items += len(request_items)
                if not request_items:
                    LOGGER.info(f"No test items to analyze in project {launch.project}, launch {launch.launchId}")
                    continue

                if launch.launchId not in stats_by_launch:
                    stats_by_launch[launch.launchId] = self._create_stats(launch)
                stats = stats_by_launch[launch.launchId]
                stats.items_to_process += len(request_items)
                context = AnalysisContext(launch=launch, stats=stats, chosen_namespaces=chosen_namespaces)
                handler_results, not_found = self.chain.handle(context, request_items)
                stats.not_found += len(not_found)
                for handler_result in handler_results:
                    stats.handler_results[handler_result.handler] += 1
                    results.append(handler_result.result)
        except Exception as exc:
            LOGGER.exception("Unable to process analysis candidates", exc_info=exc)

        stats_info = {launch_id: stats.model_dump(mode="json") for launch_id, stats in stats_by_launch.items()}
        LOGGER.debug(f"Stats info: {json.dumps(stats_info)}")
        LOGGER.info(f"Processed {processed_items} test items. It took {time() - t_start:.2f} sec.")
        results_by_handler: Counter[str] = sum(
            (stats.handler_results for stats in stats_by_launch.values()), Counter()
        )
        by_handler = ", ".join(f"{handler} {count}" for handler, count in results_by_handler.most_common())
        LOGGER.info(
            f"Finished analysis for {len(launches)} launches with {len(results)} results"
            + (f": {by_handler}" if by_handler else ".")
        )
        if self.amqp_client:
            self.amqp_client.publish_response(json.dumps([res.model_dump() for res in results]))
