#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from time import time

import elasticsearch
import elasticsearch.helpers

from app.commons import logging, similarity_calculator, object_saving
from app.commons.esclient import EsClient
from app.commons.launch_objects import SearchLogInfo, Log
from app.commons.log_merger import LogMerger
from app.commons.log_preparation import LogPreparation
from app.machine_learning.models import weighted_similarity_calculator
from app.utils import utils, text_processing

logger = logging.getLogger("analyzerApp.searchService")


class SearchService:

    def __init__(self, app_config=None, search_cfg=None):
        self.app_config = app_config or {}
        self.search_cfg = search_cfg or {}
        self.es_client = EsClient(app_config=self.app_config, search_cfg=self.search_cfg)
        self.log_preparation = LogPreparation()
        self.log_merger = LogMerger()
        self.weighted_log_similarity_calculator = None
        if self.search_cfg["SimilarityWeightsFolder"].strip():
            self.weighted_log_similarity_calculator = (
                weighted_similarity_calculator.WeightedSimilarityCalculator(
                    object_saving.create_filesystem(self.search_cfg["SimilarityWeightsFolder"])))
            self.weighted_log_similarity_calculator.load_model()

    def build_search_query(self, search_req, queried_log, search_min_should_match="95%"):
        """Build search query"""
        query = {
            "_source": ["message", "test_item", "detected_message", "stacktrace",
                        "potential_status_codes", "merged_small_logs"],
            "size": self.app_config["esChunkNumber"],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}}
                    ],
                    "must_not": [{
                        "term": {"test_item": {"value": search_req.itemId, "boost": 1.0}}
                    }],
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"wildcard": {"issue_type": "TI*"}},
                                    {"wildcard": {"issue_type": "ti*"}},
                                ]
                            }
                        },
                        {"terms": {"launch_id": search_req.filteredLaunchIds}}
                    ],
                    "should": [
                        {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                    ]}}}
        if queried_log["_source"]["message"].strip():
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            query["query"]["bool"]["must"].append(
                utils.build_more_like_this_query(
                    search_min_should_match,
                    queried_log["_source"]["message"],
                    field_name="message", boost=1.0,
                    override_min_should_match=None,
                    max_query_terms=self.search_cfg["MaxQueryTerms"]))
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            query["query"]["bool"]["must"].append(
                utils.build_more_like_this_query(
                    search_min_should_match,
                    queried_log["_source"]["merged_small_logs"],
                    field_name="merged_small_logs", boost=1.0,
                    override_min_should_match=None,
                    max_query_terms=self.search_cfg["MaxQueryTerms"]))
        if queried_log["_source"]["found_exceptions"].strip():
            query["query"]["bool"]["must"].append(
                utils.build_more_like_this_query(
                    "1",
                    queried_log["_source"]["found_exceptions"],
                    field_name="found_exceptions", boost=1.0,
                    override_min_should_match="1",
                    max_query_terms=self.search_cfg["MaxQueryTerms"]))
        utils.append_potential_status_codes(query, queried_log, boost=1.0,
                                            max_query_terms=self.search_cfg["MaxQueryTerms"])
        return query

    def find_log_ids_for_test_items_with_merged_logs(self, test_item_ids, index_name, batch_size=1000):
        test_items_dict = {}
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            test_items = test_item_ids[i * batch_size: (i + 1) * batch_size]
            if not test_items:
                continue
            for r in elasticsearch.helpers.scan(self.es_client.es_client,
                                                query=self.es_client.get_test_item_query(
                                                    test_items, False, False),
                                                index=index_name):
                test_item_id = r["_source"]["test_item"]
                if test_item_id not in test_items_dict:
                    test_items_dict[test_item_id] = []
                test_items_dict[test_item_id].append(r["_id"])
        return test_items_dict

    def prepare_final_search_results(self, similar_log_ids, index_name):
        final_results = []
        all_logs_to_find_for_test_items = set()
        for log_id, test_item, is_merged in similar_log_ids:
            if is_merged:
                all_logs_to_find_for_test_items.add(test_item)
        test_items_dict = self.find_log_ids_for_test_items_with_merged_logs(
            list(all_logs_to_find_for_test_items), index_name)
        for log_id, test_item, is_merged in similar_log_ids:
            search_result_object = similar_log_ids[(log_id, test_item, is_merged)]
            if is_merged and test_item in test_items_dict:
                for log_id in test_items_dict[test_item]:
                    final_results.append(SearchLogInfo(
                        logId=utils.extract_real_id(log_id),
                        testItemId=search_result_object.testItemId,
                        matchScore=search_result_object.matchScore))
            else:
                final_results.append(search_result_object)
        return final_results

    def filter_test_items_to_have_all_messages_match(self, similar_log_ids,
                                                     test_items_found_dict, messages_length):
        new_similar_log_ids = {}
        for log_id, test_item, is_merged in similar_log_ids:
            similar_log_result_obj = (log_id, test_item, is_merged)
            if test_item in test_items_found_dict and test_items_found_dict[test_item] == messages_length:
                new_similar_log_ids[similar_log_result_obj] = similar_log_ids[similar_log_result_obj]
        return new_similar_log_ids

    def prepare_messages_for_queries(self, search_req):
        searched_logs = set()
        logs_to_query = []
        global_id = 0
        for message in search_req.logMessages:
            if not message.strip():
                continue

            queried_log = self.log_preparation._create_log_template()
            queried_log = self.log_preparation._fill_log_fields(
                queried_log,
                Log(logId=global_id, message=message),
                search_req.logLines)

            msg_words = " ".join(text_processing.split_words(queried_log["_source"]["message"]))
            if not msg_words.strip() or msg_words in searched_logs:
                continue
            searched_logs.add(msg_words)
            logs_to_query.append(queried_log)
            global_id += 1

        logs_to_query, _ = self.log_merger.decompose_logs_merged_and_without_duplicates(logs_to_query)
        return logs_to_query

    def search_similar_items_for_log(self, search_req, queried_log,
                                     search_min_should_match, test_item_info, index_name):
        query = self.build_search_query(
            search_req,
            queried_log,
            search_min_should_match=text_processing.prepare_es_min_should_match(
                search_min_should_match))
        res = []
        for r in elasticsearch.helpers.scan(self.es_client.es_client,
                                            query=query,
                                            index=index_name):
            test_item_info[r["_id"]] = r["_source"]["test_item"]
            res.append(r)
            if len(res) >= 10000:
                break
        return {"hits": {"hits": res}}

    def search_logs(self, search_req):
        """Get all logs similar to given logs"""
        similar_log_ids = {}
        logger.info(f'Started searching for test item with id: {search_req.itemId}')
        logger.debug(f'Started searching by request: {search_req.json()}')
        logger.info("ES Url %s", text_processing.remove_credentials_from_url(self.es_client.host))
        index_name = text_processing.unite_project_name(
            str(search_req.projectId), self.app_config["esProjectIndexPrefix"])
        t_start = time()
        if not self.es_client.index_exists(index_name):
            return []

        logs_to_query = self.prepare_messages_for_queries(search_req)

        global_search_min_should_match = search_req.analyzerConfig.searchLogsMinShouldMatch / 100
        test_items_found_dict = {}
        test_item_info = {}
        for queried_log in logs_to_query:
            message_to_use = queried_log["_source"]["message"]
            if not message_to_use.strip():
                message_to_use = queried_log["_source"]["merged_small_logs"]
            search_min_should_match = utils.calculate_threshold_for_text(
                message_to_use,
                global_search_min_should_match)
            search_results = self.search_similar_items_for_log(
                search_req, queried_log,
                search_min_should_match, test_item_info, index_name)

            _similarity_calculator = similarity_calculator.SimilarityCalculator(
                {
                    "max_query_terms": self.search_cfg["MaxQueryTerms"],
                    "min_word_length": self.search_cfg["MinWordLength"],
                    "number_of_log_lines": search_req.logLines
                },
                weighted_similarity_calculator=self.weighted_log_similarity_calculator)
            _similarity_calculator.find_similarity(
                [(queried_log, search_results)], ["message", "potential_status_codes", "merged_small_logs"])

            for group_id, similarity_obj in _similarity_calculator.similarity_dict["message"].items():
                log_id, _ = group_id
                similarity_percent = similarity_obj["similarity"]
                if similarity_obj["both_empty"]:
                    similarity_obj = _similarity_calculator.similarity_dict["merged_small_logs"][group_id]
                    similarity_percent = similarity_obj["similarity"]
                logger.debug("Log with id %s has %.3f similarity with the queried log '%s'",
                             log_id, similarity_percent, message_to_use)
                potential_status_codes_match = 0.0
                _similarity_dict = _similarity_calculator.similarity_dict["potential_status_codes"]
                if group_id in _similarity_dict:
                    potential_status_codes_match = _similarity_dict[group_id]["similarity"]
                if potential_status_codes_match < 0.99:
                    continue
                if similarity_percent >= search_min_should_match:
                    log_id_extracted = utils.extract_real_id(log_id)
                    is_merged = log_id != str(log_id_extracted)
                    test_item_id = int(test_item_info[log_id])
                    match_score = max(round(similarity_percent, 2),
                                      round(global_search_min_should_match, 2))
                    similar_log_ids[(log_id_extracted, test_item_id, is_merged)] = SearchLogInfo(
                        logId=log_id_extracted,
                        testItemId=test_item_id,
                        matchScore=match_score * 100)
                    if test_item_id not in test_items_found_dict:
                        test_items_found_dict[test_item_id] = 0
                    test_items_found_dict[test_item_id] += 1
        if search_req.analyzerConfig.allMessagesShouldMatch:
            similar_log_ids = self.filter_test_items_to_have_all_messages_match(
                similar_log_ids, test_items_found_dict, len(logs_to_query))
        final_results = self.prepare_final_search_results(similar_log_ids, index_name)

        logger.info("Finished searching by request %s with %d results. It took %.2f sec.",
                    search_req.json(), len(final_results), time() - t_start)
        return final_results
