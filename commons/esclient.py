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

import re
import json
import logging
import requests
import elasticsearch
import elasticsearch.helpers
import commons.launch_objects
from elasticsearch import RequestsHttpConnection
from commons.launch_objects import SuggestAnalysisResult, SearchLogInfo
import utils.utils as utils
from boosting_decision_making.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from time import time
from datetime import datetime
from boosting_decision_making import weighted_similarity_calculator
from boosting_decision_making import similarity_calculator
from boosting_decision_making import boosting_decision_maker
from boosting_decision_making import defect_type_model, custom_defect_type_model
from commons.es_query_builder import EsQueryBuilder
from commons import minio_client
from commons.log_merger import LogMerger
from queue import Queue
from commons import namespace_finder
from commons.triggering_training.retraining_defect_type_triggering import RetrainingDefectTypeTriggering
from commons.log_preparation import LogPreparation
from amqp.amqp import AmqpClient

ERROR_LOGGING_LEVEL = 40000

logger = logging.getLogger("analyzerApp.esclient")

EARLY_FINISH = False


class EsClient:
    """Elasticsearch client implementation"""
    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.host = app_config["esHost"]
        self.search_cfg = search_cfg
        self.es_client = elasticsearch.Elasticsearch([self.host], timeout=30,
                                                     max_retries=5, retry_on_timeout=True,
                                                     use_ssl=app_config["esUseSsl"],
                                                     verify_certs=app_config["esVerifyCerts"],
                                                     ssl_show_warn=app_config["esSslShowWarn"],
                                                     ca_certs=app_config["esCAcert"],
                                                     client_cert=app_config["esClientCert"],
                                                     client_key=app_config["esClientKey"])
        self.boosting_decision_maker = None
        self.suggest_decision_maker = None
        self.weighted_log_similarity_calculator = None
        self.global_defect_type_model = None
        self.suggest_threshold = 0.4
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.minio_client = minio_client.MinioClient(self.app_config)
        self.es_query_builder = EsQueryBuilder(self.search_cfg, ERROR_LOGGING_LEVEL)
        self.log_preparation = LogPreparation()
        self.model_training_triggering = {
            "defect_type": RetrainingDefectTypeTriggering(self.app_config)
        }
        self.initialize_decision_makers()

    def create_es_client(self, app_config):
        if app_config["turnOffSslVerification"]:
            return elasticsearch.Elasticsearch(
                [self.host], timeout=30,
                max_retries=5, retry_on_timeout=True,
                use_ssl=app_config["esUseSsl"],
                verify_certs=app_config["esVerifyCerts"],
                ssl_show_warn=app_config["esSslShowWarn"],
                ca_certs=app_config["esCAcert"],
                client_cert=app_config["esClientCert"],
                client_key=app_config["esClientKey"],
                connection_class=RequestsHttpConnection)
        return elasticsearch.Elasticsearch(
            [self.host], timeout=30,
            max_retries=5, retry_on_timeout=True,
            use_ssl=app_config["esUseSsl"],
            verify_certs=app_config["esVerifyCerts"],
            ssl_show_warn=app_config["esSslShowWarn"],
            ca_certs=app_config["esCAcert"],
            client_cert=app_config["esClientCert"],
            client_key=app_config["esClientKey"])

    def initialize_decision_makers(self):
        if self.search_cfg["BoostModelFolder"].strip():
            self.boosting_decision_maker = boosting_decision_maker.BoostingDecisionMaker(
                folder=self.search_cfg["BoostModelFolder"])
        if self.search_cfg["SuggestBoostModelFolder"].strip():
            self.suggest_decision_maker = boosting_decision_maker.BoostingDecisionMaker(
                folder=self.search_cfg["SuggestBoostModelFolder"])
        if self.search_cfg["SimilarityWeightsFolder"].strip():
            self.weighted_log_similarity_calculator = weighted_similarity_calculator.\
                WeightedSimilarityCalculator(folder=self.search_cfg["SimilarityWeightsFolder"])
        if self.search_cfg["GlobalDefectTypeModelFolder"].strip():
            self.global_defect_type_model = defect_type_model.\
                DefectTypeModel(folder=self.search_cfg["GlobalDefectTypeModelFolder"])

    def choose_model(self, project_id, model_name_folder):
        model = None
        if self.minio_client.does_object_exists(project_id, model_name_folder):
            folders = self.minio_client.get_folder_objects(project_id, model_name_folder)
            if len(folders):
                try:
                    model = custom_defect_type_model.CustomDefectTypeModel(
                        self.app_config, project_id, folder=folders[0])
                except Exception as err:
                    logger.error(err)
        return model

    def update_settings_after_read_only(self, es_host):
        try:
            requests.put(
                "{}/_all/_settings".format(
                    es_host
                ),
                headers={"Content-Type": "application/json"},
                data="{\"index.blocks.read_only_allow_delete\": null}"
            ).raise_for_status()
        except Exception as err:
            logger.error(err)
            logger.error("Can't reset read only mode for elastic indices")

    def create_index(self, index_name):
        """Create index in elasticsearch"""
        logger.debug("Creating '%s' Elasticsearch index", str(index_name))
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        try:
            response = self.es_client.indices.create(index=str(index_name), body={
                'settings': utils.read_json_file("", "index_settings.json", to_json=True),
                'mappings': utils.read_json_file("", "index_mapping_settings.json", to_json=True)
            })
            logger.debug("Created '%s' Elasticsearch index", str(index_name))
            return commons.launch_objects.Response(**response)
        except Exception as err:
            logger.error("Couldn't create index")
            logger.error("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.error(err)
            return commons.launch_objects.Response()

    def list_indices(self):
        """Get all indices from elasticsearch"""
        url = utils.build_url(self.host, ["_cat", "indices?format=json"])
        res = utils.send_request(url, "GET")
        return res

    def index_exists(self, index_name, print_error=True):
        """Checks whether index exists"""
        try:
            index = self.es_client.indices.get(index=str(index_name))
            return index is not None
        except Exception as err:
            if print_error:
                logger.error("Index %s was not found", str(index_name))
                logger.error("ES Url %s", self.host)
                logger.error(err)
            return False

    def delete_index(self, index_name):
        """Delete the whole index"""
        try:
            self.namespace_finder.remove_namespaces(index_name)
            for model_type in self.model_training_triggering:
                self.model_training_triggering[model_type].remove_triggering_info(
                    {"project_id": index_name})
            self.es_client.indices.delete(index=str(index_name))
            logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.debug("Deleted index %s", str(index_name))
            return 1
        except Exception as err:
            logger.error("Not found %s for deleting", str(index_name))
            logger.error("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.error(err)
            return 0

    def create_index_if_not_exists(self, index_name):
        """Creates index if it doesn't not exist"""
        if not self.index_exists(index_name, print_error=False):
            return self.create_index(index_name)
        return True

    def index_logs(self, launches):
        """Index launches to the index with project name"""
        cnt_launches = len(launches)
        logger.info("Indexing logs for %d launches", cnt_launches)
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        t_start = time()
        bodies = []
        test_item_ids = []
        project = None
        test_item_queue = Queue()
        for launch in launches:
            project = str(launch.project)
            test_items = launch.testItems
            launch.testItems = []
            self.create_index_if_not_exists(str(launch.project))
            for test_item in test_items:
                test_item_queue.put((launch, test_item))
        del launches
        while not test_item_queue.empty():
            launch, test_item = test_item_queue.get()
            logs_added = False
            for log in test_item.logs:
                if log.logLevel < ERROR_LOGGING_LEVEL or not log.message.strip():
                    continue

                bodies.append(self.log_preparation._prepare_log(launch, test_item, log))
                logs_added = True
            if logs_added:
                test_item_ids.append(str(test_item.testItemId))

        logs_with_exceptions = utils.extract_all_exceptions(bodies)
        result = self._bulk_index(bodies)
        result.logResults = logs_with_exceptions
        _, num_logs_with_defect_types = self._merge_logs(test_item_ids, project)
        try:
            if "amqpUrl" in self.app_config and self.app_config["amqpUrl"].strip():
                AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                    self.app_config["exchangeName"], "train_models", json.dumps({
                        "model_type": "defect_type",
                        "project_id": project,
                        "num_logs_with_defect_types": num_logs_with_defect_types
                    }))
        except Exception as err:
            logger.error(err)
        logger.info("Finished indexing logs for %d launches. It took %.2f sec.",
                    cnt_launches, time() - t_start)
        return result

    def find_min_should_match_threshold(self, analyzer_config):
        return analyzer_config.minShouldMatch if analyzer_config.minShouldMatch > 0 else\
            int(re.search(r"\d+", self.search_cfg["MinShouldMatch"]).group(0))

    def _merge_logs(self, test_item_ids, project):
        bodies = []
        batch_size = 1000
        self._delete_merged_logs(test_item_ids, project)
        num_logs_with_defect_types = 0
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            test_items = test_item_ids[i * batch_size: (i + 1) * batch_size]
            if not test_items:
                continue
            test_items_dict = {}
            for r in elasticsearch.helpers.scan(self.es_client,
                                                query=self.es_query_builder.get_test_item_query(
                                                    test_items, False),
                                                index=project):
                test_item_id = r["_source"]["test_item"]
                if test_item_id not in test_items_dict:
                    test_items_dict[test_item_id] = []
                test_items_dict[test_item_id].append(r)
            for test_item_id in test_items_dict:
                merged_logs = LogMerger.decompose_logs_merged_and_without_duplicates(
                    test_items_dict[test_item_id])
                for log in merged_logs:
                    if log["_source"]["is_merged"]:
                        bodies.append(log)
                    else:
                        bodies.append({
                            "_op_type": "update",
                            "_id": log["_id"],
                            "_index": log["_index"],
                            "doc": {"merged_small_logs": log["_source"]["merged_small_logs"]}
                        })
                    log_issue_type = log["_source"]["issue_type"]
                    if log_issue_type.strip() and not log_issue_type.lower().startswith("ti"):
                        num_logs_with_defect_types += 1
        return self._bulk_index(bodies), num_logs_with_defect_types

    def _delete_merged_logs(self, test_items_to_delete, project):
        logger.debug("Delete merged logs for %d test items", len(test_items_to_delete))
        bodies = []
        batch_size = 1000
        for i in range(int(len(test_items_to_delete) / batch_size) + 1):
            test_item_ids = test_items_to_delete[i * batch_size: (i + 1) * batch_size]
            if not test_item_ids:
                continue
            for log in elasticsearch.helpers.scan(self.es_client,
                                                  query=self.es_query_builder.get_test_item_query(
                                                      test_item_ids, True),
                                                  index=project):
                bodies.append({
                    "_op_type": "delete",
                    "_id": log["_id"],
                    "_index": project
                })
        if bodies:
            self._bulk_index(bodies)

    def _bulk_index(self, bodies, host=None, es_client=None, refresh=True):
        if host is None:
            host = self.host
        if es_client is None:
            es_client = self.es_client
        if not bodies:
            return commons.launch_objects.BulkResponse(took=0, errors=False)
        logger.debug("Indexing %d logs...", len(bodies))
        try:
            try:
                success_count, errors = elasticsearch.helpers.bulk(es_client,
                                                                   bodies,
                                                                   chunk_size=1000,
                                                                   request_timeout=30,
                                                                   refresh=refresh)
            except Exception as err:
                logger.error(err)
                self.update_settings_after_read_only(host)
                success_count, errors = elasticsearch.helpers.bulk(es_client,
                                                                   bodies,
                                                                   chunk_size=1000,
                                                                   request_timeout=30,
                                                                   refresh=refresh)
            logger.debug("Processed %d logs", success_count)
            if errors:
                logger.debug("Occured errors %s", errors)
            return commons.launch_objects.BulkResponse(took=success_count, errors=len(errors) > 0)
        except Exception as err:
            logger.error("Error in bulk")
            logger.error("ES Url %s", utils.remove_credentials_from_url(host))
            logger.error(err)
            return commons.launch_objects.BulkResponse(took=0, errors=True)

    def delete_logs(self, clean_index):
        """Delete logs from elasticsearch"""
        logger.info("Delete logs %s for the project %s",
                    clean_index.ids, clean_index.project)
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        t_start = time()
        if not self.index_exists(clean_index.project):
            return 0
        test_item_ids = set()
        try:
            search_query = self.es_query_builder.build_search_test_item_ids_query(
                clean_index.ids)
            for res in elasticsearch.helpers.scan(self.es_client,
                                                  query=search_query,
                                                  index=clean_index.project,
                                                  scroll="5m"):
                test_item_ids.add(res["_source"]["test_item"])
        except Exception as err:
            logger.error("Couldn't find test items for logs")
            logger.error(err)

        bodies = []
        for _id in clean_index.ids:
            bodies.append({
                "_op_type": "delete",
                "_id":      _id,
                "_index":   clean_index.project,
            })
        result = self._bulk_index(bodies)
        self._merge_logs(list(test_item_ids), clean_index.project)
        logger.info("Finished deleting logs %s for the project %s. It took %.2f sec",
                    clean_index.ids, clean_index.project, time() - t_start)
        return result.took

    def search_logs(self, search_req):
        """Get all logs similar to given logs"""
        similar_log_ids = set()
        logger.info("Started searching by request %s", search_req.json())
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        t_start = time()
        if not self.index_exists(str(search_req.projectId)):
            return []
        searched_logs = set()
        test_item_info = {}

        for message in search_req.logMessages:
            if not message.strip():
                continue

            queried_log = self.log_preparation._create_log_template()
            queried_log = self.log_preparation._fill_log_fields(
                queried_log,
                commons.launch_objects.Log(logId=0, message=message),
                search_req.logLines)

            msg_words = " ".join(utils.split_words(queried_log["_source"]["message"]))
            if not msg_words.strip() or msg_words in searched_logs:
                continue
            searched_logs.add(msg_words)
            query = self.es_query_builder.build_search_query(
                search_req, queried_log["_source"]["message"])
            res = self.es_client.search(index=str(search_req.projectId), body=query)
            for es_res in res["hits"]["hits"]:
                test_item_info[es_res["_id"]] = es_res["_source"]["test_item"]

            _similarity_calculator = similarity_calculator.SimilarityCalculator(
                {
                    "max_query_terms": self.search_cfg["MaxQueryTerms"],
                    "min_word_length": self.search_cfg["MinWordLength"],
                    "min_should_match": "90%",
                    "number_of_log_lines": search_req.logLines
                },
                weighted_similarity_calculator=self.weighted_log_similarity_calculator)
            _similarity_calculator.find_similarity([(queried_log, res)], ["message"])

            for group_id, similarity_obj in _similarity_calculator.similarity_dict["message"].items():
                log_id, _ = group_id
                similarity_percent = similarity_obj["similarity"]
                logger.debug("Log with id %s has %.3f similarity with the queried log '%s'",
                             log_id, similarity_percent, queried_log["_source"]["message"])
                if similarity_percent >= self.search_cfg["SearchLogsMinSimilarity"]:
                    similar_log_ids.add((utils.extract_real_id(log_id), int(test_item_info[log_id])))

        logger.info("Finished searching by request %s with %d results. It took %.2f sec.",
                    search_req.json(), len(similar_log_ids), time() - t_start)
        return [SearchLogInfo(logId=log_info[0],
                              testItemId=log_info[1]) for log_info in similar_log_ids]

    def get_config_for_boosting_suggests(self, analyzerConfig):
        return {
            "max_query_terms": self.search_cfg["MaxQueryTerms"],
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg["MinWordLength"],
            "filter_min_should_match": [],
            "filter_min_should_match_any": utils.choose_fields_to_filter_suggests(
                analyzerConfig.numberOfLogLines),
            "number_of_log_lines": analyzerConfig.numberOfLogLines,
            "filter_by_unique_id": True}

    def query_es_for_suggested_items(self, test_item_info, logs):
        full_results = []

        for log in logs:
            message = log["_source"]["message"].strip()
            merged_small_logs = log["_source"]["merged_small_logs"].strip()
            if log["_source"]["log_level"] < ERROR_LOGGING_LEVEL or\
                    (not message and not merged_small_logs):
                continue

            query = self.es_query_builder.build_suggest_query(
                test_item_info, log,
                message_field="message_extended",
                det_mes_field="detected_message_extended",
                stacktrace_field="stacktrace_extended")
            es_res = self.es_client.search(index=str(test_item_info.project), body=query)
            full_results.append((log, es_res))

            query = self.es_query_builder.build_suggest_query(
                test_item_info, log,
                message_field="message_without_params_extended",
                det_mes_field="detected_message_without_params_extended",
                stacktrace_field="stacktrace_extended")
            es_res = self.es_client.search(index=str(test_item_info.project), body=query)
            full_results.append((log, es_res))

            query = self.es_query_builder.build_suggest_query(
                test_item_info, log,
                message_field="message_without_params_and_brackets",
                det_mes_field="detected_message_without_params_and_brackets",
                stacktrace_field="stacktrace_extended")
            es_res = self.es_client.search(index=str(test_item_info.project), body=query)
            full_results.append((log, es_res))
        return full_results

    def deduplicate_results(self, gathered_results, scores_by_test_items, test_item_ids):
        _similarity_calculator = similarity_calculator.SimilarityCalculator(
            {
                "max_query_terms": self.search_cfg["MaxQueryTerms"],
                "min_word_length": self.search_cfg["MinWordLength"],
                "min_should_match": "98%",
                "number_of_log_lines": -1
            },
            weighted_similarity_calculator=self.weighted_log_similarity_calculator)
        all_pairs_to_check = []
        for i in range(len(gathered_results)):
            for j in range(i + 1, len(gathered_results)):
                test_item_id_first = test_item_ids[gathered_results[i][0]]
                test_item_id_second = test_item_ids[gathered_results[j][0]]
                issue_type1 = scores_by_test_items[test_item_id_first]["mrHit"]["_source"]["issue_type"]
                issue_type2 = scores_by_test_items[test_item_id_second]["mrHit"]["_source"]["issue_type"]
                if issue_type1 != issue_type2:
                    continue
                items_to_compare = {"hits": {"hits": [scores_by_test_items[test_item_id_first]["mrHit"]]}}
                all_pairs_to_check.append((scores_by_test_items[test_item_id_second]["mrHit"],
                                           items_to_compare))
        _similarity_calculator.find_similarity(
            all_pairs_to_check,
            ["detected_message_with_numbers", "stacktrace", "merged_small_logs"])

        filtered_results = []
        deleted_indices = set()
        for i in range(len(gathered_results)):
            if i in deleted_indices:
                continue
            for j in range(i + 1, len(gathered_results)):
                test_item_id_first = test_item_ids[gathered_results[i][0]]
                test_item_id_second = test_item_ids[gathered_results[j][0]]
                group_id = (scores_by_test_items[test_item_id_first]["mrHit"]["_id"],
                            scores_by_test_items[test_item_id_second]["mrHit"]["_id"])
                if group_id not in _similarity_calculator.similarity_dict["detected_message_with_numbers"]:
                    continue
                det_message = _similarity_calculator.similarity_dict["detected_message_with_numbers"]
                detected_message_sim = det_message[group_id]
                stacktrace_sim = _similarity_calculator.similarity_dict["stacktrace"][group_id]
                merged_logs_sim = _similarity_calculator.similarity_dict["merged_small_logs"][group_id]
                if detected_message_sim["similarity"] >= 0.98 and\
                        stacktrace_sim["similarity"] >= 0.98 and merged_logs_sim["similarity"] >= 0.98:
                    deleted_indices.add(j)
            filtered_results.append(gathered_results[i])
        return filtered_results

    def sort_results(self, scores_by_test_items, test_item_ids, predicted_labels_probability):
        gathered_results = []
        for idx, prob in enumerate(predicted_labels_probability):
            test_item_id = test_item_ids[idx]
            gathered_results.append(
                (idx,
                 round(prob[1], 2),
                 scores_by_test_items[test_item_id]["mrHit"]["_source"]["start_time"]))

        gathered_results = sorted(gathered_results, key=lambda x: (x[1], x[2]), reverse=True)
        return self.deduplicate_results(gathered_results, scores_by_test_items, test_item_ids)

    @utils.ignore_warnings
    def suggest_items(self, test_item_info, num_items=5):
        logger.info("Started suggesting test items")
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        if not self.index_exists(str(test_item_info.project)):
            logger.info("Project %d doesn't exist", test_item_info.project)
            logger.info("Finished suggesting for test item with 0 results.")
            return []

        t_start = time()
        results = []
        unique_logs = utils.leave_only_unique_logs(test_item_info.logs)
        prepared_logs = [self.log_preparation._prepare_log_for_suggests(test_item_info, log)
                         for log in unique_logs if log.logLevel >= ERROR_LOGGING_LEVEL]
        logs = LogMerger.decompose_logs_merged_and_without_duplicates(prepared_logs)
        searched_res = self.query_es_for_suggested_items(test_item_info, logs)

        boosting_config = self.get_config_for_boosting_suggests(test_item_info.analyzerConfig)
        boosting_config["chosen_namespaces"] = self.namespace_finder.get_chosen_namespaces(
            test_item_info.project)

        _boosting_data_gatherer = SuggestBoostingFeaturizer(
            searched_res,
            boosting_config,
            feature_ids=self.suggest_decision_maker.get_feature_ids(),
            weighted_log_similarity_calculator=self.weighted_log_similarity_calculator)
        defect_type_model_to_use = self.choose_model(
            test_item_info.project, "defect_type_model/")
        if defect_type_model_to_use is None:
            _boosting_data_gatherer.set_defect_type_model(self.global_defect_type_model)
        else:
            _boosting_data_gatherer.set_defect_type_model(defect_type_model_to_use)
        feature_data, test_item_ids = _boosting_data_gatherer.gather_features_info()
        scores_by_test_items = _boosting_data_gatherer.scores_by_issue_type
        model_info_tags = _boosting_data_gatherer.get_used_model_info() +\
            self.suggest_decision_maker.get_model_info()

        if feature_data:
            predicted_labels, predicted_labels_probability = self.suggest_decision_maker.predict(feature_data)
            sorted_results = self.sort_results(
                scores_by_test_items, test_item_ids, predicted_labels_probability)

            logger.debug("Found %d results for test items ", len(sorted_results))
            for idx, prob, _ in sorted_results:
                test_item_id = test_item_ids[idx]
                issue_type = scores_by_test_items[test_item_id]["mrHit"]["_source"]["issue_type"]
                logger.debug("Test item id %d with issue type %s has probability %.2f",
                             test_item_id, issue_type, prob)

            global_idx = 0
            for idx, prob, _ in sorted_results[:num_items]:
                if prob >= self.suggest_threshold:
                    test_item_id = test_item_ids[idx]
                    issue_type = scores_by_test_items[test_item_id]["mrHit"]["_source"]["issue_type"]
                    relevant_log_id = utils.extract_real_id(
                        scores_by_test_items[test_item_id]["mrHit"]["_id"])
                    test_item_log_id = utils.extract_real_id(
                        scores_by_test_items[test_item_id]["compared_log"]["_id"])
                    analysis_result = SuggestAnalysisResult(
                        testItem=test_item_info.testItemId,
                        testItemLogId=test_item_log_id,
                        issueType=issue_type,
                        relevantItem=test_item_id,
                        relevantLogId=relevant_log_id,
                        matchScore=round(prob * 100, 2),
                        esScore=round(scores_by_test_items[test_item_id]["mrHit"]["_score"], 2),
                        esPosition=scores_by_test_items[test_item_id]["mrHit"]["es_pos"],
                        modelFeatureNames=";".join(
                            [str(feature) for feature in self.suggest_decision_maker.get_feature_ids()]),
                        modelFeatureValues=";".join(
                            [str(feature) for feature in feature_data[idx]]),
                        modelInfo=";".join(model_info_tags),
                        resultPosition=global_idx,
                        usedLogLines=test_item_info.analyzerConfig.numberOfLogLines,
                        minShouldMatch=self.find_min_should_match_threshold(test_item_info.analyzerConfig))
                    results.append(analysis_result)
                    logger.debug(analysis_result)
                global_idx += 1
        else:
            logger.debug("There are no results for test item %s", test_item_info.testItemId)
        results_to_share = {test_item_info.launchId: {
            "not_found": int(len(results) == 0), "items_to_process": 1,
            "processed_time": time() - t_start, "found_items": len(results),
            "launch_id": test_item_info.launchId, "launch_name": test_item_info.launchName,
            "project_id": test_item_info.project, "method": "suggest",
            "gather_date": datetime.now().strftime("%Y-%m-%d"),
            "gather_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "number_of_log_lines": test_item_info.analyzerConfig.numberOfLogLines,
            "model_info": model_info_tags,
            "module_version": self.app_config["appVersion"],
            "min_should_match": self.find_min_should_match_threshold(
                test_item_info.analyzerConfig)}}
        if "amqpUrl" in self.app_config and self.app_config["amqpUrl"].strip():
            AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                self.app_config["exchangeName"], "stats_info", json.dumps(results_to_share))

        logger.debug("Stats info %s", results_to_share)
        logger.info("Processed the test item. It took %.2f sec.", time() - t_start)
        logger.info("Finished suggesting for test item with %d results.", len(results))
        return results

    def create_index_for_stats_info(self, es_client, rp_aa_stats_index):
        index = None
        try:
            index = es_client.indices.get(index=rp_aa_stats_index)
        except Exception:
            pass
        if index is None:
            es_client.indices.create(index=rp_aa_stats_index, body={
                'settings': utils.read_json_file("", "index_settings.json", to_json=True),
                'mappings': utils.read_json_file(
                    "", "rp_aa_stats_mappings.json", to_json=True)
            })
        else:
            es_client.indices.put_mapping(
                index=rp_aa_stats_index,
                body=utils.read_json_file("", "rp_aa_stats_mappings.json", to_json=True))

    @utils.ignore_warnings
    def send_stats_info(self, stats_info):
        rp_aa_stats_index = "rp_aa_stats"
        logger.info("Started sending stats about analysis")
        self.create_index_for_stats_info(self.es_client, rp_aa_stats_index)

        stat_info_array = []
        for launch_id in stats_info:
            stat_info_array.append({
                "_index": rp_aa_stats_index,
                "_source": stats_info[launch_id]
            })
        self._bulk_index(stat_info_array)
        logger.info("Finished sending stats about analysis")

    @utils.ignore_warnings
    def update_chosen_namespaces(self, launches):
        logger.info("Started updating chosen namespaces")
        t_start = time()
        log_words, project_id = self.log_preparation.prepare_log_words(launches)
        logger.debug("Project id %s", project_id)
        logger.debug("Found namespaces %s", log_words)
        if project_id is not None:
            self.namespace_finder.update_namespaces(
                project_id, log_words)
        logger.info("Finished updating chosen namespaces %.2f s", time() - t_start)

    @utils.ignore_warnings
    def train_models(self, train_info):
        logger.info("Started training")
        t_start = time()
        assert train_info["model_type"] in self.model_training_triggering

        _retraining_defect_type_triggering = self.model_training_triggering[train_info["model_type"]]
        if _retraining_defect_type_triggering.should_model_training_be_triggered(train_info):
            print("Should be trained ", train_info)
        logger.info("Finished training %.2f s", time() - t_start)
