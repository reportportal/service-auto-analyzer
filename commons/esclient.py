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

import json
import logging
import requests
import urllib3
import traceback
import elasticsearch
import elasticsearch.helpers
import commons.launch_objects
from elasticsearch import RequestsHttpConnection
import utils.utils as utils
from time import time
from commons.log_merger import LogMerger
from queue import Queue
from commons.log_preparation import LogPreparation
from amqp.amqp import AmqpClient

logger = logging.getLogger("analyzerApp.esclient")


class EsClient:
    """Elasticsearch client implementation"""
    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.host = app_config["esHost"]
        self.search_cfg = search_cfg
        self.es_client = self.create_es_client(app_config)
        self.log_preparation = LogPreparation()
        self.tables_to_recreate = ["rp_aa_stats", "rp_model_train_stats",
                                   "rp_suggestions_info_metrics"]

    def create_es_client(self, app_config):
        if not app_config["esVerifyCerts"]:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        if app_config["turnOffSslVerification"]:
            return elasticsearch.Elasticsearch(
                [self.host], timeout=30,
                max_retries=5, retry_on_timeout=True,
                http_auth=(app_config["esUser"], app_config["esPassword"]),
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
            http_auth=(app_config["esUser"], app_config["esPassword"]),
            use_ssl=app_config["esUseSsl"],
            verify_certs=app_config["esVerifyCerts"],
            ssl_show_warn=app_config["esSslShowWarn"],
            ca_certs=app_config["esCAcert"],
            client_cert=app_config["esClientCert"],
            client_key=app_config["esClientKey"])

    def get_test_item_query(self, test_item_ids, is_merged, full_log):
        """Build test item query"""
        if full_log:
            return {
                "_source": ["message", "test_item", "log_level", "found_exceptions",
                            "potential_status_codes", "original_message_lines",
                            "original_message_words_number", "issue_type", "launch_id",
                            "launch_name", "unique_id", "test_case_hash", "start_time",
                            "is_auto_analyzed", "cluster_id", "cluster_message"],
                "size": self.app_config["esChunkNumber"],
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"test_item": [str(_id) for _id in test_item_ids]}},
                            {"term": {"is_merged": is_merged}}
                        ]
                    }
                }}
        else:
            return {
                "_source": ["test_item"],
                "size": self.app_config["esChunkNumber"],
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"test_item": [str(_id) for _id in test_item_ids]}},
                            {"term": {"is_merged": is_merged}}
                        ]
                    }
                }}

    def build_search_test_item_ids_query(self, log_ids):
        """Build search test item ids query"""
        return {
            "_source": ["test_item"],
            "size": self.app_config["esChunkNumber"],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}},
                        {"terms": {"_id": [str(log_id) for log_id in log_ids]}},
                    ]
                }
            }}

    def is_healthy(self, es_host_name):
        """Check whether elasticsearch is healthy"""
        try:
            url = utils.build_url(self.host, ["_cluster/health"])
            res = utils.send_request(url, "GET", self.app_config["esUser"], self.app_config["esPassword"])
            return res["status"] in ["green", "yellow"]
        except Exception as err:
            logger.error("Elasticsearch is not healthy")
            logger.error(err)
            return False

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
        res = utils.send_request(url, "GET", self.app_config["esUser"], self.app_config["esPassword"])
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
            self.es_client.indices.delete(index=str(index_name))
            logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.debug("Deleted index %s", str(index_name))
            return True
        except Exception as err:
            logger.error("Not found %s for deleting", str(index_name))
            logger.error("ES Url %s", utils.remove_credentials_from_url(self.host))
            logger.error(err)
            return False

    def create_index_if_not_exists(self, index_name):
        """Creates index if it doesn't not exist"""
        if not self.index_exists(index_name, print_error=False):
            return self.create_index(index_name)
        return True

    def index_logs(self, launches):
        """Index launches to the index with project name"""
        launch_ids = set()
        logger.info("Indexing logs for %d launches", len(launches))
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
            for test_item in test_items:
                test_item_queue.put((launch, test_item))
                launch_ids.add(launch.launchId)
        del launches
        project_with_prefix = utils.unite_project_name(
            project, self.app_config["esProjectIndexPrefix"])
        self.create_index_if_not_exists(project_with_prefix)
        while not test_item_queue.empty():
            launch, test_item = test_item_queue.get()
            logs_added = False
            for log in test_item.logs:
                if log.logLevel < utils.ERROR_LOGGING_LEVEL or not log.message.strip():
                    continue

                bodies.append(self.log_preparation._prepare_log(
                    launch, test_item, log, project_with_prefix))
                logs_added = True
            if logs_added:
                test_item_ids.append(str(test_item.testItemId))

        logs_with_exceptions = utils.extract_all_exceptions(bodies)
        result = self._bulk_index(bodies)
        result.logResults = logs_with_exceptions
        _, num_logs_with_defect_types = self._merge_logs(test_item_ids, project_with_prefix)
        try:
            if "amqpUrl" in self.app_config and self.app_config["amqpUrl"].strip():
                AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                    self.app_config["exchangeName"], "train_models", json.dumps({
                        "model_type": "defect_type",
                        "project_id": project,
                        "gathered_metric_total": num_logs_with_defect_types
                    }))
        except Exception as err:
            logger.error(err)
        logger.info("Finished indexing logs for %d launches %s. It took %.2f sec.",
                    len(launch_ids), launch_ids, time() - t_start)
        return result

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
                                                query=self.get_test_item_query(
                                                    test_items, False, True),
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
                                                  query=self.get_test_item_query(
                                                      test_item_ids, True, False),
                                                  index=project):
                bodies.append({
                    "_op_type": "delete",
                    "_id": log["_id"],
                    "_index": project
                })
        if bodies:
            self._bulk_index(bodies)

    def _recreate_index_if_needed(self, bodies, formatted_exception):
        index_name = ""
        if bodies:
            index_name = bodies[0]["_index"]
        if not index_name.strip():
            return
        if "'type': 'mapper_parsing_exception'" in formatted_exception or\
                "RequestError(400, 'illegal_argument_exception'" in formatted_exception:
            if index_name in self.tables_to_recreate:
                self.delete_index(index_name)
                self.create_index_for_stats_info(index_name)

    def _bulk_index(self, bodies, host=None, es_client=None, refresh=True):
        if host is None:
            host = self.host
        if es_client is None:
            es_client = self.es_client
        if not bodies:
            return commons.launch_objects.BulkResponse(took=0, errors=False)
        start_time = time()
        logger.debug("Indexing %d logs...", len(bodies))
        es_chunk_number = self.app_config["esChunkNumber"]
        try:
            try:
                success_count, errors = elasticsearch.helpers.bulk(es_client,
                                                                   bodies,
                                                                   chunk_size=es_chunk_number,
                                                                   request_timeout=30,
                                                                   refresh=refresh)
            except: # noqa
                formatted_exception = traceback.format_exc()
                self._recreate_index_if_needed(bodies, formatted_exception)
                self.update_settings_after_read_only(host)
                success_count, errors = elasticsearch.helpers.bulk(es_client,
                                                                   bodies,
                                                                   chunk_size=es_chunk_number,
                                                                   request_timeout=30,
                                                                   refresh=refresh)
            logger.debug("Processed %d logs", success_count)
            if errors:
                logger.debug("Occured errors %s", errors)
            logger.debug("Finished indexing for %.2f s", time() - start_time)
            return commons.launch_objects.BulkResponse(took=success_count, errors=len(errors) > 0)
        except Exception as err:
            logger.error("Error in bulk")
            logger.error("ES Url %s", utils.remove_credentials_from_url(host))
            logger.error(err)
            return commons.launch_objects.BulkResponse(took=0, errors=True)

    def delete_logs(self, clean_index):
        """Delete logs from elasticsearch"""
        index_name = utils.unite_project_name(
            str(clean_index.project), self.app_config["esProjectIndexPrefix"])
        logger.info("Delete logs %s for the project %s",
                    clean_index.ids, index_name)
        logger.info("ES Url %s", utils.remove_credentials_from_url(self.host))
        t_start = time()
        if not self.index_exists(index_name):
            return 0
        test_item_ids = set()
        try:
            search_query = self.build_search_test_item_ids_query(
                clean_index.ids)
            for res in elasticsearch.helpers.scan(self.es_client,
                                                  query=search_query,
                                                  index=index_name,
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
                "_index":   index_name,
            })
        result = self._bulk_index(bodies)
        self._merge_logs(list(test_item_ids), index_name)
        logger.info("Finished deleting logs %s for the project %s. It took %.2f sec",
                    clean_index.ids, index_name, time() - t_start)
        return result.took

    def create_index_for_stats_info(self, rp_aa_stats_index, override_index_name=None):
        index_name = rp_aa_stats_index
        if override_index_name is not None:
            index_name = override_index_name
        index = None
        try:
            index = self.es_client.indices.get(index=index_name)
        except Exception:
            pass
        if index is None:
            self.es_client.indices.create(index=index_name, body={
                'settings': utils.read_json_file("", "index_settings.json", to_json=True),
                'mappings': utils.read_json_file(
                    "", "%s_mappings.json" % rp_aa_stats_index, to_json=True)
            })
        else:
            try:
                self.es_client.indices.put_mapping(
                    index=index_name,
                    body=utils.read_json_file("", "%s_mappings.json" % rp_aa_stats_index, to_json=True))
            except: # noqa
                formatted_exception = traceback.format_exc()
                self._recreate_index_if_needed([{"_index": index_name}], formatted_exception)

    @utils.ignore_warnings
    def send_stats_info(self, stats_info):
        logger.info("Started sending stats about analysis")

        stat_info_array = []
        for launch_id in stats_info:
            obj_info = stats_info[launch_id]
            rp_aa_stats_index = "rp_aa_stats"
            if "method" in obj_info and obj_info["method"] == "training":
                rp_aa_stats_index = "rp_model_train_stats"
            self.create_index_for_stats_info(rp_aa_stats_index)
            stat_info_array.append({
                "_index": rp_aa_stats_index,
                "_source": obj_info
            })
        self._bulk_index(stat_info_array)
        logger.info("Finished sending stats about analysis")

    def get_test_items_by_ids_query(self, test_item_ids):
        return {"_source": ["test_item"],
                "size": self.app_config["esChunkNumber"],
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"test_item": test_item_ids}}
                        ]
                    }}}

    @utils.ignore_warnings
    def defect_update(self, defect_update_info):
        logger.info("Started updating defect types")
        t_start = time()
        test_item_ids = [int(key_) for key_ in defect_update_info["itemsToUpdate"].keys()]
        defect_update_info["itemsToUpdate"] = {
            int(key_): val for key_, val in defect_update_info["itemsToUpdate"].items()}
        index_name = utils.unite_project_name(
            str(defect_update_info["project"]), self.app_config["esProjectIndexPrefix"])
        if not self.index_exists(index_name):
            return test_item_ids
        batch_size = 1000
        log_update_queries = []
        found_test_items = set()
        for i in range(int(len(test_item_ids) / batch_size) + 1):
            sub_test_item_ids = test_item_ids[i * batch_size: (i + 1) * batch_size]
            if not sub_test_item_ids:
                continue
            for log in elasticsearch.helpers.scan(self.es_client,
                                                  query=self.get_test_items_by_ids_query(sub_test_item_ids),
                                                  index=index_name):
                issue_type = ""
                try:
                    test_item_id = int(log["_source"]["test_item"])
                    found_test_items.add(test_item_id)
                    issue_type = defect_update_info["itemsToUpdate"][test_item_id]
                except: # noqa
                    pass
                if issue_type.strip():
                    log_update_queries.append({
                        "_op_type": "update",
                        "_id": log["_id"],
                        "_index": index_name,
                        "doc": {
                            "issue_type": issue_type
                        }
                    })
        self._bulk_index(log_update_queries)
        items_not_updated = list(set(test_item_ids) - found_test_items)
        logger.debug("Not updated test items: %s", items_not_updated)
        logger.info("Finished updating defect types. It took %.2f sec", time() - t_start)
        return items_not_updated

    def build_delete_query_by_test_items(self, sub_test_item_ids):
        return {"query": {
                "bool": {
                    "filter": [
                        {"terms": {"test_item": sub_test_item_ids}}
                    ]
                }}}

    def build_delete_query_by_launch_ids(self, launch_ids):
        return {"query": {"bool": {"filter": [{"terms": {"launch_id": launch_ids}}]}}}

    @utils.ignore_warnings
    def remove_test_items(self, remove_items_info):
        logger.info("Started removing test items")
        t_start = time()
        index_name = utils.unite_project_name(
            str(remove_items_info["project"]), self.app_config["esProjectIndexPrefix"])
        deleted_logs = self.delete_by_query(
            index_name, remove_items_info["itemsToDelete"], self.build_delete_query_by_test_items)
        logger.debug("Removed %s logs by test item ids", deleted_logs)
        logger.info("Finished removing test items. It took %.2f sec", time() - t_start)
        return deleted_logs

    @utils.ignore_warnings
    def remove_launches(self, remove_launches_info):
        project = remove_launches_info["project"]
        launch_ids = remove_launches_info["launch_ids"]
        logger.info("Started removing launches")
        t_start = time()
        index_name = utils.unite_project_name(
            str(project), self.app_config["esProjectIndexPrefix"]
        )
        deleted_logs = self.delete_by_query(
            index_name,
            launch_ids,
            self.build_delete_query_by_launch_ids,
        )
        logger.debug("Removed %s logs by launch ids", deleted_logs)
        logger.info("Finished removing launches. It took %.2f sec", time() - t_start)
        return deleted_logs

    @utils.ignore_warnings
    def delete_by_query(self, index_name, ids_for_removal, delete_query_deriver):
        if not self.index_exists(index_name):
            return 0
        batch_size = 1000
        deleted_logs = 0
        for i in range(int(len(ids_for_removal) / batch_size) + 1):
            sub_ids_for_removal = ids_for_removal[i * batch_size: (i + 1) * batch_size]
            if not sub_ids_for_removal:
                continue
            result = self.es_client.delete_by_query(
                index_name, body=delete_query_deriver(sub_ids_for_removal))
            if "deleted" in result:
                deleted_logs += result["deleted"]
        return deleted_logs
