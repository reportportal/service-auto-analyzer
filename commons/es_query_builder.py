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


class EsQueryBuilder:

    def __init__(self, search_cfg, error_logging_level):
        self.search_cfg = search_cfg
        self.error_logging_level = error_logging_level

    def get_test_item_query(self, test_item_ids, is_merged):
        """Build test item query"""
        return {"size": 10000,
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
        return {"size": 10000,
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"log_level": {"gte": self.error_logging_level}}},
                            {"exists": {"field": "issue_type"}},
                            {"term": {"is_merged": False}},
                            {"terms": {"_id": [str(log_id) for log_id in log_ids]}},
                        ]
                    }
                }, }

    def build_search_query(self, search_req, message):
        """Build search query"""
        return {
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": self.error_logging_level}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}},
                    ],
                    "must_not": {
                        "term": {"test_item": {"value": search_req.itemId, "boost": 1.0}}
                    },
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"wildcard": {"issue_type": "TI*"}},
                                    {"wildcard": {"issue_type": "ti*"}},
                                ]
                            }
                        },
                        {"terms": {"launch_id": search_req.filteredLaunchIds}},
                        self.
                        build_more_like_this_query("90%",
                                                   message,
                                                   field_name="message"),
                    ],
                    "should": [
                        {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                    ]}}}

    def build_search_similar_items_query(self, launch_id, test_item, message):
        """Build search query"""
        return {
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": self.error_logging_level}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}},
                    ],
                    "must_not": {
                        "term": {"test_item": {"value": test_item, "boost": 1.0}}
                    },
                    "must": [
                        {"term": {"launch_id": launch_id}},
                        self.
                        build_more_like_this_query("98%",
                                                   message,
                                                   field_name="whole_message")
                    ]}}}

    def build_more_like_this_query(self,
                                   min_should_match, log_message,
                                   field_name="message", boost=1.0,
                                   override_min_should_match=None):
        """Build more like this query"""
        return {"more_like_this": {
            "fields":               [field_name],
            "like":                 log_message,
            "min_doc_freq":         1,
            "min_term_freq":        1,
            "minimum_should_match":
                ("5<" + min_should_match) if override_min_should_match is None else override_min_should_match,
            "max_query_terms":      self.search_cfg["MaxQueryTerms"],
            "boost": boost, }}

    def build_common_query(self, log, size=10):
        return {"size": size,
                "sort": ["_score",
                         {"start_time": "desc"}, ],
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"log_level": {"gte": self.error_logging_level}}},
                            {"exists": {"field": "issue_type"}},
                        ],
                        "must_not": [
                            {"wildcard": {"issue_type": "TI*"}},
                            {"wildcard": {"issue_type": "ti*"}},
                            {"wildcard": {"issue_type": "nd*"}},
                            {"wildcard": {"issue_type": "ND*"}},
                            {"term": {"test_item": log["_source"]["test_item"]}}
                        ],
                        "must": [],
                        "should": [
                            {"term": {"unique_id": {
                                "value": log["_source"]["unique_id"],
                                "boost": abs(self.search_cfg["BoostUniqueID"])}}},
                            {"term": {"test_case_hash": {
                                "value": log["_source"]["test_case_hash"],
                                "boost": abs(self.search_cfg["BoostUniqueID"])}}},
                            {"term": {"is_auto_analyzed": {
                                "value": str(self.search_cfg["BoostAA"] > 0).lower(),
                                "boost": abs(self.search_cfg["BoostAA"]), }}},
                        ]}}}

    def build_analyze_query(self, launch, log, size=10):
        """Build analyze query"""
        min_should_match = "{}%".format(launch.analyzerConfig.minShouldMatch)\
            if launch.analyzerConfig.minShouldMatch > 0\
            else self.search_cfg["MinShouldMatch"]

        query = self.build_common_query(log, size=size)

        if launch.analyzerConfig.analyzerMode in ["LAUNCH_NAME"]:
            query["query"]["bool"]["must"].append(
                {"term": {
                    "launch_name": {
                        "value": launch.launchName}}})
        elif launch.analyzerConfig.analyzerMode in ["CURRENT_LAUNCH"]:
            query["query"]["bool"]["must"].append(
                {"term": {
                    "launch_id": {
                        "value": launch.launchId}}})
        else:
            query["query"]["bool"]["should"].append(
                {"term": {
                    "launch_name": {
                        "value": launch.launchName,
                        "boost": abs(self.search_cfg["BoostLaunch"])}}})

        if log["_source"]["message"].strip():
            log_lines = launch.analyzerConfig.numberOfLogLines
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            if log_lines == -1:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query(min_should_match,
                                                    log["_source"]["detected_message"],
                                                    field_name="detected_message",
                                                    boost=4.0))
                if log["_source"]["stacktrace"].strip():
                    query["query"]["bool"]["must"].append(
                        self.build_more_like_this_query(min_should_match,
                                                        log["_source"]["stacktrace"],
                                                        field_name="stacktrace",
                                                        boost=2.0))
                else:
                    query["query"]["bool"]["must_not"].append({"wildcard": {"stacktrace": "*"}})
            else:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query(min_should_match,
                                                    log["_source"]["message"],
                                                    field_name="message",
                                                    boost=4.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("80%",
                                                    log["_source"]["detected_message"],
                                                    field_name="detected_message",
                                                    boost=2.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("60%",
                                                    log["_source"]["stacktrace"],
                                                    field_name="stacktrace", boost=1.0))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("80%",
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=0.5))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["only_numbers"],
                                                field_name="only_numbers",
                                                boost=4.0,
                                                override_min_should_match="1"))
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(min_should_match,
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=2.0))
        if log["_source"]["found_exceptions"].strip():
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["found_exceptions"],
                                                field_name="found_exceptions",
                                                boost=4.0,
                                                override_min_should_match="1"))
        if log["_source"]["potential_status_codes"].strip():
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["potential_status_codes"],
                                                field_name="potential_status_codes",
                                                boost=4.0,
                                                override_min_should_match="1"))

        return query

    def build_suggest_query(self, test_item_info, log, size=10,
                            message_field="message", det_mes_field="detected_message",
                            stacktrace_field="stacktrace"):
        min_should_match = "{}%".format(test_item_info.analyzerConfig.minShouldMatch)\
            if test_item_info.analyzerConfig.minShouldMatch > 0\
            else self.search_cfg["MinShouldMatch"]
        log_lines = test_item_info.analyzerConfig.numberOfLogLines

        query = self.build_common_query(log, size=size)

        if test_item_info.analyzerConfig.analyzerMode in ["LAUNCH_NAME"]:
            query["query"]["bool"]["must"].append(
                {"term": {
                    "launch_name": {
                        "value": test_item_info.launchName}}})
        elif test_item_info.analyzerConfig.analyzerMode in ["CURRENT_LAUNCH"]:
            query["query"]["bool"]["must"].append(
                {"term": {
                    "launch_id": {
                        "value": test_item_info.launchId}}})
        else:
            query["query"]["bool"]["should"].append(
                {"term": {
                    "launch_name": {
                        "value": test_item_info.launchName,
                        "boost": abs(self.search_cfg["BoostLaunch"])}}})

        if log["_source"]["message"].strip():
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            if log_lines == -1:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query("60%",
                                                    log["_source"][det_mes_field],
                                                    field_name=det_mes_field,
                                                    boost=4.0))
                if log["_source"][stacktrace_field].strip():
                    query["query"]["bool"]["must"].append(
                        self.build_more_like_this_query("60%",
                                                        log["_source"][stacktrace_field],
                                                        field_name=stacktrace_field,
                                                        boost=2.0))
                else:
                    query["query"]["bool"]["must_not"].append({"wildcard": {stacktrace_field: "*"}})
            else:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query("60%",
                                                    log["_source"][message_field],
                                                    field_name=message_field,
                                                    boost=4.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("60%",
                                                    log["_source"][stacktrace_field],
                                                    field_name=stacktrace_field,
                                                    boost=1.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query(
                        "60%",
                        log["_source"]["detected_message_without_params_extended"],
                        field_name="detected_message_without_params_extended",
                        boost=1.0))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("80%",
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=0.5))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["only_numbers"],
                                                field_name="only_numbers",
                                                boost=4.0,
                                                override_min_should_match="1"))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["message_params"],
                                                field_name="message_params",
                                                boost=4.0,
                                                override_min_should_match="1"))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["urls"],
                                                field_name="urls",
                                                boost=4.0,
                                                override_min_should_match="1"))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["paths"],
                                                field_name="paths",
                                                boost=4.0,
                                                override_min_should_match="1"))
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(min_should_match,
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=2.0))

        query["query"]["bool"]["should"].append(
            self.build_more_like_this_query("1",
                                            log["_source"]["found_exceptions_extended"],
                                            field_name="found_exceptions_extended",
                                            boost=4.0,
                                            override_min_should_match="1"))
        query["query"]["bool"]["should"].append(
            self.build_more_like_this_query("1",
                                            log["_source"]["potential_status_codes"],
                                            field_name="potential_status_codes",
                                            boost=4.0,
                                            override_min_should_match="1"))

        return query
