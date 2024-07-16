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

import os
import pickle
from datetime import datetime
from time import time
from typing import Optional, Any

import elasticsearch
import elasticsearch.helpers
import numpy as np
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from app.commons import logging, namespace_finder, object_saving
from app.commons.esclient import EsClient
from app.commons.model.ml import TrainInfo, ModelType
from app.commons.model.launch_objects import SearchConfig, ApplicationConfig
from app.commons.model_chooser import ModelChooser
from app.machine_learning.feature_encoding_configurer import FeatureEncodingConfigurer
from app.machine_learning.models import (boosting_decision_maker, custom_boosting_decision_maker,
                                         weighted_similarity_calculator)
from app.machine_learning.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.utils import utils, text_processing

logger = logging.getLogger("analyzerApp.trainingAnalysisModel")


def calculate_f1(model, x_test, y_test, _):
    return model.validate_model(x_test, y_test)


def calculate_mrr(model, x_test, y_test, test_item_ids_with_pos):
    res_labels, prob_labels = model.predict(x_test)
    test_item_ids_res = {}
    for i in range(len(test_item_ids_with_pos)):
        test_item = test_item_ids_with_pos[i]
        if test_item not in test_item_ids_res:
            test_item_ids_res[test_item] = []
        test_item_ids_res[test_item].append((res_labels[i], prob_labels[i][1], y_test[i]))
    mrr = 0
    cnt_to_use = 0
    for test_item in test_item_ids_res:
        res = sorted(test_item_ids_res[test_item], key=lambda x: x[1], reverse=True)
        has_positives = False
        for r in res:
            if r[2] == 1:
                has_positives = True
                break
        if not has_positives:
            continue
        rr_test_item = 0
        for idx, r in enumerate(res):
            if r[2] == 1 and r[0] == 1:
                rr_test_item = 1 / (idx + 1)
                break
        mrr += rr_test_item
        cnt_to_use += 1
    if cnt_to_use:
        mrr /= cnt_to_use
    return mrr


def deduplicate_data(data, labels) -> tuple[list, list]:
    data_wo_duplicates = []
    labels_wo_duplicates = []
    data_set = set()
    for i in range(len(data)):
        if tuple(data[i]) not in data_set:
            data_set.add(tuple(data[i]))
            data_wo_duplicates.append(data[i])
            labels_wo_duplicates.append(labels[i])
    return data_wo_duplicates, labels_wo_duplicates


def split_data(data, labels, random_state, test_item_ids_with_pos) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    x_ids = [i for i in range(len(data))]
    x_train_ids, x_test_ids, y_train, y_test = train_test_split(
        x_ids, labels,
        test_size=0.1, random_state=random_state, stratify=labels)
    x_train = np.asarray([data[idx] for idx in x_train_ids])
    x_test = np.asarray([data[idx] for idx in x_test_ids])
    test_item_ids_with_pos_test = [test_item_ids_with_pos[idx] for idx in x_test_ids]
    return x_train, x_test, y_train, y_test, test_item_ids_with_pos_test


def transform_data_from_feature_lists(feature_list, cur_features, desired_features):
    previously_gathered_features = utils.fill_prevously_gathered_features(feature_list, cur_features)
    gathered_data = utils.gather_feature_list(previously_gathered_features, desired_features)
    return gathered_data


def prepare_encoders(features_encoding_config, logs_found):
    _feature_encoding_configurer = FeatureEncodingConfigurer()
    _feature_encoding_configurer.initialize_encoders_from_config(features_encoding_config)
    _feature_encoding_configurer.prepare_encoders(logs_found)
    return _feature_encoding_configurer.feature_dict_with_encodings


class AnalysisModelTraining:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    model_chooser: Optional[ModelChooser]

    def __init__(self, app_config: ApplicationConfig, search_cfg: SearchConfig,
                 model_chooser: Optional[ModelChooser] = None) -> None:
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.due_proportion = 0.05
        self.due_proportion_to_smote = 0.4
        self.es_client = EsClient(app_config=app_config)
        self.baseline_folders = {
            ModelType.suggestion: self.search_cfg.SuggestBoostModelFolder,
            ModelType.auto_analysis: self.search_cfg.BoostModelFolder}
        self.model_config = {
            ModelType.suggestion: self.search_cfg.RetrainSuggestBoostModelConfig,
            ModelType.auto_analysis: self.search_cfg.RetrainAutoBoostModelConfig}
        self.weighted_log_similarity_calculator = None
        if self.search_cfg.SimilarityWeightsFolder.strip():
            self.weighted_log_similarity_calculator = (
                weighted_similarity_calculator.WeightedSimilarityCalculator(
                    object_saving.create_filesystem(self.search_cfg.SimilarityWeightsFolder)))
            self.weighted_log_similarity_calculator.load_model()
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_chooser = model_chooser
        self.metrics_calculations = {
            "F1": calculate_f1,
            "Mean Reciprocal Rank": calculate_mrr
        }

    def get_config_for_boosting(self, number_of_log_lines: int, model_type: ModelType, namespaces) -> dict[str, Any]:
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg.MinWordLength,
            "filter_min_should_match": [],
            "filter_min_should_match_any": [],
            "number_of_log_lines": number_of_log_lines,
            "filter_by_test_case_hash": False,
            "boosting_model": self.baseline_folders[model_type],
            "chosen_namespaces": namespaces,
            "calculate_similarities": False,
            "time_weight_decay": self.search_cfg.TimeWeightDecay}

    @staticmethod
    def get_info_template(project_info: TrainInfo, baseline_model: str, model_name: str,
                          metric_name: str) -> dict[str, Any]:
        return {"method": "training", "sub_model_type": "all", "model_type": project_info.model_type.name,
                "baseline_model": [baseline_model], "new_model": [model_name],
                "project_id": str(project_info.project), "model_saved": 0, "p_value": 1.0,
                "data_proportion": 0.0, "baseline_mean_metric": 0.0, "new_model_mean_metric": 0.0,
                "bad_data_proportion": 0, "metric_name": metric_name, "errors": [], "errors_count": 0}

    def calculate_metrics(self, model, x_test, y_test,
                          metrics_to_gather, test_item_ids_with_pos, new_model_results):
        for metric in metrics_to_gather:
            metric_res = 0.0
            if metric in self.metrics_calculations:
                metric_res = self.metrics_calculations[metric](
                    model, x_test, y_test, test_item_ids_with_pos)
            if metric not in new_model_results:
                new_model_results[metric] = []
            new_model_results[metric].append(metric_res)
        return new_model_results

    def train_several_times(self, data, labels, features, test_item_ids_with_pos, metrics_to_gather):
        new_model_results = {}
        baseline_model_results = {}
        random_states = [1257, 1873, 1917, 2477, 3449,
                         353, 4561, 5417, 6427, 2029]
        bad_data = False

        proportion_binary_labels = utils.calculate_proportions_for_labels(labels)

        if proportion_binary_labels < self.due_proportion:
            logger.debug("Train data has a bad proportion: %.3f", proportion_binary_labels)
            bad_data = True

        if not bad_data:
            data, labels = deduplicate_data(data, labels)
            for random_state in random_states:
                x_train, x_test, y_train, y_test, test_item_ids_with_pos_test = split_data(
                    data, labels, random_state, test_item_ids_with_pos)
                proportion_binary_labels = utils.calculate_proportions_for_labels(y_train)
                if proportion_binary_labels < self.due_proportion_to_smote:
                    oversample = SMOTE(ratio="minority")
                    x_train, y_train = oversample.fit_resample(x_train, y_train)
                self.new_model.train_model(x_train, y_train)
                logger.debug("New model results")
                new_model_results = self.calculate_metrics(
                    self.new_model, x_test, y_test, metrics_to_gather,
                    test_item_ids_with_pos_test, new_model_results)
                logger.debug("Baseline results")
                x_test_for_baseline = transform_data_from_feature_lists(
                    x_test, features, self.baseline_model.get_feature_ids())
                baseline_model_results = self.calculate_metrics(
                    self.baseline_model, x_test_for_baseline, y_test,
                    metrics_to_gather, test_item_ids_with_pos_test, baseline_model_results)
        return baseline_model_results, new_model_results, bad_data

    def query_logs(self, project_id: int, log_ids_to_find: list[str]) -> dict[str, Any]:
        log_ids_to_find = list(log_ids_to_find)
        project_index_name = text_processing.unite_project_name(
            str(project_id), self.app_config.esProjectIndexPrefix)
        batch_size = 1000
        log_id_dict = {}
        for i in range(int(len(log_ids_to_find) / batch_size) + 1):
            log_ids = log_ids_to_find[i * batch_size: (i + 1) * batch_size]
            if not log_ids:
                continue
            ids_query = {
                "size": self.app_config.esChunkNumber,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"_id": log_ids}}
                        ]
                    }
                }}
            for r in elasticsearch.helpers.scan(
                    self.es_client.es_client, query=ids_query, index=project_index_name, scroll="5m"):
                log_id_dict[str(r["_id"])] = r
        return log_id_dict

    def get_search_query_suggest(self):
        return {
            "sort": {"savedDate": "desc"},
            "size": self.app_config.esChunkNumber,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"methodName": "suggestion"}}
                    ]
                }
            }
        }

    def get_search_query_aa(self, user_choice):
        return {
            "sort": {"savedDate": "desc"},
            "size": self.app_config.esChunkNumber,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"methodName": "auto_analysis"}},
                        {"term": {"userChoice": user_choice}}
                    ]
                }
            }
        }

    @staticmethod
    def stop_gathering_info_from_suggest_query(num_of_1s, num_of_0s, max_num):
        if (num_of_1s + num_of_0s) == 0:
            return False
        percent_logs = (num_of_1s + num_of_0s) / max_num
        percent_1s = num_of_1s / (num_of_1s + num_of_0s)
        if percent_logs >= 0.8 and percent_1s <= 0.2:
            return True
        return False

    def query_es_for_suggest_info(self, project_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        log_ids_to_find = set()
        gathered_suggested_data = []
        log_id_pairs_set = set()
        index_name = text_processing.unite_project_name(
            str(project_id) + "_suggest", self.app_config.esProjectIndexPrefix)
        max_number_of_logs = 30000
        cur_number_of_logs = 0
        cur_number_of_logs_0 = 0
        cur_number_of_logs_1 = 0
        unique_saved_features = set()
        for query_name, query in [
            ("auto_analysis 0s", self.get_search_query_aa(0)),
            ("suggestion", self.get_search_query_suggest()),
            ("auto_analysis 1s", self.get_search_query_aa(1))
        ]:
            if cur_number_of_logs >= max_number_of_logs:
                break
            for res in elasticsearch.helpers.scan(self.es_client.es_client,
                                                  query=query,
                                                  index=index_name,
                                                  scroll="5m"):
                if cur_number_of_logs >= max_number_of_logs:
                    break
                saved_model_features = "{}|{}".format(
                    res["_source"]["modelFeatureNames"],
                    res["_source"]["modelFeatureValues"])
                if saved_model_features in unique_saved_features:
                    continue
                unique_saved_features.add(saved_model_features)
                log_ids_pair = (res["_source"]["testItemLogId"], res["_source"]["relevantLogId"])
                if log_ids_pair in log_id_pairs_set:
                    continue
                log_id_pairs_set.add(log_ids_pair)
                for col in ["testItemLogId", "relevantLogId"]:
                    log_id = str(res["_source"][col])
                    if res["_source"]["isMergedLog"]:
                        log_id = log_id + "_m"
                    log_ids_to_find.add(log_id)
                gathered_suggested_data.append(res)
                cur_number_of_logs += 1
                if res["_source"]["userChoice"] == 1:
                    cur_number_of_logs_1 += 1
                else:
                    cur_number_of_logs_0 += 1
                if query_name == "suggestion" and self.stop_gathering_info_from_suggest_query(
                        cur_number_of_logs_1, cur_number_of_logs_0, max_number_of_logs):
                    break
            logger.debug("Query: '%s', results number: %d, number of 1s: %d",
                         query_name, cur_number_of_logs, cur_number_of_logs_1)
        log_id_dict = self.query_logs(project_id, list(log_ids_to_find))
        return gathered_suggested_data, log_id_dict

    def gather_data(self, model_type: ModelType, project_id: int, features: list[int], defect_type_model_to_use,
                    full_config):
        namespaces = self.namespace_finder.get_chosen_namespaces(project_id)
        gathered_suggested_data, log_id_dict = self.query_es_for_suggest_info(project_id)
        features_dict_with_saved_objects = prepare_encoders(
            full_config["features_encoding_config"], list(log_id_dict.values()))
        full_data_features, labels, test_item_ids_with_pos = [], [], []
        for _suggest_res in gathered_suggested_data:
            searched_res = []
            found_logs = {}
            for col in ["testItemLogId", "relevantLogId"]:
                log_id = str(_suggest_res["_source"][col])
                if _suggest_res["_source"]["isMergedLog"]:
                    log_id = log_id + "_m"
                if log_id in log_id_dict:
                    found_logs[col] = log_id_dict[log_id]
            if len(found_logs) == 2:
                log_relevant = found_logs["relevantLogId"]
                log_relevant["_score"] = _suggest_res["_source"]["esScore"]
                searched_res = [
                    (found_logs["testItemLogId"], {"hits": {"hits": [log_relevant]}})]
            if searched_res:
                _boosting_data_gatherer = SuggestBoostingFeaturizer(
                    searched_res,
                    self.get_config_for_boosting(
                        _suggest_res["_source"]["usedLogLines"], model_type, namespaces),
                    feature_ids=features,
                    weighted_log_similarity_calculator=self.weighted_log_similarity_calculator,
                    features_dict_with_saved_objects=features_dict_with_saved_objects)
                _boosting_data_gatherer.set_defect_type_model(defect_type_model_to_use)
                _boosting_data_gatherer.fill_previously_gathered_features(
                    [utils.to_number_list(_suggest_res["_source"]["modelFeatureValues"])],
                    _suggest_res["_source"]["modelFeatureNames"])
                feature_data, _ = _boosting_data_gatherer.gather_features_info()
                if feature_data:
                    full_data_features.extend(feature_data)
                    labels.append(_suggest_res["_source"]["userChoice"])
                    test_item_ids_with_pos.append(_suggest_res["_source"]["testItem"])
        return (np.asarray(full_data_features), np.asarray(labels), test_item_ids_with_pos,
                features_dict_with_saved_objects)

    def train(self, project_info: TrainInfo) -> tuple[int, dict[str, Any]]:
        time_training = time()
        logger.debug("Started training model '%s'", project_info.model_type.name)
        model_name = "%s_model_%s" % (project_info.model_type.name, datetime.now().strftime("%d.%m.%y"))

        baseline_model_folder = os.path.basename(
            self.baseline_folders[project_info.model_type].strip("/").strip("\\"))
        self.baseline_model = boosting_decision_maker.BoostingDecisionMaker(
            object_saving.create_filesystem(self.baseline_folders[project_info.model_type]))
        self.baseline_model.load_model()

        full_config, features, monotonous_features = pickle.load(
            open(self.model_config[project_info.model_type], "rb"))
        new_model_folder = "%s_model/%s/" % (project_info.model_type.name, model_name)
        self.new_model = custom_boosting_decision_maker.CustomBoostingDecisionMaker(
            object_saving.create(self.app_config, project_id=project_info.project, path=new_model_folder))
        self.new_model.add_config_info(full_config, features, monotonous_features)

        defect_type_model_to_use = self.model_chooser.choose_model(project_info.project, ModelType.defect_type)

        metrics_to_gather = ["F1", "Mean Reciprocal Rank"]
        train_log_info = {}
        for metric in metrics_to_gather:
            train_log_info[metric] = self.get_info_template(
                project_info, baseline_model_folder, model_name, metric)

        errors = []
        errors_count = 0
        train_data = []
        try:
            logger.debug("Initialized training model '%s'", project_info.model_type.name)
            train_data, labels, test_item_ids_with_pos, features_dict_with_saved_objects = self.gather_data(
                project_info.model_type, project_info.project, self.new_model.get_feature_ids(),
                defect_type_model_to_use, full_config)
            self.new_model.features_dict_with_saved_objects = features_dict_with_saved_objects

            for metric in metrics_to_gather:
                train_log_info[metric]["data_size"] = len(labels)
                train_log_info[metric]["data_proportion"] = utils.calculate_proportions_for_labels(labels)

            logger.debug("Loaded data for training model '%s'", project_info.model_type.name)
            baseline_model_results, new_model_results, bad_data = self.train_several_times(
                train_data, labels, self.new_model.get_feature_ids(),
                test_item_ids_with_pos, metrics_to_gather)
            for metric in metrics_to_gather:
                train_log_info[metric]["bad_data_proportion"] = int(bad_data)

            use_custom_model = False
            if not bad_data:
                for metric in metrics_to_gather:
                    logger.debug("Baseline test results %s", baseline_model_results[metric])
                    logger.debug("New model test results %s", new_model_results[metric])
                    f_value, p_value = stats.f_oneway(baseline_model_results[metric], new_model_results[metric])
                    if p_value is None:
                        p_value = 1.0
                    train_log_info[metric]["p_value"] = p_value
                    mean_f1 = np.mean(new_model_results[metric])
                    train_log_info[metric]["baseline_mean_metric"] = np.mean(baseline_model_results[metric])
                    train_log_info[metric]["new_model_mean_metric"] = mean_f1
                    if p_value < 0.05 and mean_f1 > np.mean(baseline_model_results[metric]) and mean_f1 >= 0.4:
                        use_custom_model = True
                    logger.debug(
                        """Model training validation results:
                            p-value=%.3f mean baseline=%.3f mean new model=%.3f""",
                        p_value, np.mean(baseline_model_results[metric]), np.mean(new_model_results[metric]))

            if use_custom_model:
                logger.debug("Custom model should be saved")

                proportion_binary_labels = utils.calculate_proportions_for_labels(labels)
                if proportion_binary_labels < self.due_proportion_to_smote:
                    oversample = SMOTE(ratio="minority")
                    train_data, labels = oversample.fit_resample(train_data, labels)
                    proportion_binary_labels = utils.calculate_proportions_for_labels(labels)
                if proportion_binary_labels < self.due_proportion:
                    logger.debug("Train data has a bad proportion: %.3f", proportion_binary_labels)
                    bad_data = True
                for metric in metrics_to_gather:
                    train_log_info[metric]["bad_data_proportion"] = int(bad_data)
                if not bad_data:
                    for metric in metrics_to_gather:
                        train_log_info[metric]["model_saved"] = 1
                    self.new_model.train_model(train_data, labels)
                else:
                    for metric in metrics_to_gather:
                        train_log_info[metric]["model_saved"] = 0
                self.model_chooser.delete_old_model(project_info.model_type, project_info.project)
                self.new_model.save_model()
        except Exception as err:
            logger.error(err)
            errors.append(utils.extract_exception(err))
            errors_count += 1

        time_spent = (time() - time_training)
        for metric in metrics_to_gather:
            train_log_info[metric]["time_spent"] = time_spent
            train_log_info[metric]["gather_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            train_log_info[metric]["module_version"] = [self.app_config.appVersion]
            train_log_info[metric]["errors"].extend(errors)
            train_log_info[metric]["errors_count"] += errors_count

        logger.info("Finished for %d s", time_spent)
        return len(train_data), train_log_info
