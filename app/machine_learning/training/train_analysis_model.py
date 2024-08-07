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
from datetime import datetime
from time import time
from typing import Optional, Any, Callable, Type

import elasticsearch
import elasticsearch.helpers
import numpy as np
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from app.commons import logging, namespace_finder, object_saving
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import SearchConfig, ApplicationConfig
from app.commons.model.ml import TrainInfo, ModelType
from app.commons.model_chooser import ModelChooser
from app.machine_learning.models import (BoostingDecisionMaker, CustomBoostingDecisionMaker,
                                         WeightedSimilarityCalculator)
from app.machine_learning.boosting_featurizer import BoostingFeaturizer
from app.machine_learning.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.utils import utils, text_processing
from app.utils.defaultdict import DefaultDict

LOGGER = logging.getLogger("analyzerApp.trainingAnalysisModel")
TRAIN_DATA_RANDOM_STATES = [1257, 1873, 1917, 2477, 3449, 353, 4561, 5417, 6427, 2029, 2137]
DUE_PROPORTION = 0.05
SMOTE_PROPORTION = 0.4
MIN_P_VALUE = 0.05


def calculate_f1(model: BoostingDecisionMaker, x_test: list[list[float]], y_test: list[int], _) -> float:
    return model.validate_model(x_test, y_test)


def calculate_mrr(model: BoostingDecisionMaker, x_test: list[list[float]], y_test: list[int],
                  test_item_ids_with_pos: list[int]) -> float:
    res_labels, prob_labels = model.predict(x_test)
    test_item_ids_res = {}
    for i in range(len(test_item_ids_with_pos)):
        test_item = test_item_ids_with_pos[i]
        if test_item not in test_item_ids_res:
            test_item_ids_res[test_item] = []
        test_item_ids_res[test_item].append((res_labels[i], prob_labels[i][1], y_test[i]))
    mrr = 0.0
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
        rr_test_item = 0.0
        for idx, r in enumerate(res):
            if r[2] == 1 and r[0] == 1:
                rr_test_item = 1 / (idx + 1)
                break
        mrr += rr_test_item
        cnt_to_use += 1
    if cnt_to_use:
        mrr /= cnt_to_use
    return mrr


METRIC_CALCULATIONS: dict[str, Callable[[BoostingDecisionMaker, list[list[float]], list[int], list[int]], float]] = {
    "F1": calculate_f1,
    "Mean Reciprocal Rank": calculate_mrr
}


def calculate_metrics(model: Optional[BoostingDecisionMaker], x_test: list[list[float]], y_test: list[int],
                      test_item_ids_with_pos: list[int], result_dict: DefaultDict[str, list[float]]) -> None:
    for metric, metric_calc_func in METRIC_CALCULATIONS:
        if model:
            metric_res = metric_calc_func(model, x_test, y_test, test_item_ids_with_pos)
            result_dict[metric].append(metric_res)
        else:
            result_dict[metric].append(0.0)


def deduplicate_data(data: list[list[float]], labels: list[int]) -> tuple[list[list[float]], list[int]]:
    data_wo_duplicates = []
    labels_wo_duplicates = []
    data_set = set()
    for i in range(len(data)):
        data_tuple = tuple(data[i])
        if data_tuple not in data_set:
            data_set.add(data_tuple)
            data_wo_duplicates.append(data[i])
            labels_wo_duplicates.append(labels[i])
    return data_wo_duplicates, labels_wo_duplicates


def split_data(
        data: list[list[float]], labels: list[int], random_state: int, test_item_ids_with_pos: list[int]
) -> tuple[list[list[float]], list[list[float]], list[int], list[int], list[int]]:
    x_ids: list[int] = [i for i in range(len(data))]
    x_train_ids, x_test_ids, y_train, y_test = train_test_split(
        x_ids, labels, test_size=0.1, random_state=random_state, stratify=labels)
    x_train = [data[idx] for idx in x_train_ids]
    x_test = [data[idx] for idx in x_test_ids]
    test_item_ids_with_pos_test = [test_item_ids_with_pos[idx] for idx in x_test_ids]
    return x_train, x_test, y_train, y_test, test_item_ids_with_pos_test


def transform_data_from_feature_lists(feature_list, cur_features: list[int],
                                      desired_features: list[int]) -> list[list[float]]:
    previously_gathered_features = utils.fill_previously_gathered_features(feature_list, cur_features)
    gathered_data = utils.gather_feature_list(previously_gathered_features, desired_features)
    return gathered_data


def fill_metric_stats(baseline_model_metric_result: list[float], new_model_metric_results: list[float],
                      info_dict: dict[str, Any]) -> None:
    _, p_value = stats.f_oneway(baseline_model_metric_result, new_model_metric_results)
    p_value = p_value if p_value is not None else 1.0
    info_dict['p_value'] = p_value
    mean_metric = np.mean(new_model_metric_results)
    baseline_mean_metric = np.mean(baseline_model_metric_result)
    info_dict['baseline_mean_metric'] = baseline_mean_metric
    info_dict['new_model_mean_metric'] = mean_metric
    info_dict['gather_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def train_several_times(
        new_model: BoostingDecisionMaker, data: list[list[float]], labels: list[int], features: list[int],
        test_item_ids_with_pos: list[int], random_states: Optional[list[int]] = None,
        baseline_model: Optional[BoostingDecisionMaker] = None
) -> tuple[dict[str, list[float]], dict[str, list[float]], bool, float]:
    new_model_results = DefaultDict(lambda _, __: [])
    baseline_model_results = DefaultDict(lambda _, __: [])
    my_random_states = random_states if random_states else TRAIN_DATA_RANDOM_STATES

    bad_data = False
    proportion_binary_labels = utils.calculate_proportions_for_labels(labels)
    if proportion_binary_labels < DUE_PROPORTION:
        LOGGER.debug("Train data has a bad proportion: %.3f", proportion_binary_labels)
        bad_data = True

    if not bad_data:
        data, labels = deduplicate_data(data, labels)
        for random_state in my_random_states:
            x_train, x_test, y_train, y_test, test_item_ids_with_pos_test = split_data(
                data, labels, random_state, test_item_ids_with_pos)
            proportion_binary_labels = utils.calculate_proportions_for_labels(y_train)
            if proportion_binary_labels < SMOTE_PROPORTION:
                oversample = SMOTE(ratio="minority")
                x_train, y_train = oversample.fit_resample(x_train, y_train)
            new_model.train_model(x_train, y_train)
            LOGGER.debug("New model results")
            calculate_metrics(
                new_model, x_test, y_test, test_item_ids_with_pos_test, new_model_results)
            LOGGER.debug("Baseline results")
            x_test_for_baseline = transform_data_from_feature_lists(x_test, features, baseline_model.feature_ids)
            calculate_metrics(
                baseline_model, x_test_for_baseline, y_test, test_item_ids_with_pos_test, baseline_model_results)

    return baseline_model_results, new_model_results, bad_data, proportion_binary_labels


class AnalysisModelTraining:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    model_type: ModelType
    model_class: Type[BoostingDecisionMaker]
    baseline_folder: Optional[str]
    baseline_model: Optional[BoostingDecisionMaker]
    model_chooser: ModelChooser
    features: Optional[list[int]]
    mono_features: Optional[list[int]]

    def __init__(self, app_config: ApplicationConfig, search_cfg: SearchConfig, model_type: ModelType,
                 model_chooser: ModelChooser, model_class: Optional[Type[BoostingDecisionMaker]] = None) -> None:
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.due_proportion = 0.05
        self.due_proportion_to_smote = 0.4
        self.es_client = EsClient(app_config=app_config)
        self.model_type = model_type
        if model_type is ModelType.suggestion:
            self.baseline_folder = self.search_cfg.SuggestBoostModelFolder
            model_config = self.search_cfg.RetrainSuggestBoostModelConfig
        elif model_type is ModelType.auto_analysis:
            self.baseline_folder = self.search_cfg.BoostModelFolder
            model_config = self.search_cfg.RetrainAutoBoostModelConfig
        else:
            raise ValueError(f'Incorrect model type {model_type}')

        self.model_class = model_class if model_class else CustomBoostingDecisionMaker

        if self.baseline_folder:
            self.baseline_model = BoostingDecisionMaker(
                object_saving.create_filesystem(self.baseline_folder))
            self.baseline_model.load_model()
            self.features, self.monotonous_features = object_saving.create_filesystem(
                os.path.dirname(model_config)).get_project_object(os.path.basename(model_config), using_json=False)

        self.weighted_log_similarity_calculator = None
        if self.search_cfg.SimilarityWeightsFolder.strip():
            self.weighted_log_similarity_calculator = WeightedSimilarityCalculator(
                    object_saving.create_filesystem(self.search_cfg.SimilarityWeightsFolder))
            self.weighted_log_similarity_calculator.load_model()
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_chooser = model_chooser

    def get_config_for_boosting(self, number_of_log_lines: int, namespaces) -> dict[str, Any]:
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg.MinWordLength,
            "filter_min_should_match": [],
            "filter_min_should_match_any": [],
            "number_of_log_lines": number_of_log_lines,
            "filter_by_test_case_hash": False,
            "boosting_model": self.baseline_folder,
            "chosen_namespaces": namespaces,
            "calculate_similarities": False,
            "time_weight_decay": self.search_cfg.TimeWeightDecay}

    @staticmethod
    def get_info_template(project_info: TrainInfo, baseline_model: str, model_name: str,
                          metric_name: str) -> dict[str, Any]:
        return {'method': 'training', 'sub_model_type': 'all', 'model_type': project_info.model_type.name,
                'baseline_model': [baseline_model], 'new_model': [model_name],
                'project_id': str(project_info.project), 'model_saved': 0, 'p_value': 1.0, 'data_size': 0,
                'data_proportion': 0.0, 'baseline_mean_metric': 0.0, 'new_model_mean_metric': 0.0,
                'bad_data_proportion': 0, 'metric_name': metric_name, 'errors': [], 'errors_count': 0}

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

    def get_search_query_aa(self, user_choice: int) -> dict[str, Any]:
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
            for res in elasticsearch.helpers.scan(self.es_client.es_client, query=query, index=index_name,
                                                  scroll="5m"):
                if cur_number_of_logs >= max_number_of_logs:
                    break
                saved_model_features = f'{res["_source"]["modelFeatureNames"]}|{res["_source"]["modelFeatureValues"]}'
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
            LOGGER.debug("Query: '%s', results number: %d, number of 1s: %d",
                         query_name, cur_number_of_logs, cur_number_of_logs_1)
        log_id_dict = self.query_logs(project_id, list(log_ids_to_find))
        return gathered_suggested_data, log_id_dict

    def gather_data(self, projects: list[int], features: list[int]) -> tuple[list[list[float]], list[int], list[int]]:
        full_data_features, labels, test_item_ids_with_pos = [], [], []
        for project_id in projects:
            namespaces = self.namespace_finder.get_chosen_namespaces(project_id)
            gathered_suggested_data, log_id_dict = self.query_es_for_suggest_info(project_id)

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
                    searched_res = [(found_logs["testItemLogId"], {"hits": {"hits": [log_relevant]}})]
                if searched_res:
                    if self.model_type is ModelType.suggestion:
                        _boosting_data_gatherer = SuggestBoostingFeaturizer(
                            searched_res,
                            self.get_config_for_boosting(_suggest_res["_source"]["usedLogLines"], namespaces),
                            feature_ids=features,
                            weighted_log_similarity_calculator=self.weighted_log_similarity_calculator)
                    else:
                        _boosting_data_gatherer = BoostingFeaturizer(
                            searched_res,
                            self.get_config_for_boosting(_suggest_res["_source"]["usedLogLines"], namespaces),
                            feature_ids=features,
                            weighted_log_similarity_calculator=self.weighted_log_similarity_calculator)

                    # noinspection PyTypeChecker
                    _boosting_data_gatherer.set_defect_type_model(
                        self.model_chooser.choose_model(project_id, ModelType.defect_type))
                    _boosting_data_gatherer.fill_previously_gathered_features(
                        [utils.to_float_list(_suggest_res['_source']['modelFeatureValues'])],
                        [int(_id) for _id in _suggest_res['_source']['modelFeatureNames'].split(';')])
                    feature_data, _ = _boosting_data_gatherer.gather_features_info()
                    if feature_data:
                        full_data_features.extend(feature_data)
                        labels.append(_suggest_res['_source']['userChoice'])
                        test_item_ids_with_pos.append(int(_suggest_res['_source']['testItem']))
        return full_data_features, labels, test_item_ids_with_pos

    def train_several_times(
            self, new_model: BoostingDecisionMaker, data: list[list[float]], labels: list[int], features: list[int],
            test_item_ids_with_pos: list[int], random_states: Optional[list[int]] = None
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], bool, float]:
        return train_several_times(
            new_model, data, labels, features, test_item_ids_with_pos, random_states, self.baseline_model)

    def train(self, project_info: TrainInfo) -> tuple[int, dict[str, Any]]:
        time_training = time()
        model_name = f'{project_info.model_type.name}_model_{datetime.now().strftime("%Y-%m-%d")}'
        baseline_model = os.path.basename(self.baseline_folder)
        new_model_folder = f'{project_info.model_type.name}_model/{model_name}/'

        LOGGER.info(f'Train "{self.model_type.name}" model using class: {self.model_class}')
        new_model = self.model_class(
            object_saving.create(self.app_config, project_id=project_info.project, path=new_model_folder),
            features=self.features, monotonous_features=self.features)
        new_model.add_config_info(self.features, self.monotonous_features)

        train_log_info = DefaultDict(lambda _, k: self.get_info_template(project_info, baseline_model, model_name, k))

        LOGGER.debug(f'Initialized model training {project_info.model_type.name}')
        projects = [project_info.project]
        if project_info.additional_projects:
            projects.extend(project_info.additional_projects)
        train_data, labels, test_item_ids_with_pos = self.gather_data(projects, new_model.feature_ids)
        LOGGER.debug(f'Loaded data for model training {self.model_type.name}')

        baseline_model_results, new_model_results, bad_data, data_proportion = self.train_several_times(
            new_model, train_data, labels, new_model.feature_ids, test_item_ids_with_pos)
        for metric in new_model_results:
            train_log_info[metric]['data_size'] = len(labels)
            train_log_info[metric]['bad_data_proportion'] = int(bad_data)
            train_log_info[metric]['data_proportion'] = data_proportion

        use_custom_model = False
        mean_metric_results: Optional[list[float]] = None
        if not bad_data:
            LOGGER.debug(f'Baseline test results {baseline_model_results}')
            LOGGER.debug(f'New model test results {new_model_results}')
            p_values = []
            new_metrics_better = True
            for metric, metric_results in new_model_results.items():
                info_dict = train_log_info[metric]
                fill_metric_stats(baseline_model_results[metric], metric_results, info_dict)
                p_value = info_dict['p_value']
                p_values.append(p_value)
                mean_metric = info_dict['new_model_mean_metric']
                baseline_mean_metric = info_dict['baseline_mean_metric']
                new_metrics_better = (
                        new_metrics_better
                        and mean_metric > baseline_mean_metric
                        and mean_metric >= 0.4
                )
                LOGGER.info(
                    f'Model training validation results: p-value={p_value:.3f}; mean {metric} metric '
                    f'baseline={baseline_mean_metric:.3f}; mean new model={mean_metric:.3f}.')
                if mean_metric_results:
                    for i in range(len(metric_results)):
                        mean_metric_results[i] = mean_metric_results[i] * metric_results[i]
                else:
                    mean_metric_results = metric_results.copy()

            if max(p_values) < MIN_P_VALUE and new_metrics_better:
                use_custom_model = True

        if use_custom_model:
            LOGGER.debug('Custom model should be saved')
            max_train_result_idx = int(np.argmax(mean_metric_results))
            best_random_state = TRAIN_DATA_RANDOM_STATES[max_train_result_idx]

            LOGGER.info(f'Perform final training with random state: {best_random_state}')
            self.train_several_times(
                new_model, train_data, labels, new_model.feature_ids, test_item_ids_with_pos, [best_random_state])
            if self.model_chooser:
                self.model_chooser.delete_old_model(project_info.model_type, project_info.project)
            new_model.save_model()
            for metric in METRIC_CALCULATIONS:
                train_log_info[metric]['model_saved'] = 1
        else:
            for metric in METRIC_CALCULATIONS:
                train_log_info[metric]['model_saved'] = 0

        time_spent = (time() - time_training)
        for metric in METRIC_CALCULATIONS:
            train_log_info[metric]['time_spent'] = time_spent
            train_log_info[metric]['module_version'] = [self.app_config.appVersion]

        LOGGER.info(f'Finished for {time_spent} s')
        return len(train_data), train_log_info
