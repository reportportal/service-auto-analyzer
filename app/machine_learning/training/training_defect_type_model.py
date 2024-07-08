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
from time import time, sleep
from typing import Any, Optional

import elasticsearch.helpers
import numpy as np
import scipy.stats as stats
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from app.commons import logging, object_saving
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import SearchConfig, ApplicationConfig
from app.commons.model.ml import TrainInfo
from app.commons.model_chooser import ModelChooser
from app.machine_learning.models import DefectTypeModel, CustomDefectTypeModel
from app.machine_learning.models.defect_type_model import DATA_FIELD
from app.utils import utils, text_processing
from app.utils.defaultdict import DefaultDict

LOGGER = logging.getLogger('analyzerApp.trainingDefectTypeModel')
TRAIN_DATA_RANDOM_STATES = [1257, 1873, 1917, 2477, 3449, 353, 4561, 5417, 6427, 2029, 2137]
RETRY_COUNT = 5
RETRY_PAUSES = [0, 1, 5, 10, 20, 40, 60]
BASE_ISSUE_CLASS_INDEXES: dict[str, int] = {'ab': 0, 'pb': 1, 'si': 2}
MINIMAL_LABEL_PROPORTION = 0.2


def return_similar_objects_into_sample(x_train_ind: list[int], y_train: list[int],
                                       data: list[tuple[str, str, str]], additional_logs: dict[int, list[int]],
                                       label: str):
    x_train = []
    x_train_add = []
    y_train_add = []

    for idx, ind in enumerate(x_train_ind):
        x_train.append(data[ind][0])
        label_to_use = y_train[idx]
        if ind in additional_logs and label_to_use != 1:
            for idx_ in additional_logs[ind]:
                log_res, label_res, real_label = data[idx_]
                if label_res == label:
                    label_to_use = 1
                    break
        if ind in additional_logs:
            for idx_ in additional_logs[ind]:
                x_train_add.append(data[idx_][0])
                y_train_add.append(label_to_use)
    x_train.extend(x_train_add)
    y_train.extend(y_train_add)
    return x_train, y_train


def split_train_test(
        logs_to_train_idx: list[int], data: list[tuple[str, str, str]], labels_filtered, additional_logs,
        label: str, random_state: int = 1257) -> tuple[list, list, list, list]:
    x_train_ind, x_test_ind, y_train, y_test = train_test_split(
        logs_to_train_idx, labels_filtered, test_size=0.1, random_state=random_state,
        stratify=labels_filtered)
    x_train, y_train = return_similar_objects_into_sample(x_train_ind, y_train, data, additional_logs, label)
    x_test = [data[ind][0] for ind in x_test_ind]
    return x_train, x_test, y_train, y_test


def perform_light_deduplication(data: list[tuple[str, str, str]]) -> tuple[dict[int, list[int]], list[int]]:
    text_messages_set = {}
    logs_to_train_idx = []
    additional_logs = {}
    for idx, text_message_data in enumerate(data):
        text_message = text_message_data[0]
        text_message_normalized = " ".join(sorted(
            text_processing.split_words(text_message, to_lower=True)))
        if text_message_normalized not in text_messages_set:
            logs_to_train_idx.append(idx)
            text_messages_set[text_message_normalized] = idx
            additional_logs[idx] = []
        else:
            additional_logs[text_messages_set[text_message_normalized]].append(idx)
    return additional_logs, logs_to_train_idx


def create_binary_target_data(label: str, data: list[tuple[str, str, str]]):
    additional_logs, logs_to_train_idx = perform_light_deduplication(data)
    labels_filtered = []
    for ind in logs_to_train_idx:
        if data[ind][1] == label or data[ind][2] == label:
            labels_filtered.append(1)
        else:
            labels_filtered.append(0)
    proportion_binary_labels = utils.calculate_proportions_for_labels(labels_filtered)
    if proportion_binary_labels < MINIMAL_LABEL_PROPORTION:
        logs_to_train_idx, labels_filtered, proportion_binary_labels = utils.balance_data(
            logs_to_train_idx, labels_filtered, MINIMAL_LABEL_PROPORTION)
    return logs_to_train_idx, labels_filtered, additional_logs, proportion_binary_labels


def train_several_times(new_model: DefectTypeModel, label: str, data: list[tuple[str, str, str]],
                        random_states: list[int],
                        baseline_model: Optional[DefectTypeModel] = None) -> tuple[list[float], list[float], bool]:
    my_random_states = random_states if random_states else TRAIN_DATA_RANDOM_STATES
    new_model_results = []
    baseline_model_results = []
    bad_data_proportion = False

    logs_to_train_idx, labels_filtered, additional_logs, proportion_binary_labels = create_binary_target_data(
        label, data)

    if proportion_binary_labels < MINIMAL_LABEL_PROPORTION:
        LOGGER.debug('Train data has a bad proportion: %.3f', proportion_binary_labels)
        bad_data_proportion = True

    if not bad_data_proportion:
        for random_state in my_random_states:
            x_train, x_test, y_train, y_test = split_train_test(
                logs_to_train_idx, data, labels_filtered, additional_logs, label, random_state=random_state)
            new_model.train_model(label, x_train, y_train)
            LOGGER.debug('New model results')
            f1, accuracy = new_model.validate_model(label, x_test, y_test)
            new_model_results.append(f1)
            LOGGER.debug('Baseline model results')
            if baseline_model:
                f1, accuracy = baseline_model.validate_model(label, x_test, y_test)
                baseline_model_results.append(f1)
            else:
                baseline_model_results.append(0.0)
    return baseline_model_results, new_model_results, bad_data_proportion


class QueryResult(BaseModel):
    result: list[tuple[str, str, str]]
    error_count: int
    errors: list[str]


class DefectTypeModelTraining:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    es_client: EsClient
    baseline_model: DefectTypeModel
    model_chooser: Optional[ModelChooser]

    def __init__(self, app_config: ApplicationConfig, search_cfg: SearchConfig,
                 model_chooser: Optional[ModelChooser] = None) -> None:
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = EsClient(app_config=app_config)
        self.baseline_model = DefectTypeModel(object_saving.create_filesystem(search_cfg.GlobalDefectTypeModelFolder))
        self.baseline_model.load_model()
        self.model_chooser = model_chooser

    @staticmethod
    def get_messages_by_issue_type(issue_type_pattern: str) -> dict[str, Any]:
        return {
            "_source": [DATA_FIELD, "issue_type", "launch_id", '_id'],
            "sort": {"start_time": "desc"},
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}}
                    ],
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "wildcard": {
                                            "issue_type": {"value": issue_type_pattern, "case_insensitive": True}
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "should": [
                        {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                    ]
                }
            }
        }

    def execute_data_query(self, project_index_name: str, query: str) -> QueryResult:
        errors = []
        error_count = 0
        query_result = []
        while error_count <= RETRY_COUNT:
            try:
                query_result = elasticsearch.helpers.scan(
                    self.es_client.es_client, query=self.get_messages_by_issue_type(query), index=project_index_name,
                    size=self.app_config.esChunkNumber)
                break
            except Exception as exc:
                # Throttling, out of memory, connection error
                LOGGER.exception(exc)
                errors.append(utils.extract_exception(exc))
                sleep(RETRY_PAUSES[error_count] if len(RETRY_PAUSES) < error_count else RETRY_PAUSES[-1])
                error_count += 1
        if error_count >= RETRY_COUNT:
            return QueryResult(result=[], error_count=error_count, errors=errors)
        data = []
        message_launch_dict = set()
        for r in query_result:
            detected_message = r['_source'][DATA_FIELD]
            if not detected_message.strip():
                continue
            text_message_normalized = text_processing.normalize_message(detected_message)
            issue_type = r["_source"]["issue_type"]
            message_info = (text_message_normalized, r["_source"]["launch_id"], issue_type)
            if message_info not in message_launch_dict:
                data.append((detected_message, issue_type[:2], issue_type))
                message_launch_dict.add(message_info)
            if len(data) >= self.search_cfg.MaxLogsForDefectTypeModel:
                break
        return QueryResult(result=data, error_count=error_count, errors=errors)

    def query_label(self, query: str, index: str, stat: Optional[dict[str, Any]]) -> QueryResult:
        LOGGER.debug(f'Query to gather data {query}.')
        time_querying = time()
        found_data = self.execute_data_query(index, query)
        time_spent = time() - time_querying
        LOGGER.debug(f'Finished querying "{query}" for {time_spent:.2f} s')
        if stat:
            stat['time_spent'] = time_spent
            stat['data_size'] = len(found_data.result)
        return found_data

    def query_data(self, projects: list[int],
                   stat_data_storage: Optional[DefaultDict[str, dict[str, Any]]]) -> list[tuple[str, str, str]]:
        data = []
        errors = []
        error_count = 0
        start_time = time()
        for project in projects:
            project_index_name = text_processing.unite_project_name(project, self.app_config.esProjectIndexPrefix)
            for label in BASE_ISSUE_CLASS_INDEXES:
                query = f'{label}???'
                found_data = self.query_label(
                    query, project_index_name, stat_data_storage[label] if stat_data_storage else None)
                errors.append(found_data.errors)
                error_count += found_data.error_count
                data.extend(found_data.result)
                query = f'{label}_*'
                found_data = self.execute_data_query(project_index_name, query)
                errors.append(found_data.errors)
                error_count += found_data.error_count
                sub_labels = {l[2] for l in found_data.result}
                for sub_label in sub_labels:
                    found_data = self.query_label(
                        sub_label, project_index_name, stat_data_storage[sub_label] if stat_data_storage else None)
                    errors.append(found_data.errors)
                    error_count += found_data.error_count
                    data.extend(found_data.result)

        LOGGER.debug(f'Data gathered: {len(data)}')
        if stat_data_storage:
            stat_data_storage['all']['data_size'] = len(data)
            stat_data_storage['all']['errors'] = errors
            stat_data_storage['all']['errors_count'] = error_count
            stat_data_storage['all']['time_spent'] = start_time - time()
        return data

    @staticmethod
    def get_info_template(project_info: TrainInfo, label: str, baseline_model: str, model_name: str) -> dict[str, Any]:
        return {'method': 'training', 'sub_model_type': label, 'model_type': project_info.model_type.name,
                'baseline_model': [baseline_model], 'new_model': [model_name],
                'project_id': project_info.project, 'model_saved': 0, 'p_value': 1.0,
                'data_proportion': 0.0, 'baseline_mean_metric': 0.0, 'new_model_mean_metric': 0.0,
                'bad_data_proportion': 0, 'metric_name': 'F1', 'errors': [], 'errors_count': 0,
                'time_spent': 0.0}

    def copy_model_part_from_baseline(self, new_model, label):
        if label not in self.baseline_model.models:
            if label in new_model.models:
                del new_model.models[label]
            if label in new_model.count_vectorizer_models:
                del new_model.count_vectorizer_models[label]
        else:
            new_model.models[label] = self.baseline_model.models[label]
            _count_vectorizer = self.baseline_model.count_vectorizer_models[label]
            new_model.count_vectorizer_models[label] = _count_vectorizer

    def train_several_times(self, new_model: DefectTypeModel, label: str, data: list[tuple[str, str, str]],
                            random_states: list[int]) -> tuple[list[float], list[float], bool]:
        return train_several_times(new_model, label, data, random_states, self.baseline_model)

    def train(self, project_info: TrainInfo) -> tuple[int, dict[str, dict[str, Any]]]:
        start_time = time()
        model_name = f'{project_info.model_type.name}_model_{datetime.now().strftime("%d.%m.%y")}'
        baseline_model = os.path.basename(self.search_cfg.GlobalDefectTypeModelFolder)
        new_model_folder = f'{project_info.model_type.name}_model/{model_name}/'
        new_model = CustomDefectTypeModel(
            object_saving.create(self.app_config, project_info.project, new_model_folder))

        train_log_info = DefaultDict(lambda _, k: self.get_info_template(project_info, k, baseline_model, model_name))
        projects = [project_info.project]
        if project_info.additional_projects:
            projects.extend(project_info.additional_projects)
        data = self.query_data(projects, train_log_info)
        unique_labels = {l[2] for l in data}

        data_proportion_min = 1.0
        p_value_max = 0.0
        all_bad_data = 1
        custom_models = []
        f1_chosen_models = []
        f1_baseline_models = []
        errors = []
        errors_count = 0
        for label in unique_labels:
            time_training = time()
            LOGGER.debug(f'Label to train the model {label}')

            baseline_model_results, new_model_results, bad_data_proportion = self.train_several_times(
                new_model, label, data, TRAIN_DATA_RANDOM_STATES)

            use_custom_model = False
            if not bad_data_proportion:
                LOGGER.debug("Baseline test results %s", baseline_model_results)
                LOGGER.debug("New model test results %s", new_model_results)
                f_value, p_value = stats.f_oneway(baseline_model_results, new_model_results)
                if p_value is None:
                    p_value = 1.0
                train_log_info[label]["p_value"] = p_value
                baseline_mean_f1 = np.mean(baseline_model_results)
                mean_f1 = np.mean(new_model_results)
                train_log_info[label]["baseline_mean_metric"] = baseline_mean_f1
                train_log_info[label]["new_model_mean_metric"] = mean_f1

                if p_value < 0.05 and mean_f1 > baseline_mean_f1 and mean_f1 >= 0.4:
                    p_value_max = max(p_value_max, p_value)
                    use_custom_model = True
                all_bad_data = 0
                LOGGER.debug(
                    """Model training validation results:
                        p-value=%.3f mean baseline=%.3f mean new model=%.3f""",
                    p_value, np.mean(baseline_model_results), np.mean(new_model_results))
            train_log_info[label]["bad_data_proportion"] = int(bad_data_proportion)

            if use_custom_model:
                LOGGER.debug("Custom model '%s' should be saved" % label)
                max_train_result_idx = np.argmax(new_model_results)[0]

                baseline_model_results, new_model_results, bad_data_proportion = self.train_several_times(
                    new_model, label, data, TRAIN_DATA_RANDOM_STATES[max_train_result_idx])

                if not bad_data_proportion:
                    train_log_info[label]["model_saved"] = 1
                    custom_models.append(label)
                else:
                    train_log_info[label]["model_saved"] = 0
            else:
                self.copy_model_part_from_baseline(new_model, label)
                if train_log_info[label]["baseline_mean_metric"] > 0.001:
                    f1_baseline_models.append(train_log_info[label]["baseline_mean_metric"])
                    f1_chosen_models.append(train_log_info[label]["baseline_mean_metric"])
            train_log_info[label]["time_spent"] += (time() - time_training)

        LOGGER.debug(f'Custom models were for labels: {custom_models}')
        if len(custom_models):
            LOGGER.debug("The custom model should be saved")
            train_log_info["all"]["model_saved"] = 1
            train_log_info["all"]["p_value"] = p_value_max
            if self.model_chooser:
                self.model_chooser.delete_old_model(project_info.model_type, project_info.project)
            new_model.save_model()

        time_spent = time() - start_time
        LOGGER.info("Finished for %d s", time_spent)
        train_log_info["all"]["time_spent"] = time_spent
        train_log_info["all"]["data_proportion"] = data_proportion_min
        train_log_info["all"]["errors_count"] += errors_count
        train_log_info["all"]["errors"].extend(errors)
        train_log_info["all"]["baseline_mean_metric"] = np.mean(
            f1_baseline_models) if f1_baseline_models else 0.0
        train_log_info["all"]["new_model_mean_metric"] = np.mean(
            f1_chosen_models) if f1_chosen_models else 0.0
        train_log_info["all"]["bad_data_proportion"] = all_bad_data
        for label in train_log_info:
            train_log_info[label]["gather_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            train_log_info[label]["module_version"] = [self.app_config.appVersion]
        return len(data), train_log_info
