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

import dataclasses
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Any, Optional, Type

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split

from app.commons import logging, object_saving
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.os_client import OsClient
from app.ml.models import CustomDefectTypeModel, DefectTypeModel
from app.ml.models.defect_type_model import DATA_FIELD
from app.ml.training import normalize_issue_type, select_history_negative_types, validate_proportions
from app.utils.defaultdict import DefaultDict

LOGGER = logging.getLogger("analyzerApp.trainingDefectTypeModel")
DEFAULT_RANDOM_SEED = 1257
TRAIN_DATA_RANDOM_STATES = [DEFAULT_RANDOM_SEED, 1873, 1917, 2477, 3449, 353, 4561, 5417, 6427, 2029, 2137]
NEGATIVE_RATIO_MIN = 2
NEGATIVE_RATIO_MAX = 4
MINIMAL_LABEL_PROPORTION = 0.25
TEST_DATA_PROPORTION = 0.1
MINIMAL_DATA_LENGTH_FOR_TRAIN = 50
MIN_P_VALUE = 0.05


@dataclass(frozen=True)
class TrainingEntry:
    message: str
    issue_type: str
    is_positive: bool


def _get_issue_type(issue_type: str) -> str:
    if not issue_type:
        return issue_type
    return _get_base_issue_type(issue_type) if re.match(r"^[a-z]{2}\d{,3}$", issue_type) else issue_type


def split_train_test(
    train_data: list[str],
    labels_filtered: list[int],
    random_state: int = 1257,
) -> tuple[list[str], list[str], list[int], list[int]]:
    x_train, x_test, y_train, y_test = train_test_split(
        train_data,
        labels_filtered,
        test_size=TEST_DATA_PROPORTION,
        random_state=random_state,
        stratify=labels_filtered,
    )
    return x_train, x_test, y_train, y_test


def balance_data(
    train_data: list[TrainingEntry],
) -> list[TrainingEntry]:
    """Make existing train data balanced for the given label.

    This function shorten the amount of negative cases if there are to many of them and extend them out of existing
    data if there are too few of them.

    :param train_data: Existing train data based on item history.
    """
    cases: dict[str, list[TrainingEntry]] = defaultdict(list)
    for entry in train_data:
        cases[entry.issue_type].append(entry)

    cases_num: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    negative_indexes: list[int] = []
    for i, entry in enumerate(train_data):
        current = cases_num[entry.issue_type]
        if entry.is_positive:
            cases_num[entry.issue_type] = current[0] + 1, current[1]
        else:
            negative_indexes.append(i)
            cases_num[entry.issue_type] = current[0], current[1] + 1

    rnd = random.Random(DEFAULT_RANDOM_SEED)
    results: list[TrainingEntry] = train_data.copy()
    for issue_type, (positive, negative) in cases_num.items():
        if positive <= 0:
            continue
        max_negative_cases = positive * NEGATIVE_RATIO_MAX
        additional_negative_cases: list[TrainingEntry] = []
        for issue, case_list in cases.items():
            if issue_type == issue:
                continue
            for case in case_list:
                if not case.is_positive:
                    continue  # ignore uncertainty
                additional_negative_cases.append(dataclasses.replace(case, issue_type=issue_type, is_positive=False))
        if not additional_negative_cases:
            continue
        rnd.shuffle(additional_negative_cases)
        if negative + len(additional_negative_cases) <= max_negative_cases:
            # We can append all additional negative cases
            cases[issue_type].extend(additional_negative_cases)
        else:
            # Need to balance negative cases
            rnd.shuffle(negative_indexes)
            case_num_to_remove = negative + len(additional_negative_cases) - max_negative_cases
            if negative > positive:
                remove_history_cases_num = negative - positive
            else:
                remove_history_cases_num = 0
            if remove_history_cases_num < case_num_to_remove:
                remove_additional_cases_num = case_num_to_remove - remove_history_cases_num
            else:
                remove_additional_cases_num = 0
            additional_negative_cases = additional_negative_cases[:-remove_additional_cases_num]
            cases_to_remove = negative_indexes[:remove_history_cases_num]
            i = 0
            for i, case_idx in enumerate(cases_to_remove):
                results[case_idx] = additional_negative_cases[i]
            if i <= len(additional_negative_cases):
                results.extend(additional_negative_cases[i:])
    return results


def create_binary_target_data(label: str, data: list[TrainingEntry]) -> tuple[list[str], list[int]]:
    labels_filtered = []
    for entry in data:
        if label == entry.issue_type:
            labels_filtered.append(1 if entry.is_positive else 0)
        else:
            labels_filtered.append(0)
    return [entry.message for entry in data], labels_filtered


def _get_log_message(log_data: LogData) -> Optional[str]:
    message = getattr(log_data, DATA_FIELD, None)
    if message is None:
        return None
    if not str(message).strip():
        return None
    return str(message)


def train_several_times(
    new_model: DefectTypeModel,
    label: str,
    data: list[TrainingEntry],
    random_states: Optional[list[int]] = None,
    baseline_model: Optional[DefectTypeModel] = None,
) -> tuple[list[float], list[float], bool, float]:
    my_random_states = random_states if random_states else TRAIN_DATA_RANDOM_STATES
    new_model_results = []
    baseline_model_results = []

    train_data, labels_filtered = create_binary_target_data(label, data)
    bad_data_proportion, data_proportion = validate_proportions(labels_filtered)
    if not bad_data_proportion:
        for random_state in my_random_states:
            x_train, x_test, y_train, y_test = split_train_test(train_data, labels_filtered, random_state=random_state)
            new_model.train_model(label, x_train, y_train, random_state)
            LOGGER.debug("New model results")
            new_model_results.append(new_model.validate_model(label, x_test, y_test))
            if baseline_model:
                LOGGER.debug("Baseline model results")
                baseline_model_results.append(baseline_model.validate_model(label, x_test, y_test))
            else:
                baseline_model_results.append(0.0)
    return baseline_model_results, new_model_results, bad_data_proportion, data_proportion


def copy_model_part_from_baseline(label: str, new_model: DefectTypeModel, baseline_model: DefectTypeModel) -> None:
    if label not in baseline_model.models:
        if label in new_model.models:
            del new_model.models[label]
        if label in new_model.count_vectorizer_models:
            del new_model.count_vectorizer_models[label]
    else:
        new_model.models[label] = baseline_model.models[label]
        _count_vectorizer = baseline_model.count_vectorizer_models[label]
        new_model.count_vectorizer_models[label] = _count_vectorizer


def get_info_template(project_info: TrainInfo, baseline_model: str, model_name: str, label: str) -> dict[str, Any]:
    return {
        "method": "training",
        "sub_model_type": label,
        "model_type": project_info.model_type.name,
        "baseline_model": [baseline_model],
        "new_model": [model_name],
        "project_id": project_info.project,
        "model_saved": 0,
        "p_value": 1.0,
        "data_size": 0,
        "data_proportion": 0.0,
        "baseline_mean_metric": 0.0,
        "new_model_mean_metric": 0.0,
        "bad_data_proportion": 0,
        "metric_name": "F1",
        "errors": [],
        "errors_count": 0,
        "time_spent": 0.0,
    }


def _get_base_issue_type(issue_type: str) -> str:
    return issue_type[:2] if issue_type else ""


def _is_supported_issue_type(issue_type: str) -> bool:
    return _get_base_issue_type(issue_type) != "ti"


def build_entries_from_item(test_item: TestItemIndexData) -> list[TrainingEntry]:
    issue_history = list(test_item.issue_history or [])
    if not issue_history:
        return []
    positive_issue_type = normalize_issue_type(issue_history[-1].issue_type)
    if not positive_issue_type or not _is_supported_issue_type(positive_issue_type):
        return []

    negative_issue_types = select_history_negative_types(issue_history, positive_issue_type)
    negative_issue_types = [issue_type for issue_type in negative_issue_types if _is_supported_issue_type(issue_type)]

    logs = list(test_item.logs or [])
    if not logs:
        return []

    entries: list[TrainingEntry] = []
    for log_data in logs:
        log_message = _get_log_message(log_data)
        if not log_message:
            continue
        entries.append(
            TrainingEntry(
                message=log_message,
                issue_type=_get_issue_type(positive_issue_type),
                is_positive=True,
            )
        )
        for negative_issue_type in negative_issue_types:
            entries.append(
                TrainingEntry(
                    message=log_message,
                    issue_type=_get_issue_type(negative_issue_type),
                    is_positive=False,
                )
            )
    return entries


class DefectTypeModelTraining:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient
    baseline_model: Optional[DefectTypeModel] = None
    model_chooser: Optional[ModelChooser]
    model_class: Type[DefectTypeModel]

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        model_chooser: Optional[ModelChooser] = None,
        model_class: Optional[Type[DefectTypeModel]] = None,
        *,
        os_client: Optional[OsClient] = None,
    ) -> None:
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.os_client = os_client or OsClient(app_config=app_config)
        if search_cfg.GlobalDefectTypeModelFolder:
            self.baseline_model = DefectTypeModel(
                object_saving.create_filesystem(search_cfg.GlobalDefectTypeModelFolder)
            )
            self.baseline_model.load_model()
        self.model_chooser = model_chooser
        self.model_class = model_class if model_class else CustomDefectTypeModel

    def _build_issue_history_query(self) -> dict[str, Any]:
        return {
            "_source": ["test_item_id", "issue_history", "logs"],
            "size": self.app_config.esChunkNumber,
            "query": {
                "nested": {
                    "path": "issue_history",
                    "query": {"exists": {"field": "issue_history.issue_type"}},
                }
            },
        }

    def _query_data(
        self, projects: list[int], stat_data_storage: Optional[DefaultDict[str, dict[str, Any]]]
    ) -> list[TrainingEntry]:
        data: list[TrainingEntry] = []
        start_time = time()
        query = self._build_issue_history_query()

        for project in projects:
            for hit in self.os_client.search(project, query) or []:
                data.extend(build_entries_from_item(hit.source))

        LOGGER.debug(f"Data gathered: {len(data)}")
        if stat_data_storage:
            label_counts: dict[str, int] = {}
            for entry in data:
                if entry.is_positive and entry.issue_type:
                    label_counts[entry.issue_type] = label_counts.get(entry.issue_type, 0) + 1
            for label, count in label_counts.items():
                stat_data_storage[label]["data_size"] = count
            stat_data_storage["all"]["data_size"] = len(data)
            stat_data_storage["all"]["time_spent"] = time() - start_time
        return data

    def _train_several_times(
        self,
        new_model: DefectTypeModel,
        label: str,
        data: list[TrainingEntry],
        random_states: Optional[list[int]] = None,
    ) -> tuple[list[float], list[float], bool, float]:
        return train_several_times(new_model, label, data, random_states, self.baseline_model)

    def train(self, project_info: TrainInfo) -> tuple[int, dict[str, dict[str, Any]]]:
        start_time = time()
        model_name = f'{project_info.model_type.name}_model_{datetime.now().strftime("%Y-%m-%d")}'
        baseline_model_folder = os.path.basename(self.search_cfg.GlobalDefectTypeModelFolder)
        new_model_folder = f"{project_info.model_type.name}_model/{model_name}/"

        LOGGER.info(f'Train "{ModelType.defect_type.name}" model using class: {self.model_class}')
        new_model = self.model_class(
            object_saving.create(self.app_config, project_info.project, new_model_folder),
            n_estimators=self.search_cfg.DefectTypeModelNumEstimators,
        )

        train_log_info: DefaultDict[str, dict[str, Any]] = DefaultDict(
            lambda _, k: get_info_template(project_info, baseline_model_folder, model_name, k)
        )
        projects = [project_info.project]
        if project_info.additional_projects:
            projects.extend(project_info.additional_projects)
        data = self._query_data(projects, train_log_info)
        train_data = balance_data(data)

        LOGGER.debug(f"Loaded data for model training {project_info.model_type.name}")

        data_proportion_min = 1.0
        p_value_max = 0.0
        all_bad_data = 1
        custom_models = []
        f1_chosen_models = []
        f1_baseline_models = []

        unique_labels = {entry.issue_type for entry in data if entry.issue_type}
        for label in unique_labels:
            time_training = time()
            LOGGER.info(f"Label to train the model {label}")

            (baseline_model_results, new_model_results, bad_data_proportion, proportion_binary_labels) = (
                self._train_several_times(new_model, label, train_data, TRAIN_DATA_RANDOM_STATES)
            )
            data_proportion_min = min(proportion_binary_labels, data_proportion_min)

            use_custom_model = False
            if not bad_data_proportion:
                LOGGER.debug(f"Baseline test results {baseline_model_results}")
                LOGGER.debug(f"New model test results {new_model_results}")
                _, p_value = stats.f_oneway(baseline_model_results, new_model_results)
                if p_value is None or math.isnan(p_value):
                    p_value = 1.0
                train_log_info[label]["p_value"] = p_value
                baseline_mean_f1 = np.mean(baseline_model_results)
                mean_f1 = np.mean(new_model_results)
                train_log_info[label]["baseline_mean_metric"] = baseline_mean_f1
                train_log_info[label]["new_model_mean_metric"] = mean_f1

                if p_value < MIN_P_VALUE and mean_f1 > baseline_mean_f1 and mean_f1 >= 0.4:
                    p_value_max = max(p_value_max, p_value)
                    use_custom_model = True
                all_bad_data = 0
                LOGGER.info(
                    f"Model training validation results: p-value={p_value:.3f}; mean F1 metric "
                    f"baseline={baseline_mean_f1:.3f}; mean new model={mean_f1:.3f}."
                )
            train_log_info[label]["bad_data_proportion"] = int(bad_data_proportion)

            if use_custom_model:
                LOGGER.debug(f"Custom model {label} should be saved")
                max_train_result_idx = int(np.argmax(new_model_results))
                best_random_state = TRAIN_DATA_RANDOM_STATES[max_train_result_idx]

                LOGGER.info(f"Perform final training with random state: {best_random_state}")
                self._train_several_times(new_model, label, train_data, [best_random_state])

                train_log_info[label]["model_saved"] = 1
                custom_models.append(label)
            else:
                train_log_info[label]["model_saved"] = 0
                if self.baseline_model:
                    copy_model_part_from_baseline(label, new_model, self.baseline_model)
                if train_log_info[label]["baseline_mean_metric"] > 0.001:
                    f1_baseline_models.append(train_log_info[label]["baseline_mean_metric"])
                    f1_chosen_models.append(train_log_info[label]["baseline_mean_metric"])
            train_log_info[label]["time_spent"] += time() - time_training

        LOGGER.debug(f"Custom models were for labels: {custom_models}")
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
        train_log_info["all"]["baseline_mean_metric"] = np.mean(f1_baseline_models) if f1_baseline_models else 0.0
        train_log_info["all"]["new_model_mean_metric"] = np.mean(f1_chosen_models) if f1_chosen_models else 0.0
        train_log_info["all"]["bad_data_proportion"] = all_bad_data
        for label in train_log_info:
            train_log_info[label]["gather_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            train_log_info[label]["module_version"] = [self.app_config.appVersion]
        return len(data), train_log_info
