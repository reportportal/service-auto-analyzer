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
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from time import time
from typing import Any, Optional, Type, cast

import numpy as np
import scipy.stats as stats
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split

from app.commons import logging, namespace_finder, object_saving
from app.commons.model.db import Hit
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.log_item_index import LogItemIndexData
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.os_client import OsClient
from app.ml.boosting_featurizer import BoostingFeaturizer
from app.ml.models import BoostingDecisionMaker, CustomBoostingDecisionMaker, DefectTypeModel
from app.ml.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.ml.training import (
    DEFAULT_RANDOM_SEED,
    NEGATIVE_RATIO_MAX,
    NEGATIVE_RATIO_MIN,
    TRAIN_DATA_RANDOM_STATES,
    TrainingEntry,
    build_issue_history_query,
    select_history_negative_types,
    validate_proportions,
)
from app.utils import text_processing, utils
from app.utils.defaultdict import DefaultDict
from app.utils.os_migration import (
    bucket_sort_logs_by_similarity,
    build_search_results,
    extract_inner_hit_logs,
    get_request_logs,
)
from app.utils.utils import is_supported_issue_type, normalize_base_issue_type, normalize_issue_type, safe_int

LOGGER = logging.getLogger("analyzerApp.trainingAnalysisModel")
SMOTE_PROPORTION = 0.4
MIN_POSITIVE_CASES_FOR_SMOTE = 5
MIN_P_VALUE = 0.05
METRIC = "F1"

ITEM_FIELDS_TO_RETRIEVE = [
    "test_item_id",
    "test_item_name",
    "unique_id",
    "test_case_hash",
    "launch_id",
    "launch_name",
    "issue_type",
    "is_auto_analyzed",
    "start_time",
    "logs",
    "issue_history",
]


def split_data(
    data: list[list[float]], labels: list[int], random_state: int
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    x_ids: list[int] = list(range(len(data)))
    x_train_ids, x_test_ids, y_train, y_test = train_test_split(
        x_ids, labels, test_size=0.1, random_state=random_state, stratify=labels
    )
    x_train = [data[idx] for idx in x_train_ids]
    x_test = [data[idx] for idx in x_test_ids]
    return x_train, x_test, y_train, y_test


def transform_data_from_feature_lists(
    feature_list: list[list[float]], cur_features: list[int], desired_features: list[int]
) -> list[list[float]]:
    previously_gathered_features = utils.fill_previously_gathered_features(feature_list, cur_features)
    gathered_data = utils.gather_feature_list(previously_gathered_features, desired_features)
    return gathered_data


def fill_metric_stats(
    baseline_model_metric_result: list[float], new_model_metric_results: list[float], info_dict: dict[str, Any]
) -> None:
    _, p_value = stats.f_oneway(baseline_model_metric_result, new_model_metric_results)
    if p_value is None or math.isnan(p_value):
        p_value = 1.0
    info_dict["p_value"] = p_value
    mean_metric = np.mean(new_model_metric_results)
    baseline_mean_metric = np.mean(baseline_model_metric_result)
    info_dict["baseline_mean_metric"] = baseline_mean_metric
    info_dict["new_model_mean_metric"] = mean_metric
    info_dict["gather_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def train_several_times(
    new_model: BoostingDecisionMaker,
    data: list[list[float]],
    labels: list[int],
    random_states: Optional[list[int]] = None,
    baseline_model: Optional[BoostingDecisionMaker] = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]], bool, float]:
    new_model_results: dict[str, list[float]] = DefaultDict(lambda _, __: [])
    baseline_model_results: dict[str, list[float]] = DefaultDict(lambda _, __: [])
    my_random_states: list[int] = random_states if random_states else TRAIN_DATA_RANDOM_STATES

    bad_data, data_proportion = validate_proportions(labels)
    if not bad_data:
        for random_state in my_random_states:
            x_train, x_test, y_train, y_test = split_data(data, labels, random_state)
            LOGGER.debug(
                f"Train data split with random state {random_state}: train size {len(x_train)}, test size "
                + str(len(x_test))
            )
            proportion_binary_labels = utils.calculate_proportions_for_labels(y_train)
            LOGGER.debug(f"Train data proportion: {proportion_binary_labels:.2f}")
            positives_train = sum(1 for label in y_train if label == 1)
            negatives_train = sum(1 for label in y_train if label == 0)
            if MIN_POSITIVE_CASES_FOR_SMOTE > positives_train > 1 and negatives_train > 1:
                LOGGER.debug(
                    "Applying SMOTE due to low positive count: %d (negatives=%d)",
                    positives_train,
                    negatives_train,
                )
                oversample = BorderlineSMOTE(sampling_strategy=SMOTE_PROPORTION, random_state=random_state)
                x_train, y_train = oversample.fit_resample(x_train, y_train)
            new_model.train_model(x_train, y_train)
            LOGGER.debug("New model results")
            new_model_results[METRIC].append(new_model.validate_model(x_test, y_test))
            LOGGER.debug("Baseline results")
            if baseline_model:
                x_test_for_baseline = transform_data_from_feature_lists(
                    x_test, new_model.feature_ids, baseline_model.feature_ids
                )
                baseline_model_results[METRIC].append(baseline_model.validate_model(x_test_for_baseline, y_test))
    return baseline_model_results, new_model_results, bad_data, data_proportion


def get_info_template(
    project_info: TrainInfo, baseline_model: str, model_name: str, metric_name: str
) -> dict[str, Any]:
    return {
        "method": "training",
        "sub_model_type": "all",
        "model_type": project_info.model_type.name,
        "baseline_model": [baseline_model],
        "new_model": [model_name],
        "project_id": str(project_info.project),
        "model_saved": 0,
        "p_value": 1.0,
        "data_size": 0,
        "data_proportion": 0.0,
        "baseline_mean_metric": 0.0,
        "new_model_mean_metric": 0.0,
        "bad_data_proportion": 0,
        "metric_name": metric_name,
        "errors": [],
        "errors_count": 0,
    }


def get_candidates_by_type(candidate_issue_types: list[str]) -> dict[str, list[int]]:
    candidates_by_type: dict[str, list[int]] = defaultdict(list)
    for idx, issue_type in enumerate(candidate_issue_types):
        candidates_by_type[issue_type].append(idx)
    return candidates_by_type


def select_candidate_entries(
    candidate_issue_types: list[str],
    positive_issue_type: str,
    history_negative_types: list[str],
) -> list[tuple[int, int]]:
    """
    Select candidate entries and assign labels based on issue history.

    :param candidate_issue_types: Issue types aligned with feature rows
    :param positive_issue_type: Issue type treated as positive
    :param history_negative_types: Negative issue types from history (max 2)
    :return: List of (index, label) tuples
    """
    candidates_by_type = get_candidates_by_type(candidate_issue_types)

    positive_indices = candidates_by_type.get(positive_issue_type, [])
    if not positive_indices:
        return []

    history_indices: list[int] = []
    for issue_type in history_negative_types:
        indices = candidates_by_type.get(issue_type, [])
        if indices:
            history_indices.append(indices[0])

    other_negatives: list[int] = []
    for issue_type, indices in candidates_by_type.items():
        if issue_type == positive_issue_type or issue_type in history_negative_types:
            continue
        other_negatives.extend(indices)

    rng = random.Random(DEFAULT_RANDOM_SEED)
    rng.shuffle(other_negatives)
    selected_negatives: list[int] = []
    if other_negatives:
        selected_negatives.append(other_negatives.pop(0))

    selected_negatives.extend(history_indices)

    remaining_negatives = [idx for idx in other_negatives if idx not in selected_negatives]
    max_negatives = len(positive_indices) * NEGATIVE_RATIO_MAX
    rng.shuffle(remaining_negatives)
    for idx in remaining_negatives:
        if len(selected_negatives) >= max_negatives:
            break
        selected_negatives.append(idx)

    selected_positive_indices = list(positive_indices)
    min_negatives = len(positive_indices) * NEGATIVE_RATIO_MIN
    if len(selected_negatives) < min_negatives:
        max_positives = max(1, len(selected_negatives) // NEGATIVE_RATIO_MIN)
        if max_positives < len(selected_positive_indices):
            rng.shuffle(selected_positive_indices)
            selected_positive_indices = selected_positive_indices[:max_positives]

    selected_entries: list[tuple[int, int]] = [(idx, 1) for idx in selected_positive_indices]
    selected_entries.extend((idx, 0) for idx in selected_negatives)
    return selected_entries


def _make_synthetic_test_item_id(base_id: int, index: int) -> int:
    """
    Build a deterministic synthetic test item ID.

    :param base_id: Base test item identifier
    :param index: Index of the synthetic entry
    :return: Synthetic test item ID
    """
    base = abs(base_id)
    if base == 0:
        base = 1000000
    return base * 10 + index + 1


def build_history_negative_hits(
    request_logs: list[LogItemIndexData],
    history_negative_types: list[str],
    base_test_item_id: str,
) -> list[Hit[LogItemIndexData]]:
    """
    Create synthetic hits for history-negative issue types using request logs.

    :param request_logs: Log items used as search requests
    :param history_negative_types: Issue types from history to label as negatives
    :param base_test_item_id: Test item identifier from the request item
    :return: Synthetic log hits labeled with history issue types
    """
    base_id = safe_int(base_test_item_id)
    synthetic_hits: list[Hit[LogItemIndexData]] = []
    for idx, issue_type in enumerate(history_negative_types):
        synthetic_test_item_id = _make_synthetic_test_item_id(base_id, idx)
        for log_item in request_logs:
            synthetic_log = log_item.model_copy(
                update={"issue_type": issue_type, "test_item": synthetic_test_item_id},
                deep=True,
            )
            synthetic_hits.append(
                Hit[LogItemIndexData].from_dict(
                    {"_id": synthetic_log.log_id, "_score": 0.0001, "_source": synthetic_log}
                )
            )
    return synthetic_hits


def build_entries_from_item(project_id: int, test_item: TestItemIndexData) -> list[TrainingEntry[TestItemIndexData]]:
    issue_history = test_item.issue_history or []
    if not issue_history:
        return []
    positive_issue_type = normalize_issue_type(issue_history[-1].issue_type)
    if not positive_issue_type or not is_supported_issue_type(positive_issue_type):
        return []

    negative_issue_types = select_history_negative_types(issue_history, positive_issue_type)
    negative_issue_types = [issue_type for issue_type in negative_issue_types if is_supported_issue_type(issue_type)]

    logs = list(test_item.logs or [])
    if not logs:
        return []

    entries: list[TrainingEntry[TestItemIndexData]] = [
        TrainingEntry[TestItemIndexData](
            data=test_item,
            project_id=project_id,
            issue_type=normalize_base_issue_type(positive_issue_type),
            is_positive=True,
        )
    ]
    for negative_issue_type in negative_issue_types:
        entries.append(
            TrainingEntry[TestItemIndexData](
                data=test_item,
                project_id=project_id,
                issue_type=normalize_base_issue_type(negative_issue_type),
                is_positive=False,
            )
        )
    return entries


def has_inner_hits(hit: Hit[TestItemIndexData]) -> bool:
    inner_hits = hit.inner_hits or {}
    return bool(inner_hits.get("logs", {}).get("hits", {}).get("hits", []))


class AnalysisModelTraining:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient
    model_type: ModelType
    model_class: Type[BoostingDecisionMaker]
    featurizer_class: Type[BoostingFeaturizer]
    baseline_folder: Optional[str]
    baseline_model: Optional[BoostingDecisionMaker]
    model_chooser: ModelChooser
    features: list[int]
    monotonous_features: list[int]
    n_estimators: int
    max_depth: int

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        model_type: ModelType,
        model_chooser: ModelChooser,
        model_class: Optional[Type[BoostingDecisionMaker]] = None,
        use_baseline_features: bool = True,
        *,
        os_client: Optional[OsClient] = None,
    ) -> None:
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.due_proportion = 0.05
        self.due_proportion_to_smote = 0.4
        self.os_client = os_client or OsClient(app_config=app_config)
        self.model_type = model_type
        if model_type is ModelType.suggestion:
            self.baseline_folder = self.search_cfg.SuggestBoostModelFolder
            self.features = text_processing.transform_string_feature_range_into_list(
                self.search_cfg.SuggestBoostModelFeatures
            )
            self.monotonous_features = text_processing.transform_string_feature_range_into_list(
                self.search_cfg.SuggestBoostModelMonotonousFeatures
            )
            self.n_estimators = self.search_cfg.SuggestBoostModelNumEstimators
            self.max_depth = self.search_cfg.SuggestBoostModelMaxDepth
            self.featurizer_class = SuggestBoostingFeaturizer
        elif model_type is ModelType.auto_analysis:
            self.baseline_folder = self.search_cfg.BoostModelFolder
            self.features = text_processing.transform_string_feature_range_into_list(
                self.search_cfg.AutoBoostModelFeatures
            )
            self.monotonous_features = text_processing.transform_string_feature_range_into_list(
                self.search_cfg.AutoBoostModelMonotonousFeatures
            )
            self.n_estimators = self.search_cfg.AutoBoostModelNumEstimators
            self.max_depth = self.search_cfg.AutoBoostModelMaxDepth
            self.featurizer_class = BoostingFeaturizer
        else:
            raise ValueError(f"Incorrect model type {model_type}")

        self.model_class = model_class if model_class else CustomBoostingDecisionMaker

        if self.baseline_folder:
            self.baseline_model = BoostingDecisionMaker(object_saving.create_filesystem(self.baseline_folder))
            self.baseline_model.load_model()
            # Take features from baseline model if this is retraining
            if use_baseline_features:
                self.features = self.baseline_model.feature_ids
                self.monotonous_features = list(self.baseline_model.monotonous_features)
        else:
            self.baseline_model = None

        if not self.features:
            raise ValueError('No feature config found, please either correct values in "search_cfg" parameter')

        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_chooser = model_chooser

    def _get_config_for_boosting(self, number_of_log_lines: int, namespaces) -> dict[str, Any]:
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
            "time_weight_decay": self.search_cfg.TimeWeightDecay,
        }

    def _build_similar_items_query(
        self,
        request_logs: list[LogItemIndexData],
        request_test_item_id: str,
        min_should_match: float,
        exclude_issue_type: str = "",
    ) -> dict[str, Any]:
        log_messages = [
            log_item.whole_message
            for log_item in request_logs
            if log_item.whole_message and log_item.whole_message.strip()
        ]
        if not log_messages:
            return {}

        min_should_match_str = text_processing.prepare_es_min_should_match(min_should_match)
        nested_should = [
            utils.build_more_like_this_query(
                min_should_match_str,
                message,
                field_name="logs.whole_message",
                boost=1.0,
                max_query_terms=self.search_cfg.MaxQueryTerms,
            )
            for message in log_messages
        ]
        inner_hits_source = [
            "logs.log_id",
            "logs.log_time",
            "logs.log_level",
            "logs.cluster_id",
            "logs.cluster_message",
            "logs.cluster_with_numbers",
            "logs.original_message",
            "logs.message",
            "logs.message_extended",
            "logs.message_without_params_extended",
            "logs.message_without_params_and_brackets",
            "logs.detected_message",
            "logs.detected_message_with_numbers",
            "logs.detected_message_extended",
            "logs.detected_message_without_params_extended",
            "logs.detected_message_without_params_and_brackets",
            "logs.stacktrace",
            "logs.stacktrace_extended",
            "logs.only_numbers",
            "logs.potential_status_codes",
            "logs.found_exceptions",
            "logs.found_tests_and_methods",
            "logs.urls",
            "logs.message_params",
            "logs.whole_message",
        ]
        nested_query = {
            "nested": {
                "path": "logs",
                "score_mode": "max",
                "query": {"bool": {"should": nested_should}},
                "inner_hits": {
                    "size": max(5, len(log_messages)),
                    "_source": inner_hits_source,
                },
            }
        }
        query: dict[str, Any] = {
            "_source": [
                "test_item_id",
                "test_item_name",
                "unique_id",
                "test_case_hash",
                "launch_id",
                "launch_name",
                "issue_type",
                "is_auto_analyzed",
                "start_time",
            ],
            "size": self.app_config.esChunkNumber,
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": "issue_type"}}],
                    "must_not": [{"term": {"test_item_id": str(request_test_item_id)}}],
                    "must": [nested_query],
                }
            },
        }
        if exclude_issue_type:
            query["query"]["bool"]["must_not"].append({"term": {"issue_type": exclude_issue_type}})
        utils.append_aa_ma_boosts(query, self.search_cfg)
        return query

    def _collect_similar_hits(
        self,
        project_id: int,
        request_logs: list[LogItemIndexData],
        request_test_item_id: str,
        positive_issue_type: str,
    ) -> list[Hit[TestItemIndexData]]:
        query = self._build_similar_items_query(request_logs, request_test_item_id, 0.4)
        if not query:
            return []
        hits = [hit for hit in self.os_client.search(project_id, query) or [] if has_inner_hits(hit)]
        issue_types = {normalize_issue_type(hit.source.issue_type) for hit in hits if hit.source.issue_type}
        if positive_issue_type and (not issue_types or issue_types == {positive_issue_type}):
            extra_query = self._build_similar_items_query(
                request_logs, request_test_item_id, 0.4, exclude_issue_type=positive_issue_type
            )
            extra_hits = [hit for hit in self.os_client.search(project_id, extra_query) or [] if has_inner_hits(hit)]
            hits_by_id: dict[str, Hit[TestItemIndexData]] = {}
            for hit in hits + extra_hits:
                hits_by_id[str(hit.source.test_item_id)] = hit
            hits = list(hits_by_id.values())
        return hits

    def _query_data(self, projects: list[int]) -> list[TrainingEntry[TestItemIndexData]]:
        data: list[TrainingEntry[TestItemIndexData]] = []
        query = build_issue_history_query(self.app_config.esChunkNumber, ITEM_FIELDS_TO_RETRIEVE)

        for project_id in projects:
            for hit in self.os_client.search(project_id, query) or []:
                data.extend(build_entries_from_item(project_id, hit.source))
        return data

    def _featurize_data(
        self, test_items: list[TrainingEntry[TestItemIndexData]], features: list[int]
    ) -> tuple[list[list[float]], list[int]]:
        full_data_features, labels = [], []
        project_to_entries: dict[int, list[TrainingEntry[TestItemIndexData]]] = defaultdict(list)
        for item in test_items:
            if not item.project_id:
                continue
            project_to_entries[item.project_id].append(item)

        for project_id, entries in project_to_entries.items():
            namespaces = self.namespace_finder.get_chosen_namespaces(project_id)
            raw_defect_type_model = self.model_chooser.choose_model(project_id, ModelType.defect_type)
            defect_type_model = cast(DefectTypeModel, raw_defect_type_model) if raw_defect_type_model else None
            for entry in entries:
                test_item: TestItemIndexData = entry.data
                issue_history = list(test_item.issue_history or [])
                if not issue_history:
                    continue
                positive_issue_type = normalize_issue_type(issue_history[-1].issue_type or test_item.issue_type)
                if not positive_issue_type:
                    continue
                history_negative_types = select_history_negative_types(issue_history, positive_issue_type)

                request_logs = get_request_logs(test_item, entry.issue_type)
                if not request_logs:
                    continue

                similar_hits = self._collect_similar_hits(
                    project_id, request_logs, str(test_item.test_item_id), positive_issue_type
                )
                if not similar_hits:
                    continue

                found_hits = extract_inner_hit_logs(similar_hits)
                if history_negative_types:
                    found_hits = [
                        hit
                        for hit in found_hits
                        if normalize_issue_type(hit.source.issue_type) not in history_negative_types
                    ]
                    found_hits.extend(
                        build_history_negative_hits(
                            request_logs,
                            history_negative_types,
                            str(test_item.test_item_id),
                        )
                    )
                if not found_hits:
                    continue

                buckets = bucket_sort_logs_by_similarity(request_logs, found_hits)
                search_results = build_search_results(request_logs, buckets)
                if not search_results:
                    continue

                _boosting_data_gatherer: BoostingFeaturizer = self.featurizer_class(
                    search_results, self._get_config_for_boosting(-1, namespaces), feature_ids=features
                )

                _boosting_data_gatherer.set_defect_type_model(defect_type_model)
                feature_data, candidate_names = _boosting_data_gatherer.gather_features_info()
                if not feature_data or not candidate_names:
                    continue

                scores_by_type = _boosting_data_gatherer.find_most_relevant_by_type()
                candidate_issue_types: list[str] = []
                for candidate_name in candidate_names:
                    score_info = scores_by_type.get(candidate_name)
                    if not score_info:
                        candidate_issue_types.append("")
                        continue
                    candidate_hit = score_info.get("mrHit")
                    if not candidate_hit:
                        candidate_issue_types.append("")
                        continue
                    candidate_issue_types.append(normalize_issue_type(candidate_hit.source.issue_type))

                filtered_features: list[list[float]] = []
                filtered_issue_types: list[str] = []
                for idx, issue_type in enumerate(candidate_issue_types):
                    if not issue_type:
                        continue
                    filtered_features.append(feature_data[idx])
                    filtered_issue_types.append(issue_type)

                selected_entries = select_candidate_entries(
                    filtered_issue_types, positive_issue_type, history_negative_types
                )
                if not selected_entries:
                    continue
                for idx, label in selected_entries:
                    full_data_features.append(filtered_features[idx])
                    labels.append(label)
        return full_data_features, labels

    def _train_several_times(
        self,
        new_model: BoostingDecisionMaker,
        data: list[list[float]],
        labels: list[int],
        random_states: Optional[list[int]] = None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], bool, float]:
        return train_several_times(new_model, data, labels, random_states, self.baseline_model)

    def train(self, project_info: TrainInfo) -> tuple[int, dict[str, Any]]:
        time_training = time()
        model_name = f'{project_info.model_type.name}_model_{datetime.now().strftime("%Y-%m-%d")}'
        baseline_model = os.path.basename(self.baseline_folder) if self.baseline_folder else ""
        new_model_folder = f"{project_info.model_type.name}_model/{model_name}/"

        LOGGER.info(f'Train "{self.model_type.name}" model using class: {self.model_class}')
        new_model = self.model_class(
            object_saving.create(self.app_config, project_id=project_info.project, path=new_model_folder),
            features=self.features,
            monotonous_features=self.monotonous_features,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )

        train_log_info: DefaultDict[str, dict[str, Any]] = DefaultDict(
            lambda _, k: get_info_template(project_info, baseline_model, model_name, k)
        )

        LOGGER.debug(f"Initialized model training {project_info.model_type.name}")
        projects = [project_info.project]
        if project_info.additional_projects:
            projects.extend(project_info.additional_projects)
        raw_train_data = self._query_data(projects)
        train_data, labels = self._featurize_data(raw_train_data, new_model.feature_ids)
        LOGGER.debug(f"Loaded data for model training {self.model_type.name}")

        baseline_model_results, new_model_results, bad_data, data_proportion = self._train_several_times(
            new_model, train_data, labels
        )
        for metric in new_model_results:
            train_log_info[metric]["data_size"] = len(labels)
            train_log_info[metric]["bad_data_proportion"] = int(bad_data)
            train_log_info[metric]["data_proportion"] = data_proportion

        use_custom_model = False
        mean_metric_results: list[float] = []
        mean_metric_count: dict[int, int] = defaultdict(lambda: 1)
        if not bad_data:
            LOGGER.debug(f"Baseline test results {baseline_model_results}")
            LOGGER.debug(f"New model test results {new_model_results}")
            p_values = []
            new_metrics_better = True
            for metric, metric_results in new_model_results.items():
                info_dict = train_log_info[metric]
                fill_metric_stats(baseline_model_results[metric], metric_results, info_dict)
                p_value = info_dict["p_value"]
                p_values.append(p_value)
                mean_metric = info_dict["new_model_mean_metric"]
                baseline_mean_metric = info_dict["baseline_mean_metric"]
                new_metrics_better = new_metrics_better and mean_metric > baseline_mean_metric and mean_metric >= 0.4
                LOGGER.info(
                    f"Model training validation results: p-value={p_value:.3f}; mean {metric} metric "
                    f"baseline={baseline_mean_metric:.3f}; mean new model={mean_metric:.3f}."
                )
                if mean_metric_results:
                    for i in range(len(metric_results)):
                        metric_count = mean_metric_count[i] + 1
                        mean_metric_result = mean_metric_results[i]
                        mean_metric_results[i] = mean_metric_result + (
                            (mean_metric_result - metric_results[i]) / metric_count
                        )
                        mean_metric_count[i] = metric_count
                else:
                    mean_metric_results = metric_results.copy()

            if max(p_values) < MIN_P_VALUE and new_metrics_better:
                use_custom_model = True

        if use_custom_model:
            LOGGER.debug("Custom model should be saved")
            max_train_result_idx = int(np.argmax(mean_metric_results)) if mean_metric_results else 0
            best_random_state = TRAIN_DATA_RANDOM_STATES[max_train_result_idx]

            LOGGER.info(f"Perform final training with random state: {best_random_state}")
            self._train_several_times(new_model, train_data, labels, [best_random_state])
            if self.model_chooser:
                self.model_chooser.delete_old_model(project_info.model_type, project_info.project)
            new_model.save_model()
            train_log_info[METRIC]["model_saved"] = 1
        else:
            train_log_info[METRIC]["model_saved"] = 0

        time_spent = time() - time_training
        train_log_info[METRIC]["time_spent"] = time_spent
        train_log_info[METRIC]["module_version"] = [self.app_config.appVersion]

        LOGGER.info(f"Finished for {time_spent} s")
        return len(train_data), train_log_info
