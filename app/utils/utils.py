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

import json
import os
import re
import traceback
import warnings
from collections import Counter
from functools import wraps
from typing import Any, Optional

import numpy as np

from app.commons import logging
from app.commons.model.launch_objects import AnalyzerConf, SearchConfig, SimilarityResult
from app.utils.text_processing import split_words

logger = logging.getLogger("analyzerApp.utils")

NOT_CLASSIFIED_BASE_ISSUE_TYPE = "ti"
NO_DEFECT_BASE_ISSUE_TYPE = "nd"
CERTAIN_BASE_ISSUE_TYPES = {"ab", "pb", "si", "ti"}
CLASSIFIED_BASE_ISSUE_TYPES = set(list(CERTAIN_BASE_ISSUE_TYPES) + ["nd"])
ALL_BASE_ISSUE_TYPES = set(list(CLASSIFIED_BASE_ISSUE_TYPES) + [NOT_CLASSIFIED_BASE_ISSUE_TYPE])


def ignore_warnings(func):
    """Decorator for ignoring warnings"""

    @wraps(func)
    def _inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args, **kwargs)
        return result

    return _inner


def read_file(folder: str, filename: str) -> str:
    """Read file content as string (UTF-8)."""
    with open(os.path.join(folder, filename), "r") as file:
        return file.read()


def read_json_file(folder: str, filename: str, to_json: bool = False) -> Any:
    """Read fixture from file."""
    content = read_file(folder, filename)
    return content if not to_json else json.loads(content)


def read_resource_file(filename: str, to_json: bool = False) -> Any:
    """Read fixture from file."""
    return read_json_file("res", filename, to_json)


def validate_file(file_path: str) -> bool:
    """Check that passed path points to a file and it exists."""
    return bool(file_path and file_path.strip() and os.path.exists(file_path) and os.path.isfile(file_path))


def extract_real_id(elastic_id):
    real_id = str(elastic_id)
    if real_id.endswith("_m"):
        return int(real_id[:-2])
    return int(real_id)


MINIMAL_VALUE_FOR_GOOD_PROPORTION = 2


def calculate_proportions_for_labels(labels: list[int]) -> float:
    counted_labels = Counter(labels)
    if len(counted_labels.keys()) >= 2:
        min_val = min(counted_labels.values())
        max_val = max(counted_labels.values())
        if min_val > MINIMAL_VALUE_FOR_GOOD_PROPORTION:
            return np.round(min_val / max_val, 3)
    return 0.0


def fill_previously_gathered_features(
    feature_list: list[list[float]], feature_ids: list[int]
) -> dict[int, list[list[float]]]:
    previously_gathered_features: dict[int, list[list[float]]] = {}
    try:
        for i in range(len(feature_list)):
            for idx, feature in enumerate(feature_ids):
                if feature not in previously_gathered_features:
                    previously_gathered_features[feature] = []
                if len(previously_gathered_features[feature]) <= i:
                    previously_gathered_features[feature].append([])
                previously_gathered_features[feature][i].append(feature_list[i][idx])
    except Exception as err:
        logger.error(err)
    return previously_gathered_features


def gather_feature_list(gathered_data_dict: dict[int, list[list[float]]], feature_ids: list[int]) -> list[list[float]]:
    if not gathered_data_dict:
        return []
    axis_x_size = max([len(x) for x in gathered_data_dict.values()])
    if axis_x_size <= 0:
        return []

    # Initialize result list with empty lists for each row
    result: list[list[float]] = [[] for _ in range(axis_x_size)]

    for feature in feature_ids:
        if feature not in gathered_data_dict or len(gathered_data_dict[feature]) == 0:
            gathered_data_dict[feature] = [[0.0] for _ in range(axis_x_size)]

        # Append features from this feature ID to each row
        for row_idx, row in enumerate(gathered_data_dict[feature]):
            result[row_idx].extend(row)

    return result


def extract_exception(err: Exception) -> str:
    err_message_list = traceback.format_exception_only(type(err), err)
    if len(err_message_list):
        err_message = err_message_list[-1]
    else:
        err_message = ""
    return err_message


def get_allowed_number_of_missed(cur_threshold: float) -> int:
    if 0.95 <= cur_threshold <= 0.99:
        return 1
    if 0.9 <= cur_threshold < 0.95:
        return 2
    if 0.8 <= cur_threshold < 0.9:
        return 3
    return 0


def calculate_threshold(text_size: int, cur_threshold: float, min_recalculated_threshold: float = 0.8) -> float:
    if not text_size:
        return cur_threshold
    allowed_words_missed = get_allowed_number_of_missed(cur_threshold)
    new_threshold = cur_threshold
    for _ in range(allowed_words_missed, 0, -1):
        threshold = (text_size - allowed_words_missed) / text_size
        if threshold >= min_recalculated_threshold:
            new_threshold = round(threshold, 4)
            break
    return min(new_threshold, cur_threshold)


def calculate_threshold_for_text(text: str, cur_threshold: float, min_recalculated_threshold: float = 0.8):
    text_size = len(split_words(text))
    return calculate_threshold(text_size, cur_threshold, min_recalculated_threshold=min_recalculated_threshold)


def find_min_should_match_threshold(analyzer_config: AnalyzerConf, search_cfg: SearchConfig) -> int:
    """Get the minimum share of terms a found log must match, configured for the project or globally.

    :param analyzer_config: Project analyzer configuration
    :param search_cfg: Search configuration with the global default
    :return: Minimum should match, in percent
    """
    if analyzer_config.minShouldMatch > 0:
        return analyzer_config.minShouldMatch
    return int(search_cfg.MinShouldMatch.rstrip("%"))


def extract_clustering_setting(cluster_id):
    if not cluster_id or int(cluster_id) == 0:
        return False
    last_bit = cluster_id % 10
    return (last_bit % 2) == 1


def strip_path(path: str) -> str:
    """Strip trailing slashes from a path."""
    return path.strip().rstrip("/").rstrip("\\")


def normalize_issue_type(issue_type: Any) -> str:
    """
    Normalize issue type to lowercase string.

    :param issue_type: Raw issue type
    :return: Normalized issue type
    """
    if issue_type is None:
        return ""
    return str(issue_type).strip().lower()


def is_no_defect(issue_type: Optional[str]) -> bool:
    """Check whether an issue type is a "No Defect" one, of any subtype.

    :param issue_type: Issue type locator, e.g. `nd001`
    :return: True for "No Defect" issue types
    """
    return normalize_issue_type(issue_type).startswith(NO_DEFECT_BASE_ISSUE_TYPE)


def safe_int(value: Any) -> int:
    """
    Safely cast a value to integer.

    :param value: Value to cast
    :return: Integer value or 0 on failure
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_max_similarity_idx(per_log_similarity: list[SimilarityResult]) -> Optional[int]:
    if not per_log_similarity:
        return None
    max_similarity = 0.0
    best_log_idx = None
    for log_idx, sim_obj in enumerate(per_log_similarity):
        if max_similarity < sim_obj.similarity:
            max_similarity = sim_obj.similarity
            best_log_idx = log_idx
    return best_log_idx


def _get_base_issue_type(issue_type: str) -> str:
    return issue_type[:2].lower() if issue_type else ""


def is_supported_issue_type(issue_type: str) -> bool:
    if not issue_type:
        return False
    base_type = _get_base_issue_type(issue_type)
    return base_type in CLASSIFIED_BASE_ISSUE_TYPES and base_type != NOT_CLASSIFIED_BASE_ISSUE_TYPE


def normalize_base_issue_type(issue_type: str) -> str:
    if not issue_type:
        return issue_type
    return _get_base_issue_type(issue_type) if re.match(r"^[a-z]{2}\d{,3}$", issue_type) else issue_type
