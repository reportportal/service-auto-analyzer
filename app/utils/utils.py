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
import traceback
import warnings
from collections import Counter
from functools import wraps
from typing import Any, Optional

import numpy as np
import requests
from requests import RequestException

from app.commons import logging
from app.commons.model import launch_objects
from app.commons.model.launch_objects import SimilarityResult
from app.ml.predictor import PredictionResult
from app.utils.text_processing import remove_credentials_from_url, split_words

logger = logging.getLogger("analyzerApp.utils")


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


def send_request(
    url: str, method: str, username: Optional[str], password: Optional[str], data: Optional[str] = None
) -> Optional[Any]:
    """Send request with specified url and http method"""

    kwargs: dict[str, Any] = {}
    if username:
        kwargs["auth"] = (username, password)

    if data:
        kwargs["data"] = data
        kwargs["headers"] = {"Content-Type": "application/json"}

    try:
        response = requests.request(method.lower(), url, **kwargs)
        response.raise_for_status()
        if response.content:
            return json.loads(response.text, strict=False)
    except RequestException as err:
        logger.exception(f"Error sending {method} request to URL: {remove_credentials_from_url(url)}", exc_info=err)
    return None


MINIMAL_VALUE_FOR_GOOD_PROPORTION = 2
ERROR_LOG_LEVEL = 40000


def calculate_proportions_for_labels(labels: list[int]) -> float:
    counted_labels = Counter(labels)
    if len(counted_labels.keys()) >= 2:
        min_val = min(counted_labels.values())
        max_val = max(counted_labels.values())
        if min_val > MINIMAL_VALUE_FOR_GOOD_PROPORTION:
            return np.round(min_val / max_val, 3)
    return 0.0


def calculate_log_weight(log_level: int, message_length: int, max_message_length: int) -> float:
    """Calculate log contribution weight for central-weighted scoring.

    ERROR log level (40000) has level weight 1.0; other levels are scaled relatively.
    The longest message has length weight 1.0; shorter messages are scaled relatively.

    :param log_level: Numeric log level
    :param message_length: Length of the current log message
    :param max_message_length: Maximum message length within the compared group
    :return: Combined weight
    """
    level_weight = log_level / ERROR_LOG_LEVEL if ERROR_LOG_LEVEL > 0 else 0.0
    length_weight = message_length / max_message_length if max_message_length > 0 else 0.0
    return level_weight * length_weight


def topological_sort(feature_graph: dict[int, list[int]]) -> list[int]:
    visited = {}
    for key_ in feature_graph:
        visited[key_] = 0
    stack = []

    for key_ in feature_graph:
        if visited[key_] == 0:
            stack_vertices = [key_]
            while len(stack_vertices):
                vert = stack_vertices[-1]
                if vert not in visited:
                    continue
                if visited[vert] == 1:
                    stack_vertices.pop()
                    visited[vert] = 2
                    stack.append(vert)
                else:
                    visited[vert] = 1
                    for key_i in feature_graph[vert]:
                        if key_i not in visited:
                            continue
                        if visited[key_i] == 0:
                            stack_vertices.append(key_i)
    return stack


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


def build_more_like_this_query(
    min_should_match: str,
    log_message,
    field_name: str = "message",
    boost: float = 1.0,
    override_min_should_match: Optional[str] = None,
    max_query_terms: int = 50,
):
    return {
        "more_like_this": {
            "fields": [field_name],
            "like": log_message,
            "min_doc_freq": 1,
            "min_term_freq": 1,
            "minimum_should_match": override_min_should_match or "5<" + min_should_match,
            "max_query_terms": max_query_terms,
            "boost": boost,
        }
    }


def extract_clustering_setting(cluster_id):
    if not cluster_id or int(cluster_id) == 0:
        return False
    last_bit = cluster_id % 10
    return (last_bit % 2) == 1


def create_path(query: dict, path: tuple[str, ...], value: Any) -> Any:
    """Create path in a dictionary and assign passed value on the last element in path."""
    path_length = len(path)
    last_element = path[path_length - 1]
    current_node = query
    for i in range(path_length - 1):
        element = path[i]
        if element not in current_node:
            current_node[element] = {}
        current_node = current_node[element]
    if last_element not in current_node:
        current_node[last_element] = value
    return current_node[last_element]


def compute_if_absent(on: dict[str, Any], key: str, default_value: Any) -> Any:
    """Compute value for key in dictionary if it is absent.

    It is here just to mute SonarLint warning about possible KeyError.
    """
    if key not in on:
        on[key] = default_value
    return on[key]


def append_aa_ma_boosts(query: dict[str, Any], search_cfg: launch_objects.SearchConfig) -> None:
    """Append boosts for auto-analyzed and manually-analyzed fields to ES/OS query.

    :param query: ES/OS query
    :param search_cfg: Search configuration
    """
    should = create_path(query, ("query", "bool", "should"), [])
    boost_aa = search_cfg.BoostAA
    boost_ma = search_cfg.BoostMA
    if boost_aa > boost_ma:
        should.append(
            {
                "term": {
                    "is_auto_analyzed": {
                        "value": True,
                        "boost": boost_aa - boost_ma,
                    }
                }
            }
        )
    else:
        should.append(
            {
                "term": {
                    "is_auto_analyzed": {
                        "value": False,
                        "boost": boost_ma - boost_aa,
                    }
                }
            }
        )


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


def group_predictions_by_test_item(
    prediction_results: list[PredictionResult],
) -> dict[int, list[PredictionResult]]:
    """Group prediction results by the found test item ID.

    :param prediction_results: List of prediction results from the Predictor
    :return: Dictionary mapping test item ID to its prediction results
    """
    groups: dict[int, list[PredictionResult]] = {}
    for result in prediction_results:
        test_item_id = result.data["mrHit"].source.test_item
        if test_item_id not in groups:
            groups[test_item_id] = []
        groups[test_item_id].append(result)
    return groups


def score_and_rank_test_items(
    grouped_predictions: dict[int, list[PredictionResult]],
) -> list[tuple[float, PredictionResult]]:
    """Calculate central-weighted score per test item and rank them.

    For each test item group:
    1. Find max message length across all logs
    2. Compute weight and weighted score for each prediction
    3. Compute weighted average score
    4. Pick the most significant log (highest weight)

    :param grouped_predictions: Predictions grouped by test item ID
    :return: List of (weighted_avg, most_significant_result) sorted descending
    """
    ranked: list[tuple[float, PredictionResult]] = []
    for _test_item_id, results in grouped_predictions.items():
        max_message_length = max(len(r.data["mrHit"].source.message) for r in results)
        if max_message_length <= 0:
            max_message_length = 1

        weighted_sum = 0.0
        weight_sum = 0.0
        best_weight = -1.0
        best_result = results[0]

        for result in results:
            log_level = result.data["mrHit"].source.log_level
            msg_len = len(result.data["mrHit"].source.message)
            weight = calculate_log_weight(log_level, msg_len, max_message_length)
            prob = result.probability[1]
            weighted_sum += prob * weight
            weight_sum += weight
            if weight > best_weight:
                best_weight = weight
                best_result = result

        weighted_avg = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        ranked.append((weighted_avg, best_result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def prepare_restrictions_by_issue_type(filter_no_defect: bool = True) -> list[dict]:
    if filter_no_defect:
        return [{"wildcard": {"issue_type": "ti*"}}, {"wildcard": {"issue_type": "nd*"}}]
    return [{"term": {"issue_type": "ti001"}}]


def get_max_similarity_idx(per_log_similarity: list[SimilarityResult]) -> int:
    max_similarity = 0.0
    best_log_idx = 0
    for log_idx, sim_obj in enumerate(per_log_similarity):
        if max_similarity < sim_obj.similarity:
            max_similarity = sim_obj.similarity
            best_log_idx = log_idx
    return best_log_idx
