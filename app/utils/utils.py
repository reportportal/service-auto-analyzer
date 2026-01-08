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
import random
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


def validate_folder(folder_path: str) -> bool:
    """Check that passed path points to a directory and it exists."""
    return bool(folder_path and folder_path.strip() and os.path.exists(folder_path) and os.path.isdir(folder_path))


def validate_file(file_path: str) -> bool:
    """Check that passed path points to a file and it exists."""
    return bool(file_path and file_path.strip() and os.path.exists(file_path) and os.path.isfile(file_path))


def extract_real_id(elastic_id):
    real_id = str(elastic_id)
    if real_id.endswith("_m"):
        return int(real_id[:-2])
    return int(real_id)


def jaccard_similarity(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0


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


def extract_all_exceptions(bodies):
    logs_with_exceptions = []
    for log_body in bodies:
        exceptions = [exc.strip() for exc in log_body["_source"]["found_exceptions"].split()]
        logs_with_exceptions.append(
            launch_objects.LogExceptionResult(logId=int(log_body["_id"]), foundExceptions=exceptions)
        )
    return logs_with_exceptions


MINIMAL_VALUE_FOR_GOOD_PROPORTION = 2


def calculate_proportions_for_labels(labels: list[int]) -> float:
    counted_labels = Counter(labels)
    if len(counted_labels.keys()) >= 2:
        min_val = min(counted_labels.values())
        max_val = max(counted_labels.values())
        if min_val > MINIMAL_VALUE_FOR_GOOD_PROPORTION:
            return np.round(min_val / max_val, 3)
    return 0.0


def balance_data(
    train_data_indexes: list[int], train_labels: list[int], due_proportion: float
) -> tuple[list[int], list[int], float]:
    one_data = [train_data_indexes[i] for i in range(len(train_data_indexes)) if train_labels[i] == 1]
    zero_data = [train_data_indexes[i] for i in range(len(train_data_indexes)) if train_labels[i] == 0]
    zero_count = len(zero_data)
    one_count = len(one_data)
    min_count = min(zero_count, one_count)
    max_count = max(zero_count, one_count)
    if zero_count > one_count:
        min_data = one_data
        max_data = zero_data
        min_label = 1
        max_label = 0
    else:
        min_data = zero_data
        max_data = one_data
        min_label = 0
        max_label = 1

    all_data = []
    all_data_labels = []
    real_proportion = 0.0
    if min_count > MINIMAL_VALUE_FOR_GOOD_PROPORTION:
        real_proportion = np.round(min_count / max_count, 3)
    if min_count > 0 and real_proportion < due_proportion:
        all_data.extend(min_data)
        all_data_labels.extend([min_label] * len(min_data))
        random.seed(1763)
        random.shuffle(max_data)
        max_size = int(min_count * (1 / due_proportion) - 1)
        all_data.extend(max_data[:max_size])
        all_data_labels.extend([max_label] * max_size)
        real_proportion = calculate_proportions_for_labels(all_data_labels)
        if real_proportion / due_proportion >= 0.9:
            real_proportion = due_proportion

    random.seed(1257)
    random.shuffle(all_data)
    random.seed(1257)
    random.shuffle(all_data_labels)
    return all_data, all_data_labels, real_proportion


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


def to_int_list(features_list: str) -> list[int]:
    feature_numbers_list = []
    for feature_name in features_list.split(";"):
        feature_name = feature_name.split("_")[0]
        feature_numbers_list.append(int(feature_name))
    return feature_numbers_list


def to_float_list(features_list: str) -> list[float]:
    feature_numbers_list = []
    for feature_name in features_list.split(";"):
        feature_name = feature_name.split("_")[0]
        feature_numbers_list.append(float(feature_name))
    return feature_numbers_list


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


def calculate_threshold_for_text(text, cur_threshold, min_recalculated_threshold=0.8):
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


def append_potential_status_codes(
    query: dict[str, Any], log: dict[str, Any], *, boost: float = 8.0, max_query_terms: int = 50
) -> None:
    potential_status_codes = log["_source"]["potential_status_codes"].strip()
    if potential_status_codes:
        number_of_status_codes = str(len(set(potential_status_codes.split())))
        query["query"]["bool"]["must"].append(
            build_more_like_this_query(
                "1",
                potential_status_codes,
                field_name="potential_status_codes",
                boost=boost,
                override_min_should_match=number_of_status_codes,
                max_query_terms=max_query_terms,
            )
        )


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
