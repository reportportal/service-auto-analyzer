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
from typing import Any

import numpy as np
import requests

from app.commons import logging
from app.commons.model import launch_objects
from app.utils.text_processing import split_words, remove_credentials_from_url

logger = logging.getLogger("analyzerApp.utils")
ERROR_LOGGING_LEVEL = 40000


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


def validate_folder(folder_path: str) -> bool:
    """Check that passed path points to a directory and it exists."""
    return folder_path and folder_path.strip() and os.path.exists(folder_path) and os.path.isdir(folder_path)


def validate_file(file_path: str) -> bool:
    """Check that passed path points to a file and it exists."""
    return file_path and file_path.strip() and os.path.exists(file_path) and os.path.isfile(file_path)


def extract_real_id(elastic_id):
    real_id = str(elastic_id)
    if real_id[-2:] == "_m":
        return int(real_id[:-2])
    return int(real_id)


def jaccard_similarity(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0


def choose_issue_type(predicted_labels, predicted_labels_probability,
                      issue_type_names, scores_by_issue_type):
    predicted_issue_type = ""
    max_prob = 0.0
    max_val_start_time = None
    global_idx = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1:
            issue_type = issue_type_names[i]
            chosen_type = scores_by_issue_type[issue_type]
            start_time = chosen_type["mrHit"]["_source"]["start_time"]
            predicted_prob = round(predicted_labels_probability[i][1], 4)
            if (predicted_prob > max_prob) or \
                    ((predicted_prob == max_prob) and  # noqa
                     (max_val_start_time is None or start_time > max_val_start_time)):
                max_prob = predicted_prob
                predicted_issue_type = issue_type
                global_idx = i
                max_val_start_time = start_time
    return predicted_issue_type, max_prob, global_idx


def send_request(url, method, username, password):
    """Send request with specified url and http method"""
    try:
        if username.strip() and password.strip():
            response = requests.get(url, auth=(username, password)) if method == "GET" else {}
        else:
            response = requests.get(url) if method == "GET" else {}
        data = response._content.decode("utf-8")
        content = json.loads(data, strict=False)
        return content
    except Exception as err:
        logger.error("Error with loading url: %s",
                     remove_credentials_from_url(url))
        logger.error(err)
    return []


def extract_all_exceptions(bodies):
    logs_with_exceptions = []
    for log_body in bodies:
        exceptions = [
            exc.strip() for exc in log_body["_source"]["found_exceptions"].split()]
        logs_with_exceptions.append(
            launch_objects.LogExceptionResult(
                logId=int(log_body["_id"]),
                foundExceptions=exceptions))
    return logs_with_exceptions


def calculate_proportions_for_labels(labels: list[int]) -> float:
    counted_labels = Counter(labels)
    if len(counted_labels.keys()) >= 2:
        min_val = min(counted_labels.values())
        max_val = max(counted_labels.values())
        if max_val > 0:
            return np.round(min_val / max_val, 3)
    return 0.0


def rebalance_data(train_data, train_labels, due_proportion):
    one_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 1]
    zero_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 0]
    zero_count = len(zero_data)
    one_count = len(one_data)
    all_data = []
    all_data_labels = []
    real_proportion = 0.0 if zero_count < 0.001 else np.round(one_count / zero_count, 3)
    if zero_count > 0 and real_proportion < due_proportion:
        all_data.extend(one_data)
        all_data_labels.extend([1] * len(one_data))
        random.seed(1763)
        random.shuffle(zero_data)
        zero_size = int(one_count * (1 / due_proportion) - 1)
        all_data.extend(zero_data[:zero_size])
        all_data_labels.extend([0] * zero_size)
        real_proportion = calculate_proportions_for_labels(all_data_labels)
        if real_proportion / due_proportion >= 0.9:
            real_proportion = due_proportion

    random.seed(1257)
    random.shuffle(all_data)
    random.seed(1257)
    random.shuffle(all_data_labels)
    return all_data, all_data_labels, real_proportion


def topological_sort(feature_graph):
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


def to_number_list(features_list):
    feature_numbers_list = []
    for feature_name in features_list.split(";"):
        feature_name = feature_name.split("_")[0]
        try:
            feature_numbers_list.append(int(feature_name))
        except:  # noqa
            try:
                feature_numbers_list.append(float(feature_name))
            except:  # noqa
                pass
    return feature_numbers_list


def fill_prevously_gathered_features(feature_list, feature_ids):
    previously_gathered_features = {}
    try:
        if type(feature_ids) is str:
            feature_ids = to_number_list(feature_ids)
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


def gather_feature_list(gathered_data_dict, feature_ids, to_list=False):
    features_array = None
    axis_x_size = max(map(lambda x: len(x), gathered_data_dict.values()))
    if axis_x_size <= 0:
        return []
    for idx, feature in enumerate(feature_ids):
        if feature not in gathered_data_dict or len(gathered_data_dict[feature]) == 0:
            gathered_data_dict[feature] = [[0.0] for _ in range(axis_x_size)]
        if features_array is None:
            features_array = np.asarray(gathered_data_dict[feature])
        else:
            features_array = np.concatenate([features_array, gathered_data_dict[feature]], axis=1)
    return features_array.tolist() if to_list else features_array


def extract_exception(err):
    err_message = traceback.format_exception_only(type(err), err)
    if len(err_message):
        err_message = err_message[-1]
    else:
        err_message = ""
    return err_message


def get_allowed_number_of_missed(cur_threshold):
    if cur_threshold >= 0.95 and cur_threshold <= 0.99:
        return 1
    if cur_threshold >= 0.9 and cur_threshold < 0.95:
        return 2
    if cur_threshold >= 0.8 and cur_threshold < 0.9:
        return 3
    return 0


def calculate_threshold(
        text_size, cur_threshold, min_recalculated_threshold=0.8):
    if not text_size:
        return cur_threshold
    allowed_words_missed = get_allowed_number_of_missed(cur_threshold)
    new_threshold = cur_threshold
    for words_num in range(allowed_words_missed, 0, -1):
        threshold = (text_size - allowed_words_missed) / text_size
        if threshold >= min_recalculated_threshold:
            new_threshold = round(threshold, 2)
            break
    return min(new_threshold, cur_threshold)


def calculate_threshold_for_text(text, cur_threshold, min_recalculated_threshold=0.8):
    text_size = len(split_words(text))
    return calculate_threshold(
        text_size, cur_threshold,
        min_recalculated_threshold=min_recalculated_threshold)


def build_more_like_this_query(min_should_match: str, log_message,
                               field_name: str = "message", boost: float = 1.0,
                               override_min_should_match=None,
                               max_query_terms: int = 50):
    return {"more_like_this": {
        "fields": [field_name],
        "like": log_message,
        "min_doc_freq": 1,
        "min_term_freq": 1,
        "minimum_should_match": override_min_should_match or "5<" + min_should_match,
        "max_query_terms": max_query_terms,
        "boost": boost}}


def append_potential_status_codes(query, log, *, boost=8.0, max_query_terms=50):
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
                max_query_terms=max_query_terms
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
