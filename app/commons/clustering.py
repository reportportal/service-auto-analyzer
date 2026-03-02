#  Copyright 2025 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import hashlib
import heapq
from collections import defaultdict
from time import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.commons import logging
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.clustering")


def __calculate_hashes(messages: list[list[str]], n_gram_length: int = 2, n_permutations: int = 64) -> list[list[str]]:
    if n_gram_length <= 0:
        raise ValueError("n_gram_length must be greater than 0")
    hashes: list[list[str]] = []
    for words in messages:
        hash_print: set[str] = set()
        ngram_num = len(words)
        if n_gram_length > 1:
            if len(words) > n_gram_length:
                ngram_num = len(words) - n_gram_length + 1
            else:
                ngram_num = 1
        for i in range(ngram_num):
            hash_print.add(
                hashlib.md5(" ".join(words[i : i + n_gram_length]).encode("utf-8"), usedforsecurity=False).hexdigest()
            )
        hashes.append(list(heapq.nlargest(n_permutations, hash_print)))
    return hashes


def __similarity_grouping(
    messages: list[list[str]],
    block_size: int = 1000,
    threshold: float = 0.95,
) -> dict[int, int]:
    if len(messages) == 0:
        return {}
    if len(messages) == 1:
        return {0: 0}
    num_of_blocks = int(np.ceil(len(messages) / block_size))
    groups = {}
    global_ind = 0
    for idx in range(num_of_blocks):
        for jdx in range((idx + 1 if num_of_blocks > 1 else idx), num_of_blocks):
            block_i = messages[idx * block_size : (idx + 1) * block_size]
            block_j = messages[jdx * block_size : (jdx + 1) * block_size] if num_of_blocks > 1 else []
            indices_looked = list(range(idx * block_size, idx * block_size + len(block_i))) + list(
                range(jdx * block_size, jdx * block_size + len(block_j))
            )

            _count_vector = CountVectorizer(binary=True, analyzer=lambda x: x)
            vectors: csr_matrix = _count_vector.fit_transform(block_i + block_j).astype(np.int8)
            vectors_count_words = np.asarray(np.sum(vectors, axis=1))
            similarities = cosine_similarity(vectors)

            for seq_num_i in range(len(indices_looked)):
                i = indices_looked[seq_num_i]
                if i not in groups:
                    groups[i] = global_ind
                    global_ind += 1
                for seq_num_j in range(seq_num_i + 1, len(indices_looked)):
                    j = indices_looked[seq_num_j]
                    if j not in groups:
                        min_words_num = min(
                            vectors_count_words[seq_num_i][0],
                            vectors_count_words[seq_num_j][0],
                        )
                        recalculated_threshold = utils.calculate_threshold(min_words_num, threshold)
                        similarity = similarities[seq_num_i][seq_num_j]
                        if similarity >= recalculated_threshold:
                            groups[j] = groups[i]
    return groups


def __unite_groups_by_hashes(messages: list[list[str]], threshold: float = 0.95) -> dict[int, list[int]]:
    start_time = time()
    hash_prints = __calculate_hashes(messages)
    if not any(hash_prints):
        return {}
    hash_groups: dict[int, int] = __similarity_grouping(hash_prints, threshold=threshold)
    rearranged_groups: dict[int, list[int]] = defaultdict(list)
    for key, cluster in hash_groups.items():
        rearranged_groups[cluster].append(key)
    LOGGER.debug("Time for finding hash groups: %.2f s", time() - start_time)
    return dict(rearranged_groups)


def __find_groups_by_similarity(messages: list[list[str]], threshold: float = 0.95) -> dict[int, list[int]]:
    if len(messages) == 0:
        return {}
    groups_to_check = __unite_groups_by_hashes(messages, threshold=threshold)
    group_id = 0
    start_time = time()
    rearranged_groups: dict[int, list[int]] = {}
    for hash_cluster_id, group in groups_to_check.items():
        selected_texts: list[list[str]] = [messages[i] for i in group]
        text_groups = __similarity_grouping(selected_texts, threshold=threshold)
        new_group_id = group_id
        for key, value in text_groups.items():
            cluster = value + group_id
            new_group_id = max(group_id, cluster)
            real_id = groups_to_check[hash_cluster_id][key]
            if cluster not in rearranged_groups:
                rearranged_groups[cluster] = []
            rearranged_groups[cluster].append(real_id)
        new_group_id += 1
        group_id = new_group_id
    LOGGER.debug("Time for finding groups: %.2f s", time() - start_time)
    return rearranged_groups


def __perform_light_deduplication(messages: list[str]) -> tuple[list[list[str]], dict[int, list[int]]]:
    text_messages_set = {}
    messages_to_cluster = []
    ids_with_duplicates = {}
    new_id = 0
    for idx, text_message in enumerate(messages):
        text_message_normalized = text_processing.preprocess_text_for_similarity(text_message)
        text_message_joined = " ".join(text_message_normalized)
        if text_message_joined not in text_messages_set:
            messages_to_cluster.append(text_message_normalized)
            text_messages_set[text_message_joined] = new_id
            ids_with_duplicates[new_id] = [idx]
            new_id += 1
        else:
            ids_with_duplicates[text_messages_set[text_message_joined]].append(idx)
    return messages_to_cluster, ids_with_duplicates


def find_clusters(messages: list[str], threshold: float = 0.95) -> dict[int, list[int]]:
    """Cluster similar text messages using a two-stage approach for performance optimization.

    The function uses hash-based pre-filtering followed by text-based refinement to reduce
    O(N²) comparisons. Messages are first deduplicated, then grouped by n-gram hash similarity,
    and finally clustered by cosine similarity within each hash group.

    :param messages: List of text messages to cluster (e.g., error logs, stack traces)
    :param threshold: Similarity threshold (0.0-1.0). Default 0.95. Higher = stricter grouping
    :return: Dictionary mapping cluster IDs to lists of original message indices
             Example: {0: [0, 2, 5], 1: [1, 3]} means messages [0,2,5] are similar
    """
    messages_to_cluster, ids_with_duplicates = __perform_light_deduplication(messages)
    groups = __find_groups_by_similarity(messages_to_cluster, threshold=threshold)
    new_groups: dict[int, list[int]] = {}
    for cluster in groups:
        new_log_ids: list[int] = []
        for idx in groups[cluster]:
            new_log_ids.extend(ids_with_duplicates[idx])
        new_groups[cluster] = new_log_ids
    return new_groups
