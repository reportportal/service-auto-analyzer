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

import hashlib
import heapq
from time import time

import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

from app.commons import logging
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.clusterizer")


class Clusterizer:

    def calculate_hashes(self, messages: list[str], n_gram: int = 2, n_permutations: int = 64) -> list[list[str]]:
        hashes = []
        for message in messages:
            words = message.split()
            hash_print = set()
            len_words = (len(words) - n_gram) if len(words) > n_gram else len(words)
            for i in range(len_words):
                hash_print.add(hashlib.md5(" ".join(words[i : i + n_gram]).encode("utf-8")).hexdigest())
            hashes.append(list(heapq.nlargest(n_permutations, hash_print)))
        return hashes

    def find_groups_by_similarity(
        self, messages: list[str], groups_to_check: dict[int, list[int]], threshold: float = 0.95
    ) -> dict[int, list[int]]:
        if len(messages) == 0:
            return {}
        rearranged_groups = {}
        group_id = 0
        start_time = time()
        for key_word in groups_to_check:
            hash_prints = []
            for i in groups_to_check[key_word]:
                hash_prints.append(messages[i])
            hash_groups = self.similarity_groupping(hash_prints, for_text=True, threshold=threshold)
            new_group_id = group_id
            for key in hash_groups:
                cluster = hash_groups[key] + group_id
                new_group_id = max(group_id, cluster)
                real_id = groups_to_check[key_word][key]
                if cluster not in rearranged_groups:
                    rearranged_groups[cluster] = []
                rearranged_groups[cluster].append(real_id)
            new_group_id += 1
            group_id = new_group_id
        LOGGER.debug("Time for finding groups: %.2f s", time() - start_time)
        return rearranged_groups

    def similarity_groupping(
        self,
        hash_prints: list[list[str]] | list[str],
        block_size: int = 1000,
        for_text: bool = True,
        threshold: float = 0.95,
    ) -> dict[int, int]:
        num_of_blocks = int(np.ceil(len(hash_prints) / block_size))
        hash_groups = {}
        global_ind = 0
        for idx in range(num_of_blocks):
            for jdx in range((idx + 1 if num_of_blocks > 1 else idx), num_of_blocks):
                if for_text:
                    _count_vector = CountVectorizer(
                        binary=True, analyzer="word", token_pattern="[^ ]+", ngram_range=(2, 2)
                    )
                else:
                    _count_vector = CountVectorizer(binary=True, analyzer=lambda x: x)

                block_i = hash_prints[idx * block_size : (idx + 1) * block_size]
                block_j = hash_prints[jdx * block_size : (jdx + 1) * block_size] if num_of_blocks > 1 else []

                transformed_hashes = _count_vector.fit_transform(block_i + block_j).astype(np.int8)
                transformed_hashes_count_words = np.asarray(np.sum(transformed_hashes, axis=1))
                similarities = sklearn.metrics.pairwise.cosine_similarity(transformed_hashes)

                indices_looked = list(range(idx * block_size, idx * block_size + len(block_i))) + list(
                    range(jdx * block_size, jdx * block_size + len(block_j))
                )

                for seq_num_i in range(len(indices_looked)):
                    i = indices_looked[seq_num_i]
                    if i not in hash_groups:
                        hash_groups[i] = global_ind
                        global_ind += 1
                    for seq_num_j in range(seq_num_i + 1, len(indices_looked)):
                        j = indices_looked[seq_num_j]
                        if j not in hash_groups:
                            min_words_num = min(
                                transformed_hashes_count_words[seq_num_i][0],
                                transformed_hashes_count_words[seq_num_j][0],
                            )
                            recalculated_threshold = utils.calculate_threshold(min_words_num, threshold)
                            if similarities[seq_num_i][seq_num_j] >= recalculated_threshold:
                                hash_groups[j] = hash_groups[i]
        return hash_groups

    def unite_groups_by_hashes(self, messages: list[str], threshold: float = 0.95) -> dict[int, list[int]]:
        start_time = time()
        hash_prints = self.calculate_hashes(messages)
        has_no_empty = False
        for hash_print in hash_prints:
            if len(hash_print):
                has_no_empty = True
                break
        if not has_no_empty:
            return {}
        hash_groups = self.similarity_groupping(hash_prints, for_text=False, threshold=threshold)
        rearranged_groups = {}
        for key in hash_groups:
            cluster = hash_groups[key]
            if cluster not in rearranged_groups:
                rearranged_groups[cluster] = []
            rearranged_groups[cluster].append(key)
        LOGGER.debug("Time for finding hash groups: %.2f s", time() - start_time)
        return rearranged_groups

    def perform_light_deduplication(self, messages: list[str]) -> tuple[list[str], dict[int, list[int]]]:
        text_messages_set = {}
        messages_to_cluster = []
        ids_with_duplicates = {}
        new_id = 0
        for idx, text_message in enumerate(messages):
            text_message_normalized = " ".join(sorted(text_processing.split_words(text_message, to_lower=True)))
            if text_message_normalized not in text_messages_set:
                messages_to_cluster.append(text_message)
                text_messages_set[text_message_normalized] = new_id
                ids_with_duplicates[new_id] = [idx]
                new_id += 1
            else:
                ids_with_duplicates[text_messages_set[text_message_normalized]].append(idx)
        return messages_to_cluster, ids_with_duplicates

    def find_clusters(self, messages: list[str], threshold: float = 0.95) -> dict[int, list[int]]:
        messages_to_cluster, ids_with_duplicates = self.perform_light_deduplication(messages)
        hash_groups = self.unite_groups_by_hashes(messages_to_cluster, threshold=threshold)
        groups = self.find_groups_by_similarity(messages_to_cluster, hash_groups, threshold=threshold)
        new_groups = {}
        for cluster in groups:
            new_log_ids = []
            for idx in groups[cluster]:
                new_log_ids.extend(ids_with_duplicates[idx])
            new_groups[cluster] = new_log_ids
        return new_groups
