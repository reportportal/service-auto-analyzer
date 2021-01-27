"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""
import logging
import hashlib
import heapq
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from utils import utils

logger = logging.getLogger("analyzerApp.clusterizer")


class Clusterizer:

    def __init__(self):
        pass

    def calculate_hashes(self, messages, n_gram=2, n_permutations=64):
        hashes = []
        for message in messages:
            words = message.split()
            hash_print = set()
            for i in range(len(words) - n_gram):
                hash_print.add(hashlib.md5(" ".join(words[i:i + n_gram]).encode("utf-8")).hexdigest())
            hash_print = list(heapq.nlargest(n_permutations, hash_print))
            hashes.append(hash_print)
        return hashes

    def find_groups_by_similarity(self, messages, groups_to_check, threshold=0.98):
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
        logger.debug("Time for finding groups: %.2f s", time() - start_time)
        return rearranged_groups

    def similarity_groupping(self, hash_prints, block_size=1000, for_text=True, threshold=0.98):
        num_of_blocks = int(np.ceil(len(hash_prints) / block_size))
        hash_groups = {}
        global_ind = 0
        for idx in range(num_of_blocks):
            for jdx in range((idx + 1 if num_of_blocks > 1 else idx), num_of_blocks):
                if for_text:
                    _count_vector = CountVectorizer(binary=True, analyzer="word", token_pattern="[^ ]+")
                else:
                    _count_vector = CountVectorizer(binary=True, analyzer=lambda x: x)

                block_i = hash_prints[idx * block_size: (idx + 1) * block_size]
                block_j = hash_prints[jdx * block_size: (jdx + 1) * block_size] if num_of_blocks > 1 else []

                transformed_hashes = _count_vector.fit_transform(block_i + block_j).astype(np.int8)
                similarities = sklearn.metrics.pairwise.cosine_similarity(transformed_hashes)

                indices_looked = list(range(idx * block_size, idx * block_size + len(block_i))) + list(
                    range(jdx * block_size, jdx * block_size + len(block_j)))

                for seq_num_i in range(len(indices_looked)):
                    i = indices_looked[seq_num_i]
                    if i not in hash_groups:
                        hash_groups[i] = global_ind
                        global_ind += 1
                    for seq_num_j in range(seq_num_i + 1, len(indices_looked)):
                        j = indices_looked[seq_num_j]
                        if j not in hash_groups:
                            if similarities[seq_num_i][seq_num_j] >= threshold:
                                hash_groups[j] = hash_groups[i]
        return hash_groups

    def unite_groups_by_hashes(self, messages, min_jaccard_sim=0.98):
        start_time = time()
        hash_prints = self.calculate_hashes(messages)
        has_no_empty = False
        for hash_print in hash_prints:
            if len(hash_print):
                has_no_empty = True
                break
        if not has_no_empty:
            return {}
        hash_groups = self.similarity_groupping(hash_prints, for_text=False, threshold=min_jaccard_sim)
        rearranged_groups = {}
        for key in hash_groups:
            cluster = hash_groups[key]
            if cluster not in rearranged_groups:
                rearranged_groups[cluster] = []
            rearranged_groups[cluster].append(key)
        logger.debug("Time for finding hash groups: %.2f s", time() - start_time)
        return rearranged_groups

    def perform_light_deduplication(self, messages):
        text_messages_set = {}
        messages_to_cluster = []
        ids_with_duplicates = {}
        new_id = 0
        for idx, text_message in enumerate(messages):
            text_message_normalized = " ".join(sorted(
                utils.split_words(text_message, to_lower=True)))
            if text_message_normalized not in text_messages_set:
                messages_to_cluster.append(text_message)
                text_messages_set[text_message_normalized] = new_id
                ids_with_duplicates[new_id] = [idx]
                new_id += 1
            else:
                ids_with_duplicates[text_messages_set[text_message_normalized]].append(idx)
        return messages_to_cluster, ids_with_duplicates

    def find_clusters(self, messages):
        messages_to_cluster, ids_with_duplicates = self.perform_light_deduplication(messages)
        hash_groups = self.unite_groups_by_hashes(messages_to_cluster)
        groups = self.find_groups_by_similarity(messages_to_cluster, hash_groups)
        new_groups = {}
        for cluster in groups:
            new_log_ids = []
            for idx in groups[cluster]:
                new_log_ids.extend(ids_with_duplicates[idx])
            new_groups[cluster] = new_log_ids
        return new_groups
