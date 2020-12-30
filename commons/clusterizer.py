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
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import spatial
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
            hash_print = set(heapq.nlargest(n_permutations, hash_print))
            hashes.append(hash_print)
        return hashes

    def find_groups_by_similarity(self, messages, groups_to_check, threshold=0.98):
        if len(messages) == 0:
            return {}
        global_group_map = {}
        group_id = 0
        start_time = time()
        count_vectorizer = HashingVectorizer(binary=True, analyzer="word", token_pattern="[^ ]+")
        t1 = time()
        transformed_logs = count_vectorizer.fit_transform(messages)
        print("Transformed messages ", time() - t1)
        for key_word in groups_to_check:
            t1 = time()
            groups_map_average = {}
            groups_map = {}
            for i in groups_to_check[key_word]:
                max_sim_group = 0
                max_similarity = 0
                transformed_vec = np.asarray(transformed_logs[i].toarray())[0]
                for group in groups_map_average:
                    cos_distance = round(
                        1 - spatial.distance.cosine(transformed_vec, groups_map_average[group]), 2)
                    if cos_distance > max_similarity:
                        max_sim_group = group
                        max_similarity = cos_distance

                if max_similarity >= threshold:
                    groups_map_average[max_sim_group] *= len(groups_map[max_sim_group])
                    groups_map_average[max_sim_group] += transformed_vec
                    groups_map_average[max_sim_group] /= (len(groups_map[max_sim_group]) + 1)
                    groups_map[max_sim_group].append(i)
                else:
                    groups_map[group_id] = [i]
                    groups_map_average[group_id] = transformed_vec
                    group_id += 1
            for gr_id in groups_map:
                global_group_map[gr_id] = groups_map[gr_id]
            if len(groups_to_check[key_word]) >= 100:
                print(key_word, len(groups_to_check[key_word]))
                print(time() - t1)
        logger.debug("Time for finding groups: %.2f s", time() - start_time)
        return global_group_map

    def unite_groups_by_hashes(self, messages, min_jaccard_sim=0.98):
        start_time = time()
        hash_prints = self.calculate_hashes(messages)
        print("Calculated hashes ", time() - start_time)
        hash_groups = {}
        global_ind = 0
        for i in range(len(hash_prints)):
            if i not in hash_groups:
                hash_groups[i] = global_ind
                global_ind += 1
            for j in range(i + 1, len(hash_prints)):
                if j not in hash_groups:
                    if utils.jaccard_similarity(hash_prints[i], hash_prints[j]) > min_jaccard_sim:
                        hash_groups[j] = hash_groups[i]
        rearranged_groups = {}
        for key in hash_groups:
            cluster = hash_groups[key]
            if cluster not in rearranged_groups:
                rearranged_groups[cluster] = []
            rearranged_groups[cluster].append(key)
        logger.debug("Time for finding hash groups: %.2f s", time() - start_time)
        return rearranged_groups

    def find_clusters(self, messages):
        t1 = time()
        hash_groups = self.unite_groups_by_hashes(messages)
        print("Found hash groups ", time() - t1)
        groups = self.find_groups_by_similarity(messages, hash_groups)
        return groups
