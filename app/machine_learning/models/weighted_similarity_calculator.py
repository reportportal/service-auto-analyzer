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
from typing import Any

import numpy as np

from app.machine_learning.models import MlModel
from app.utils import text_processing

MODEL_FILES: list[str] = ['weights.pickle']


class WeightedSimilarityCalculator(MlModel):

    def __init__(self, folder, block_to_split=10, min_log_number_in_block=1):
        super().__init__(folder, 'global similarity model')
        self.block_to_split = block_to_split
        self.min_log_number_in_block = min_log_number_in_block
        weights = self.load_model()
        self.block_to_split, self.min_log_number_in_block, self.weights, self.softmax_weights = tuple(*weights)

    def load_model(self) -> list[Any]:
        return self.load_models(MODEL_FILES)

    def save_model(self):
        self.save_models(zip(
            MODEL_FILES,
            [[self.block_to_split, self.min_log_number_in_block, self.weights, self.softmax_weights]]))

    def message_to_array(self, detected_message_res, stacktrace_res):
        all_lines = [" ".join(text_processing.split_words(detected_message_res))]
        split_log_lines = text_processing.filter_empty_lines(
            [" ".join(text_processing.split_words(line)) for line in stacktrace_res.split("\n")])
        split_log_lines_num = len(split_log_lines)
        data_in_block = max(self.min_log_number_in_block,
                            math.ceil(split_log_lines_num / self.block_to_split))
        blocks_num = math.ceil(split_log_lines_num / data_in_block)

        for block in range(blocks_num):
            all_lines.append("\n".join(
                split_log_lines[block * data_in_block: (block + 1) * data_in_block]))
        if len([line for line in all_lines if line.strip()]) == 0:
            return []
        return all_lines

    def weigh_data_rows(self, data_rows, use_softmax=False):
        padded_data_rows = np.concatenate([data_rows,
                                           np.zeros((max(0, self.block_to_split + 1 - len(data_rows)),
                                                     data_rows.shape[1]))], axis=0)
        result = None
        if use_softmax:
            result = np.dot(np.reshape(self.softmax_weights, [-1]), padded_data_rows)
        else:
            result = np.dot(np.reshape(self.weights, [-1]), padded_data_rows)
        return np.clip(result, a_min=0, a_max=1)
