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

from boosting_decision_making import defect_type_model, custom_defect_type_model
from sklearn.model_selection import train_test_split
from commons.esclient import EsClient
from utils import utils
from time import time
import scipy.stats as stats
import numpy as np
import logging
from datetime import datetime
import pickle

logger = logging.getLogger("analyzerApp.trainingDefectTypeModel")


class DefectTypeModelTraining:

    def __init__(self, app_config, search_cfg):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.label2inds = {"ab": 0, "pb": 1, "si": 2}
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)
        self.baseline_model = defect_type_model.DefectTypeModel(
            folder=search_cfg["GlobalDefectTypeModelFolder"])

    def split_train_test(
            self, logs_to_train_idx, data,
            additional_logs, label, random_state=1257):
        labels_filtered = [1 if data[ind][1] == label else 0 for ind in logs_to_train_idx]
        proportion_binary_labels = utils.calculate_proportions_for_labels(labels_filtered)
        if proportion_binary_labels <= 0.1 and proportion_binary_labels > 0.001:
            logs_to_train_idx, labels_filtered, proportion_binary_labels = utils.rebalance_data(
                logs_to_train_idx, labels_filtered)
        x_train_ind, x_test_ind, y_train, y_test = train_test_split(
            logs_to_train_idx, labels_filtered,
            test_size=0.1, random_state=random_state, stratify=labels_filtered)
        x_train = []
        x_test = []
        x_train_add = []
        y_train_add = []

        for ind in x_train_ind:
            x_train.append(data[ind][0])
            if ind in additional_logs:
                for idx_ in additional_logs[ind]:
                    log_res, label_res = data[idx_]
                    x_train_add.append(log_res)
                    y_train_add.append(1 if label_res == label else 0)
        for ind in x_test_ind:
            x_test.append(data[ind][0])
        x_train.extend(x_train_add)
        y_train.extend(y_train_add)
        return x_train, x_test, y_train, y_test, proportion_binary_labels

    def query_data(self, project, label):
        label_data = self.es_client.es_client.search(
            project,
            {
                "_source": ["detected_message_without_params_and_brackets", "issue_type"],
                "sort": {"start_time": "desc"},
                "size": 10000,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "bool": {
                                    "should": [
                                        {"wildcard": {"issue_type": "{}*".format(label.upper())}},
                                        {"wildcard": {"issue_type": "{}*".format(label.lower())}},
                                    ]
                                }
                            }
                        ],
                        "should": [
                            {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                        ]
                    }
                }
            })
        data = []
        for d in label_data["hits"]["hits"]:
            text_message = utils.enrich_text_with_method_and_classes(
                d["_source"]["detected_message_without_params_and_brackets"])
            data.append((text_message, label))
        return data

    def train(self, project_info):
        start_time = time()
        self.new_model = custom_defect_type_model.CustomDefectTypeModel(
            self.app_config, project_info["project_id"])
        data = []
        for label in self.label2inds:
            logger.debug("Label to gather data %s", label)
            data.extend(self.query_data(project_info["project_id"], label))
            logger.debug("Finished quering for %d s", (time() - start_time))
        logger.debug("Data gathered: %d" % len(data))

        text_messages_set = {}
        logs_to_train_idx = []
        additional_logs = {}
        for idx, text_message_data in enumerate(data):
            text_message = text_message_data[0]
            text_message_normalized = " ".join(sorted(
                utils.split_words(text_message, to_lower=True)))
            if text_message_normalized not in text_messages_set:
                logs_to_train_idx.append(idx)
                text_messages_set[text_message_normalized] = idx
                additional_logs[idx] = []
            else:
                additional_logs[text_messages_set[text_message_normalized]].append(idx)

        custom_models = []
        for label in self.label2inds:
            logger.debug("Label to train the model %s", label)
            new_model_results = []
            baseline_model_results = []
            random_states = [1257, 1873, 1917]
            bad_data = False
            for random_state in random_states:
                x_train, x_test, y_train, y_test, proportion_binary_labels = self.split_train_test(
                    logs_to_train_idx, data, additional_logs, label, random_state=random_state)
                if proportion_binary_labels <= 0.1:
                    logger.debug("Train data has a bad proportion: %s", proportion_binary_labels)
                    bad_data = True
                    break
                self.new_model.train_model(label, x_train, y_train)
                logger.debug("New model results")
                f1, accuracy = self.new_model.validate_model(label, x_test, y_test)
                new_model_results.append(f1)
                logger.debug("Baseline results")
                f1, accuracy = self.baseline_model.validate_model(label, x_test, y_test)
                baseline_model_results.append(f1)

            if not bad_data:
                logger.debug("Baseline test results %s", baseline_model_results)
                logger.debug("New model test results %s", new_model_results)
                pvalue = stats.f_oneway(baseline_model_results, new_model_results).pvalue
                if pvalue < 0.05 and np.mean(new_model_results) > np.mean(baseline_model_results):
                    logger.info(
                        "Model should be used: p-value=%.3f mean baseline=%.3f mean new model=%.3f",
                        pvalue, np.mean(baseline_model_results), np.mean(new_model_results))
                    self.new_model.train_model(
                        label, [d[0] for d in data], [1 if d[1] == label else 0 for d in data])
                    custom_models.append(label)
                else:
                    self.new_model.models[label] = self.baseline_model.models[label]
                    _count_vectorizer = self.baseline_model.count_vectorizer_models[label]
                    self.new_model.count_vectorizer_models[label] = _count_vectorizer

        logger.debug("Custom models were for labels: %s" % custom_models)
        if len(custom_models):
            logger.debug("The custom model should be saved")
            self.new_model.delete_old_model()
            pickle.dump(self.new_model.count_vectorizer_models, open("custom_model/cnt_models.pickle", "wb"))
            self.new_model.save_model(
                "defect_type_model/defect_type_model_%s/" % datetime.now().strftime("%d.%m.%y"))

        logger.info("Finished for %d s", (time() - start_time))
        return len(data)
