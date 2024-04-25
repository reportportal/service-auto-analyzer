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

import os
import re
from datetime import datetime
from queue import Queue
from time import time

import elasticsearch.helpers
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split

from app.commons import logging, object_saving
from app.commons.esclient import EsClient
from app.commons.launch_objects import SearchConfig, ApplicationConfig
from app.commons.model_chooser import ModelChooser
from app.machine_learning.models import defect_type_model, custom_defect_type_model
from app.utils import utils, text_processing

logger = logging.getLogger('analyzerApp.trainingDefectTypeModel')


class DefectTypeModelTraining:
    app_config: ApplicationConfig
    search_cfg: SearchConfig

    def __init__(self, model_chooser: ModelChooser, app_config: ApplicationConfig, search_cfg: SearchConfig):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.label2inds = {"ab": 0, "pb": 1, "si": 2}
        self.due_proportion = 0.2
        self.es_client = EsClient(app_config=app_config)
        self.baseline_model = defect_type_model.DefectTypeModel(
            object_saving.create_filesystem(search_cfg.GlobalDefectTypeModelFolder))
        self.baseline_model.load_model()
        self.model_chooser = model_chooser

    def return_similar_objects_into_sample(self, x_train_ind, y_train, data, additional_logs, label):
        x_train = []
        x_train_add = []
        y_train_add = []

        for idx, ind in enumerate(x_train_ind):
            x_train.append(data[ind][0])
            label_to_use = y_train[idx]
            if ind in additional_logs and label_to_use != 1:
                for idx_ in additional_logs[ind]:
                    log_res, label_res, real_label = data[idx_]
                    if label_res == label:
                        label_to_use = 1
                        break
            if ind in additional_logs:
                for idx_ in additional_logs[ind]:
                    x_train_add.append(data[idx_][0])
                    y_train_add.append(label_to_use)
        x_train.extend(x_train_add)
        y_train.extend(y_train_add)
        return x_train, y_train

    def split_train_test(
            self, logs_to_train_idx, data, labels_filtered,
            additional_logs, label, random_state=1257):
        x_train_ind, x_test_ind, y_train, y_test = train_test_split(
            logs_to_train_idx, labels_filtered,
            test_size=0.1, random_state=random_state, stratify=labels_filtered)
        x_train, y_train = self.return_similar_objects_into_sample(
            x_train_ind, y_train, data, additional_logs, label)
        x_test = []
        for ind in x_test_ind:
            x_test.append(data[ind][0])
        return x_train, x_test, y_train, y_test

    def get_message_query_by_label(self, label):
        return {
            "_source": ["detected_message_without_params_extended", "issue_type", "launch_id"],
            "sort": {"start_time": "desc"},
            "size": self.app_config["esChunkNumber"],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}}
                    ],
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"wildcard": {"issue_type": "{}*".format(label.upper())}},
                                    {"wildcard": {"issue_type": "{}*".format(label.lower())}},
                                    {"wildcard": {"issue_type": "{}*".format(label)}},
                                ]
                            }
                        }
                    ],
                    "should": [
                        {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                    ]
                }
            }
        }

    def query_data(self, project, label):
        message_launch_dict = set()
        project_index_name = text_processing.unite_project_name(
            str(project), self.app_config["esProjectIndexPrefix"])
        data = []
        for r in elasticsearch.helpers.scan(self.es_client.es_client,
                                            query=self.get_message_query_by_label(
                                                label),
                                            index=project_index_name):
            detected_message = r["_source"]["detected_message_without_params_extended"]
            text_message_normalized = " ".join(sorted(
                text_processing.split_words(detected_message, to_lower=True)))
            message_info = (text_message_normalized,
                            r["_source"]["launch_id"],
                            r["_source"]["issue_type"])
            if message_info not in message_launch_dict:
                data.append((detected_message, label, r["_source"]["issue_type"]))
                message_launch_dict.add(message_info)
            if len(data) >= self.search_cfg.MaxLogsForDefectTypeModel:
                break
        return data

    def perform_light_deduplication(self, data):
        text_messages_set = {}
        logs_to_train_idx = []
        additional_logs = {}
        for idx, text_message_data in enumerate(data):
            text_message = text_message_data[0]
            text_message_normalized = " ".join(sorted(
                text_processing.split_words(text_message, to_lower=True)))
            if text_message_normalized not in text_messages_set:
                logs_to_train_idx.append(idx)
                text_messages_set[text_message_normalized] = idx
                additional_logs[idx] = []
            else:
                additional_logs[text_messages_set[text_message_normalized]].append(idx)
        return additional_logs, logs_to_train_idx

    def get_info_template(self, project_info, label, baseline_model, model_name):
        return {"method": "training", "sub_model_type": label, "model_type": project_info["model_type"],
                "baseline_model": [baseline_model], "new_model": [model_name],
                "project_id": project_info["project_id"], "model_saved": 0, "p_value": 1.0,
                "data_proportion": 0.0, "baseline_mean_metric": 0.0, "new_model_mean_metric": 0.0,
                "bad_data_proportion": 0, "metric_name": "F1", "errors": [], "errors_count": 0,
                "time_spent": 0.0}

    def load_data_for_training(self, project_info, baseline_model, model_name):
        train_log_info = {}
        data = []
        found_sub_categories = {}
        labels_to_find_queue = Queue()
        errors = []
        errors_count = 0

        for label in self.label2inds:
            labels_to_find_queue.put(label)
        while not labels_to_find_queue.empty():
            try:
                label = labels_to_find_queue.get()
                train_log_info[label] = self.get_info_template(
                    project_info, label, baseline_model, model_name)
                time_querying = time()
                logger.debug("Label to gather data %s", label)
                found_data = self.query_data(project_info["project_id"], label)
                for _, _, _issue_type in found_data:
                    if re.search(r"\w{2}_\w+", _issue_type) and _issue_type not in found_sub_categories:
                        found_sub_categories[_issue_type] = []
                        labels_to_find_queue.put(_issue_type)
                if label in self.label2inds:
                    data.extend(found_data)
                else:
                    found_sub_categories[label] = found_data
                time_spent = time() - time_querying
                logger.debug("Finished quering for %d s", time_spent)
                train_log_info[label]["time_spent"] = time_spent
                train_log_info[label]["data_size"] = len(found_data)
            except Exception as exc:
                logger.exception(exc)
                errors.append(utils.extract_exception(exc))
                errors_count += 1
            labels_to_find_queue.task_done()
        logger.debug("Data gathered: %d" % len(data))
        train_log_info["all"] = self.get_info_template(
            project_info, "all", baseline_model, model_name)
        train_log_info["all"]["data_size"] = len(data)
        train_log_info["all"]["errors"] = errors
        train_log_info["all"]["errors_count"] = errors_count
        return data, found_sub_categories, train_log_info

    def creating_binary_target_data(self, label, data, found_sub_categories):
        data_to_train = data
        if label in found_sub_categories:
            data_to_train = [d for d in data if d[2] != label] + found_sub_categories[label]
        additional_logs, logs_to_train_idx = self.perform_light_deduplication(data_to_train)
        labels_filtered = []
        for ind in logs_to_train_idx:
            if (data_to_train[ind][1] == label or data_to_train[ind][2] == label):
                labels_filtered.append(1)
            else:
                labels_filtered.append(0)
        proportion_binary_labels = utils.calculate_proportions_for_labels(labels_filtered)
        if proportion_binary_labels < self.due_proportion:
            logs_to_train_idx, labels_filtered, proportion_binary_labels = utils.rebalance_data(
                logs_to_train_idx, labels_filtered, self.due_proportion)
        return logs_to_train_idx, labels_filtered, data_to_train, additional_logs, proportion_binary_labels

    def copy_model_part_from_baseline(self, label):
        if label not in self.baseline_model.models:
            if label in self.new_model.models:
                del self.new_model.models[label]
            if label in self.new_model.count_vectorizer_models:
                del self.new_model.count_vectorizer_models[label]
        else:
            self.new_model.models[label] = self.baseline_model.models[label]
            _count_vectorizer = self.baseline_model.count_vectorizer_models[label]
            self.new_model.count_vectorizer_models[label] = _count_vectorizer

    def train_several_times(self, label, data, found_sub_categories):
        new_model_results = []
        baseline_model_results = []
        random_states = [1257, 1873, 1917, 2477, 3449,
                         353, 4561, 5417, 6427, 2029]
        bad_data = False

        logs_to_train_idx, labels_filtered, data_to_train, additional_logs, proportion_binary_labels = \
            self.creating_binary_target_data(label, data, found_sub_categories)

        if proportion_binary_labels < self.due_proportion:
            logger.debug("Train data has a bad proportion: %.3f", proportion_binary_labels)
            bad_data = True

        if not bad_data:
            for random_state in random_states:
                x_train, x_test, y_train, y_test = self.split_train_test(
                    logs_to_train_idx, data_to_train, labels_filtered,
                    additional_logs, label,
                    random_state=random_state)
                self.new_model.train_model(label, x_train, y_train)
                logger.debug("New model results")
                f1, accuracy = self.new_model.validate_model(label, x_test, y_test)
                new_model_results.append(f1)
                if label in found_sub_categories:
                    baseline_model_results.append(0.001)
                else:
                    logger.debug("Baseline results")
                    f1, accuracy = self.baseline_model.validate_model(label, x_test, y_test)
                    baseline_model_results.append(f1)
        return baseline_model_results, new_model_results, bad_data

    def train(self, project_info):
        start_time = time()
        model_name = "defect_type_model_%s" % datetime.now().strftime("%d.%m.%y")
        baseline_model = os.path.basename(self.search_cfg.GlobalDefectTypeModelFolder)
        new_model_folder = 'defect_type_model/%s/' % model_name
        self.new_model = custom_defect_type_model.CustomDefectTypeModel(
            object_saving.create(self.app_config, project_info["project_id"], new_model_folder))

        data, found_sub_categories, train_log_info = self.load_data_for_training(
            project_info, baseline_model, model_name)

        data_proportion_min = 1.0
        p_value_max = 0.0
        all_bad_data = 1
        custom_models = []
        f1_chosen_models = []
        f1_baseline_models = []
        errors = []
        errors_count = 0
        for label in list(self.label2inds.keys()) + list(found_sub_categories.keys()):
            try:
                time_training = time()
                logger.debug("Label to train the model %s", label)

                baseline_model_results, new_model_results, bad_data = self.train_several_times(
                    label, data, found_sub_categories)

                use_custom_model = False
                if not bad_data:
                    logger.debug("Baseline test results %s", baseline_model_results)
                    logger.debug("New model test results %s", new_model_results)
                    f_value, p_value = stats.f_oneway(baseline_model_results, new_model_results)
                    if p_value is None:
                        p_value = 1.0
                    train_log_info[label]["p_value"] = p_value
                    mean_f1 = np.mean(new_model_results)
                    train_log_info[label]["baseline_mean_metric"] = np.mean(baseline_model_results)
                    train_log_info[label]["new_model_mean_metric"] = mean_f1
                    if p_value < 0.05 and mean_f1 > np.mean(baseline_model_results) and mean_f1 >= 0.4:
                        p_value_max = max(p_value_max, p_value)
                        use_custom_model = True
                    all_bad_data = 0
                    logger.debug(
                        """Model training validation results:
                            p-value=%.3f mean baseline=%.3f mean new model=%.3f""",
                        p_value, np.mean(baseline_model_results), np.mean(new_model_results))
                train_log_info[label]["bad_data_proportion"] = int(bad_data)

                if use_custom_model:
                    logger.debug("Custom model '%s' should be saved" % label)

                    logs_to_train_idx, labels_filtered, data_to_train, additional_logs, proportion_binary_labels = \
                        self.creating_binary_target_data(label, data, found_sub_categories)
                    if proportion_binary_labels < self.due_proportion:
                        logger.debug("Train data has a bad proportion: %.3f", proportion_binary_labels)
                        bad_data = True
                    train_log_info[label]["bad_data_proportion"] = int(bad_data)
                    train_log_info[label]["data_proportion"] = proportion_binary_labels
                    if not bad_data:
                        x_train, y_train = self.return_similar_objects_into_sample(
                            logs_to_train_idx, labels_filtered, data_to_train, additional_logs, label)
                        train_log_info[label]["model_saved"] = 1
                        data_proportion_min = min(
                            train_log_info[label]["data_proportion"], data_proportion_min)
                        self.new_model.train_model(label, x_train, y_train)
                        custom_models.append(label)
                        if label not in found_sub_categories:
                            f1_baseline_models.append(train_log_info[label]["baseline_mean_metric"])
                            f1_chosen_models.append(train_log_info[label]["new_model_mean_metric"])
                    else:
                        train_log_info[label]["model_saved"] = 0
                else:
                    self.copy_model_part_from_baseline(label)
                    if train_log_info[label]["baseline_mean_metric"] > 0.001:
                        f1_baseline_models.append(train_log_info[label]["baseline_mean_metric"])
                        f1_chosen_models.append(train_log_info[label]["baseline_mean_metric"])
                train_log_info[label]["time_spent"] += (time() - time_training)
            except Exception as exc:
                logger.exception(exc)
                train_log_info[label]["errors_count"] += 1
                train_log_info[label]["errors"].append(utils.extract_exception(exc))
                errors.append(utils.extract_exception(exc))
                errors_count += 1
                self.copy_model_part_from_baseline(label)

        logger.debug("Custom models were for labels: %s" % custom_models)
        if len(custom_models):
            logger.debug("The custom model should be saved")
            train_log_info["all"]["model_saved"] = 1
            train_log_info["all"]["p_value"] = p_value_max
            self.model_chooser.delete_old_model("defect_type_model", project_info["project_id"])
            self.new_model.save_model()

        time_spent = time() - start_time
        logger.info("Finished for %d s", time_spent)
        train_log_info["all"]["time_spent"] = time_spent
        train_log_info["all"]["data_proportion"] = data_proportion_min
        train_log_info["all"]["errors_count"] += errors_count
        train_log_info["all"]["errors"].extend(errors)
        train_log_info["all"]["baseline_mean_metric"] = np.mean(
            f1_baseline_models) if f1_baseline_models else 0.0
        train_log_info["all"]["new_model_mean_metric"] = np.mean(
            f1_chosen_models) if f1_chosen_models else 0.0
        train_log_info["all"]["bad_data_proportion"] = all_bad_data
        for label in train_log_info:
            train_log_info[label]["gather_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            train_log_info[label]["module_version"] = [self.app_config["appVersion"]]
        return len(data), train_log_info
