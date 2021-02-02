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

from boosting_decision_making import boosting_decision_maker, custom_boosting_decision_maker
from sklearn.model_selection import train_test_split
from commons.esclient import EsClient
from utils import utils
from time import time
import scipy.stats as stats
import numpy as np
import logging
from datetime import datetime
import os
import pickle

logger = logging.getLogger("analyzerApp.trainingAnalysisModel")


class AnalysisModelTraining:

    def __init__(self, app_config, search_cfg):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.due_proportion = 0.2
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)
        self.baseline_folders = {
            "suggestion": self.search_cfg["SuggestBoostModelFolder"],
            "auto_analysis": self.search_cfg["BoostModelFolder"]}
        self.model_config = {
            "suggestion": self.search_cfg["RetrainSuggestBoostModelConfig"],
            "auto_analysis": self.search_cfg["RetrainAutoBoostModelConfig"]}

    def get_info_template(self, project_info, baseline_model, model_name):
        return {"method": "training", "sub_model_type": "all", "model_type": project_info["model_type"],
                "baseline_model": [baseline_model], "new_model": [model_name],
                "project_id": str(project_info["project_id"]), "model_saved": 0, "p_value": 1.0,
                "data_proportion": 0.0, "baseline_mean_metric": 0.0, "new_model_mean_metric": 0.0,
                "bad_data_proportion": 0, "metric_name": "F1"}

    def train_several_times(self, data, labels):
        new_model_results = []
        baseline_model_results = []
        random_states = [1257, 1873, 1917]
        bad_data = False

        proportion_binary_labels = utils.calculate_proportions_for_labels(labels)

        if proportion_binary_labels < self.due_proportion:
            logger.debug("Train data has a bad proportion: %.3f", proportion_binary_labels)
            bad_data = True

        if not bad_data:
            for random_state in random_states:
                x_train, x_test, y_train, y_test = train_test_split(
                    data, labels,
                    test_size=0.1, random_state=random_state, stratify=labels)
                self.new_model.train_model(x_train, y_train)
                logger.debug("New model results")
                f1 = self.new_model.validate_model(x_test, y_test)
                new_model_results.append(f1)
                logger.debug("Baseline results")
                f1 = self.baseline_model.validate_model(x_test, y_test)
                baseline_model_results.append(f1)
        return baseline_model_results, new_model_results, bad_data

    def train(self, project_info):
        time_training = time()
        logger.debug("Started training model '%s'", project_info["model_type"])
        model_name = "%s_model_%s" % (project_info["model_type"], datetime.now().strftime("%d.%m.%y"))

        baseline_model_folder = os.path.basename(
            self.baseline_folders[project_info["model_type"]].strip("/").strip("\\"))
        self.baseline_model = boosting_decision_maker.BoostingDecisionMaker(
            folder=self.baseline_folders[project_info["model_type"]])
        full_config, features, monotonous_features = pickle.load(
            open(self.model_config[project_info["model_type"]], "rb"))
        self.new_model = custom_boosting_decision_maker.CustomBoostingDecisionMaker(
            self.app_config, project_info["project_id"])
        self.new_model.add_config_info(full_config, features, monotonous_features)

        train_log_info = self.get_info_template(project_info, baseline_model_folder, model_name)
        logger.debug("Initialized training model '%s'", project_info["model_type"])
        data = pickle.load(open("model/train_data.pickle", "rb"))
        train_data, labels = data[3], data[4]
        train_log_info["data_size"] = len(labels)
        _, features, _ = pickle.load(open(os.path.join(
            self.baseline_folders[project_info["model_type"]], "data_features_config.pickle"), "rb"))

        logger.debug("Loaded data for training model '%s'", project_info["model_type"])

        train_log_info["data_proportion"] = utils.calculate_proportions_for_labels(labels)
        baseline_model_results, new_model_results, bad_data = self.train_several_times(train_data, labels)

        use_custom_model = True
        if not bad_data:
            logger.debug("Baseline test results %s", baseline_model_results)
            logger.debug("New model test results %s", new_model_results)
            pvalue = stats.f_oneway(baseline_model_results, new_model_results).pvalue
            if pvalue != pvalue:
                pvalue = 1.0
            train_log_info["p_value"] = pvalue
            mean_f1 = np.mean(new_model_results)
            train_log_info["baseline_mean_metric"] = np.mean(baseline_model_results)
            train_log_info["new_model_mean_metric"] = mean_f1
            if pvalue < 0.05 and mean_f1 > np.mean(baseline_model_results) and mean_f1 >= 0.4:
                use_custom_model = True
            logger.debug(
                "Model training validation results: p-value=%.3f mean baseline=%.3f mean new model=%.3f",
                pvalue, np.mean(baseline_model_results), np.mean(new_model_results))
        train_log_info["bad_data_proportion"] = int(bad_data)

        if use_custom_model:
            logger.debug("Custom model should be saved")

            proportion_binary_labels = utils.calculate_proportions_for_labels(labels)
            if proportion_binary_labels < self.due_proportion:
                logger.debug("Train data has a bad proportion: %.3f", proportion_binary_labels)
                bad_data = True
            train_log_info["bad_data_proportion"] = int(bad_data)
            if not bad_data:
                train_log_info["model_saved"] = 1
                self.new_model.train_model(train_data, labels)
            else:
                train_log_info["model_saved"] = 0
            self.new_model.delete_old_model("%s_model" % project_info["model_type"])
            self.new_model.save_model(
                "%s_model/%s/" % (project_info["model_type"], model_name))
        train_log_info["time_spent"] = (time() - time_training)
        logger.info("Finished for %d s", train_log_info["time_spent"])

        train_log_info["gather_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_log_info["module_version"] = [self.app_config["appVersion"]]
        return len(data), {"all": train_log_info}
