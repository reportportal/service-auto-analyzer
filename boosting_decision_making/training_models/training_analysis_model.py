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
from boosting_decision_making.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from boosting_decision_making import weighted_similarity_calculator
from sklearn.model_selection import train_test_split
import elasticsearch
import elasticsearch.helpers
from commons import model_chooser
from commons.esclient import EsClient
from commons import namespace_finder
from imblearn.over_sampling import SMOTE
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
        self.due_proportion_to_smote = 0.4
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)
        self.baseline_folders = {
            "suggestion": self.search_cfg["SuggestBoostModelFolder"],
            "auto_analysis": self.search_cfg["BoostModelFolder"]}
        self.model_config = {
            "suggestion": self.search_cfg["RetrainSuggestBoostModelConfig"],
            "auto_analysis": self.search_cfg["RetrainAutoBoostModelConfig"]}
        self.weighted_log_similarity_calculator = None
        if self.search_cfg["SimilarityWeightsFolder"].strip():
            self.weighted_log_similarity_calculator = weighted_similarity_calculator.\
                WeightedSimilarityCalculator(folder=self.search_cfg["SimilarityWeightsFolder"])
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_chooser = model_chooser.ModelChooser(app_config=app_config, search_cfg=search_cfg)

    def get_config_for_boosting(self, numberOfLogLines, boosting_model_name, namespaces):
        return {
            "max_query_terms": self.search_cfg["MaxQueryTerms"],
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg["MinWordLength"],
            "filter_min_should_match": [],
            "filter_min_should_match_any": [],
            "number_of_log_lines": numberOfLogLines,
            "filter_by_unique_id": False,
            "boosting_model": self.baseline_folders[boosting_model_name],
            "chosen_namespaces": namespaces,
            "calculate_similarities": False}

    def get_info_template(self, project_info, baseline_model, model_name):
        return {"method": "training", "sub_model_type": "all", "model_type": project_info["model_type"],
                "baseline_model": [baseline_model], "new_model": [model_name],
                "project_id": str(project_info["project_id"]), "model_saved": 0, "p_value": 1.0,
                "data_proportion": 0.0, "baseline_mean_metric": 0.0, "new_model_mean_metric": 0.0,
                "bad_data_proportion": 0, "metric_name": "F1"}

    def train_several_times(self, data, labels, features):
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
                proportion_binary_labels = utils.calculate_proportions_for_labels(y_train)
                if proportion_binary_labels < self.due_proportion_to_smote:
                    oversample = SMOTE(ratio="minority")
                    x_train, y_train = oversample.fit_sample(x_train, y_train)
                self.new_model.train_model(x_train, y_train)
                logger.debug("New model results")
                f1 = self.new_model.validate_model(x_test, y_test)
                new_model_results.append(f1)
                logger.debug("Baseline results")
                x_test_for_baseline = self.transform_data_from_feature_lists(
                    x_test, features, self.baseline_model.get_feature_ids())
                f1 = self.baseline_model.validate_model(x_test_for_baseline, y_test)
                baseline_model_results.append(f1)
        return baseline_model_results, new_model_results, bad_data

    def transform_data_from_feature_lists(self, feature_list, cur_features, desired_features):
        previously_gathered_features = utils.fill_prevously_gathered_features(feature_list, cur_features)
        gathered_data = utils.gather_feature_list(previously_gathered_features, desired_features)
        return gathered_data

    def query_es_for_suggest_info(self, project_id):
        search_query = {
            "sort": {"savedDate": "desc"},
            "size": 10000,
        }
        log_ids_to_find = set()
        gathered_suggested_data = []
        for res in elasticsearch.helpers.scan(self.es_client.es_client,
                                              query=search_query,
                                              index=str(project_id) + "_suggest",
                                              scroll="5m"):
            if len(gathered_suggested_data) >= 30000:
                break
            for col in ["testItemLogId", "relevantLogId"]:
                log_id = res["_source"][col]
                if res["_source"]["isMergedLog"]:
                    log_id = log_id + "_m"
                log_ids_to_find.add(log_id)
            gathered_suggested_data.append(res)
        log_ids_to_find = list(log_ids_to_find)
        batch_size = 1000
        log_id_dict = {}
        for i in range(int(len(log_ids_to_find) / batch_size) + 1):
            log_ids = log_ids_to_find[i * batch_size: (i + 1) * batch_size]
            ids_query = {
                "size": 10000,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"_id": [str(_id) for _id in log_ids]}}
                        ]
                    }
                }}
            if not log_ids:
                continue
            for r in elasticsearch.helpers.scan(self.es_client.es_client,
                                                query=ids_query,
                                                index=project_id,
                                                scroll="5m"):
                log_id_dict[r["_id"]] = r
        return gathered_suggested_data, log_id_dict

    def gather_data(self, model_type, project_id, features, defect_type_model_to_use):
        namespaces = self.namespace_finder.get_chosen_namespaces(project_id)
        gathered_suggested_data, log_id_dict = self.query_es_for_suggest_info(project_id)
        full_data_features, labels = [], []
        for _suggest_res in gathered_suggested_data:
            searched_res = []
            found_logs = {}
            for col in ["testItemLogId", "relevantLogId"]:
                log_id = str(_suggest_res["_source"][col])
                if _suggest_res["_source"]["isMergedLog"]:
                    log_id = log_id + "_m"
                if log_id in log_id_dict:
                    found_logs[col] = log_id_dict[log_id]
            if len(found_logs) == 2:
                log_relevent = found_logs["relevantLogId"]
                log_relevent["_score"] = _suggest_res["_source"]["esScore"]
                searched_res = [
                    (found_logs["testItemLogId"], {"hits": {"hits": [log_relevent]}})]
            if searched_res:
                _boosting_data_gatherer = SuggestBoostingFeaturizer(
                    searched_res,
                    self.get_config_for_boosting(
                        _suggest_res["_source"]["usedLogLines"], model_type, namespaces),
                    feature_ids=features,
                    weighted_log_similarity_calculator=self.weighted_log_similarity_calculator)
                _boosting_data_gatherer.set_defect_type_model(defect_type_model_to_use)
                _boosting_data_gatherer.fill_prevously_gathered_features(
                    [utils.to_number_list(_suggest_res["_source"]["modelFeatureValues"])],
                    _suggest_res["_source"]["modelFeatureNames"])
                feature_data, _ = _boosting_data_gatherer.gather_features_info()
                if feature_data:
                    full_data_features.extend(feature_data)
                    labels.append(int(_suggest_res["_source"]["isUserChoice"]))
        return np.asarray(full_data_features), np.asarray(labels)

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

        defect_type_model_to_use = self.model_chooser.choose_model(
            project_info["project_id"], "defect_type_model/")

        train_log_info = self.get_info_template(project_info, baseline_model_folder, model_name)
        logger.debug("Initialized training model '%s'", project_info["model_type"])
        train_data, labels = self.gather_data(
            project_info["model_type"], project_info["project_id"],
            self.new_model.get_feature_ids(), defect_type_model_to_use)
        train_log_info["data_size"] = len(labels)
        _, features, _ = pickle.load(open(os.path.join(
            self.baseline_folders[project_info["model_type"]], "data_features_config.pickle"), "rb"))

        logger.debug("Loaded data for training model '%s'", project_info["model_type"])

        train_log_info["data_proportion"] = utils.calculate_proportions_for_labels(labels)
        baseline_model_results, new_model_results, bad_data = self.train_several_times(
            train_data, labels, self.new_model.get_feature_ids())

        use_custom_model = False
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
            if proportion_binary_labels < self.due_proportion_to_smote:
                oversample = SMOTE(ratio="minority")
                train_data, labels = oversample.fit_sample(train_data, labels)
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
        return len(train_data), {"all": train_log_info}
