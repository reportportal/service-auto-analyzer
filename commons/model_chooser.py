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
from boosting_decision_making import custom_boosting_decision_maker, boosting_decision_maker
from commons.object_saving.object_saver import ObjectSaver
import logging
import numpy as np
import os

logger = logging.getLogger("analyzerApp.modelChooser")


class ModelChooser:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.object_saver = ObjectSaver(self.app_config)
        self.model_folder_mapping = {
            "defect_type_model/": custom_defect_type_model.CustomDefectTypeModel,
            "suggestion_model/": custom_boosting_decision_maker.CustomBoostingDecisionMaker,
            "auto_analysis_model/": custom_boosting_decision_maker.CustomBoostingDecisionMaker
        }
        self.initialize_global_models()

    def initialize_global_models(self):
        self.global_models = {}
        for model_name, folder, class_to_use in [
            ("defect_type_model/",
             self.search_cfg["GlobalDefectTypeModelFolder"], defect_type_model.DefectTypeModel),
            ("suggestion_model/",
             self.search_cfg["SuggestBoostModelFolder"], boosting_decision_maker.BoostingDecisionMaker),
            ("auto_analysis_model/",
             self.search_cfg["BoostModelFolder"], boosting_decision_maker.BoostingDecisionMaker)]:
            if folder.strip():
                self.global_models[model_name] = class_to_use(folder=folder)
            else:
                self.global_models[model_name] = None

    def choose_model(self, project_id, model_name_folder, custom_model_prob=1.0):
        model = self.global_models[model_name_folder]
        prob_for_model = np.random.uniform()
        if prob_for_model > custom_model_prob:
            return model
        folders = self.object_saver.get_folder_objects(project_id, model_name_folder)
        if len(folders):
            try:
                model = self.model_folder_mapping[model_name_folder](
                    self.app_config, project_id, folder=folders[0])
            except Exception as err:
                logger.error(err)
        return model

    def delete_old_model(self, model_name, project_id):
        all_folders = self.object_saver.get_folder_objects(
            project_id, "%s/" % model_name)
        deleted_models = 0
        for folder in all_folders:
            if os.path.basename(
                    folder.strip("/").strip("\\")).startswith(model_name):
                deleted_models += self.object_saver.remove_folder_objects(project_id, folder)
        return deleted_models

    def delete_all_custom_models(self, project_id):
        for model_name_folder in self.model_folder_mapping:
            self.delete_old_model(model_name_folder.strip("/").strip("\\"), project_id)

    def get_model_info(self, model_name, project_id):
        all_folders = self.object_saver.get_folder_objects(
            project_id, "%s/" % model_name)
        return all_folders[0] if len(all_folders) else ""
