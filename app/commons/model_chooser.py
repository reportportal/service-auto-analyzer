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

import enum
import os
from typing import Any

import numpy as np

from app.commons import logging
from app.commons import object_saving
from app.commons.object_saving.object_saver import ObjectSaver
from app.machine_learning.models import (defect_type_model, custom_defect_type_model, custom_boosting_decision_maker,
                                         boosting_decision_maker)

logger = logging.getLogger("analyzerApp.modelChooser")


class ModelType(str, enum.Enum):
    DEFECT_TYPE_MODEL = 'defect_type_model/'
    SUGGESTION_MODEL = 'suggestion_model/'
    AUTO_ANALYSIS_MODEL = 'auto_analysis_model/'


class ModelChooser:
    app_config: dict[str, Any]
    object_saver: ObjectSaver

    def __init__(self, app_config=None, search_cfg=None):
        self.app_config = app_config or {}
        self.search_cfg = search_cfg or {}
        self.object_saver = object_saving.create(self.app_config)
        self.model_folder_mapping = {
            ModelType.DEFECT_TYPE_MODEL: custom_defect_type_model.CustomDefectTypeModel,
            ModelType.SUGGESTION_MODEL: custom_boosting_decision_maker.CustomBoostingDecisionMaker,
            ModelType.AUTO_ANALYSIS_MODEL: custom_boosting_decision_maker.CustomBoostingDecisionMaker
        }
        self.global_models = {}
        self.initialize_global_models()

    def initialize_global_models(self):
        for model_type, folder, class_to_use in [
            (ModelType.DEFECT_TYPE_MODEL,
             self.search_cfg["GlobalDefectTypeModelFolder"], defect_type_model.DefectTypeModel),
            (ModelType.SUGGESTION_MODEL,
             self.search_cfg["SuggestBoostModelFolder"], boosting_decision_maker.BoostingDecisionMaker),
            (ModelType.AUTO_ANALYSIS_MODEL,
             self.search_cfg["BoostModelFolder"], boosting_decision_maker.BoostingDecisionMaker)
        ]:
            if folder.strip():
                model = class_to_use(object_saving.create_filesystem(folder))
                model.load_model()
                self.global_models[model_type] = model
            else:
                self.global_models[model_type] = None

    def choose_model(self, project_id: int, model_type: ModelType, custom_model_prob: float = 1.0):
        model = self.global_models[model_type]
        prob_for_model = np.random.uniform()
        if prob_for_model > custom_model_prob:
            return model
        folders = self.object_saver.get_folder_objects(model_type, project_id)
        if len(folders):
            try:
                model = self.model_folder_mapping[model_type](object_saving.create(self.app_config, project_id,
                                                                                   folders[0]))
                model.load_model()
            except Exception as err:
                logger.exception(err)
        return model

    def delete_old_model(self, model_name, project_id):
        all_folders = self.object_saver.get_folder_objects(f'{model_name}/', project_id)
        deleted_models = 0
        for folder in all_folders:
            if os.path.basename(folder.strip("/").strip("\\")).startswith(model_name):
                deleted_models += int(self.object_saver.remove_folder_objects(folder, project_id))
        return deleted_models

    def delete_all_custom_models(self, project_id):
        for model_name_folder in self.model_folder_mapping:
            self.delete_old_model(model_name_folder.strip("/").strip("\\"), project_id)

    def get_model_info(self, model_name, project_id):
        all_folders = self.object_saver.get_folder_objects(f'{model_name}/', project_id)
        return all_folders[0] if len(all_folders) else ''
