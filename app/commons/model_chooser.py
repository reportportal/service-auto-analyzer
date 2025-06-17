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
from typing import Any, Optional

import numpy as np

from app.commons import logging, object_saving
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.ml import ModelType
from app.commons.object_saving.object_saver import ObjectSaver
from app.machine_learning.models import (
    MlModel,
    boosting_decision_maker,
    custom_boosting_decision_maker,
    custom_defect_type_model,
    defect_type_model,
)

logger = logging.getLogger("analyzerApp.modelChooser")

DEFAULT_RANDOM_SEED = 1337

CUSTOM_MODEL_MAPPING = {
    ModelType.defect_type: custom_defect_type_model.CustomDefectTypeModel,
    ModelType.suggestion: custom_boosting_decision_maker.CustomBoostingDecisionMaker,
    ModelType.auto_analysis: custom_boosting_decision_maker.CustomBoostingDecisionMaker,
}

GLOBAL_MODEL_MAPPING = {
    ModelType.defect_type: defect_type_model.DefectTypeModel,
    ModelType.suggestion: boosting_decision_maker.BoostingDecisionMaker,
    ModelType.auto_analysis: boosting_decision_maker.BoostingDecisionMaker,
}


class ModelChooser:
    app_config: ApplicationConfig
    object_saver: ObjectSaver
    search_cfg: SearchConfig
    global_models: dict[ModelType, MlModel]
    random_generator: np.random.Generator

    def __init__(self, app_config: ApplicationConfig, search_cfg: SearchConfig):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.object_saver = object_saving.create(self.app_config)
        self.global_models = self.initialize_global_models()

    def initialize_global_models(self) -> dict[ModelType, MlModel]:
        result = {}
        for model_type, folder in zip(
            [ModelType.defect_type, ModelType.suggestion, ModelType.auto_analysis],
            [
                self.search_cfg.GlobalDefectTypeModelFolder,
                self.search_cfg.SuggestBoostModelFolder,
                self.search_cfg.BoostModelFolder,
            ],
        ):
            if folder.strip():
                model = GLOBAL_MODEL_MAPPING[model_type](object_saving.create_filesystem(folder))
                model.load_model()
                result[model_type] = model
            else:
                result[model_type] = None
        return result

    def choose_model(
        self,
        project_id: int,
        model_type: ModelType,
        *,
        custom_model_prob: float = 1.0,
        hash_source: Optional[Any] = None,
    ) -> MlModel:
        hash_code = hash(hash_source) if hash_source else hash(project_id)
        test_value = hash_code % 100
        model = self.global_models[model_type]
        if test_value > custom_model_prob:  # use hash instead of random to ensure in repeatable results
            logger.debug(
                f"Using global model of type '{model_type.name}', for project {project_id}, "
                f"for hash source {str(hash_source)}."
            )
            return model
        logger.debug(
            f"Using custom model of type '{model_type.name}', for project {project_id}, "
            f"for hash source {str(hash_source)}."
        )
        folders = self.object_saver.get_folder_objects(f"{model_type.name}_model/", project_id)
        if len(folders):
            try:
                model = CUSTOM_MODEL_MAPPING[model_type](object_saving.create(self.app_config, project_id, folders[0]))
                model.load_model()
            except Exception as err:
                logger.exception(err)
        return model

    def delete_old_model(self, model_type: ModelType, project_id: str | int | None = None):
        all_folders = self.object_saver.get_folder_objects(f"{model_type.name}_model/", project_id)
        deleted_models = 0
        for folder in all_folders:
            if os.path.basename(folder.strip("/").strip("\\")).startswith(model_type.name):
                deleted_models += int(self.object_saver.remove_folder_objects(folder, project_id))
        return deleted_models

    def delete_all_custom_models(self, project_id: str | int | None = None):
        for model in CUSTOM_MODEL_MAPPING.keys():
            self.delete_old_model(model, project_id)

    def get_model_info(self, model_type: ModelType, project_id: str | int | None = None):
        all_folders = self.object_saver.get_folder_objects(f"{model_type.name}_model/", project_id)
        return all_folders[0] if len(all_folders) else ""
