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
from typing import Any

from app.boosting_decision_making.defect_type_model import DefectTypeModel
from app.commons.object_saving.object_saver import ObjectSaver


class CustomDefectTypeModel(DefectTypeModel):
    project_id: int | str

    def __init__(self, folder: str, app_config: dict[str, Any], project_id: int | str):
        super().__init__(folder, tags='custom boosting model')
        self.project_id = project_id
        self.object_saver = ObjectSaver(app_config)

    def load_model(self):
        self.count_vectorizer_models = self.object_saver.get_project_object(
            os.path.join(self.folder, "count_vectorizer_models"), self.project_id, using_json=False)
        assert len(self.count_vectorizer_models) > 0
        self.models = self.object_saver.get_project_object(os.path.join(self.folder, "models"), self.project_id,
                                                           using_json=False)
        assert len(self.models) > 0

    def save_model(self):
        self.object_saver.put_project_object(self.count_vectorizer_models,
                                             os.path.join(self.folder, "count_vectorizer_models"), self.project_id,
                                             using_json=False)
        self.object_saver.put_project_object(self.models, os.path.join(self.folder, "models"), self.project_id,
                                             using_json=False)
