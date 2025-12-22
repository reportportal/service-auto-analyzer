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

from typing import Optional

from typing_extensions import override

from app.commons.object_saving.object_saver import ObjectSaver
from app.ml.models.defect_type_model import DefectTypeModel

MODEL_TAG = "custom defect type model"


class CustomDefectTypeModel(DefectTypeModel):

    def __init__(self, object_saver: ObjectSaver, n_estimators: Optional[int] = None):
        super().__init__(object_saver, MODEL_TAG, n_estimators=n_estimators)

    @property
    @override
    def is_custom(self) -> bool:
        return True
