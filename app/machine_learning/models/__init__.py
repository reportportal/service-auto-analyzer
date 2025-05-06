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

"""Common package for ML models."""

from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker
from app.machine_learning.models.custom_boosting_decision_maker import CustomBoostingDecisionMaker
from app.machine_learning.models.custom_defect_type_model import CustomDefectTypeModel
from app.machine_learning.models.defect_type_model import DefectTypeModel
from app.machine_learning.models.ml_model import MlModel
from app.machine_learning.models.weighted_similarity_calculator import WeightedSimilarityCalculator

__all__ = [
    "MlModel",
    "DefectTypeModel",
    "CustomDefectTypeModel",
    "BoostingDecisionMaker",
    "CustomBoostingDecisionMaker",
    "WeightedSimilarityCalculator",
]
