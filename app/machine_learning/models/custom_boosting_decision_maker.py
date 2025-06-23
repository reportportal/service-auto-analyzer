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

from app.commons.object_saving.object_saver import ObjectSaver
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker


class CustomBoostingDecisionMaker(BoostingDecisionMaker):

    def __init__(
        self,
        object_saver: ObjectSaver,
        *,
        features: Optional[list[int]] = None,
        monotonous_features: Optional[list[int]] = None,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            object_saver,
            "custom boosting model",
            features=features,
            monotonous_features=monotonous_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
