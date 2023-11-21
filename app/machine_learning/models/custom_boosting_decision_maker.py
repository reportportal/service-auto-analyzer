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

from typing import Any

from app.commons.object_saving.object_saver import ObjectSaver
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker


class CustomBoostingDecisionMaker(BoostingDecisionMaker):

    def __init__(self, folder: str, app_config: dict[str, Any], project_id: int | str):
        super().__init__(folder=folder, tags='custom boosting model', app_config=app_config,
                         object_saver=ObjectSaver(app_config, project_id))
