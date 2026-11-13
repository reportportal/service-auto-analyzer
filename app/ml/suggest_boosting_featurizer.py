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

from typing import Any, Optional, override

from app.commons.model.db import Hit
from app.commons.model.test_item_index import TestItemIndexData
from app.ml.boosting_featurizer import BoostingFeaturizer
from app.ml.models.defect_type_model import DefectTypeModel


class SuggestBoostingFeaturizer(BoostingFeaturizer):
    """Gather Gradient Boosting features by found Test Items: every found Test Item gets its own row."""

    def __init__(
        self,
        results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
        config: dict[str, Any],
        feature_ids: str | list[int],
        defect_type_model: Optional[DefectTypeModel] = None,
        **_: Any,
    ) -> None:
        super().__init__(results, config, feature_ids, defect_type_model)

    @override
    def _get_identity(self, hit: Hit[TestItemIndexData]) -> str:
        """Get the identity a found Test Item represents in the output.

        :param hit: Found Test Item
        :return: ID of the found Test Item
        """
        return str(hit.source.test_item_id)
