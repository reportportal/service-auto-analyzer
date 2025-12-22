#  Copyright 2024 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pytest

from app.commons import object_saving
from app.ml.models.defect_type_model import DefectTypeModel
from app.utils import utils

WEB_DRIVER_ERROR = (
    "org.openqa.selenium.TimeoutException: Expected condition failed: waiting for visibility of "
    "element located by By.xpath: "
    "//*[contains(@class,'nav-bar_menu-items') and contains(text(),'Blog')] "
    "(tried for 20 second(s) with 500 milliseconds interval)"
)


@pytest.fixture(scope="session")
def object_saver() -> object_saving.ObjectSaver:
    model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
    return object_saving.create_filesystem(model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"])


@pytest.fixture(scope="session")
def defect_type_model(object_saver: object_saving.ObjectSaver) -> DefectTypeModel:
    model = DefectTypeModel(object_saver)
    model.load_model()
    return model


@pytest.mark.parametrize(
    "defect_type, expected",
    [
        ("nd001", 0.0),
        ("pb001", 0.0),
        ("ab001", 1.0),
        ("si001", 0.0),
        ("pd001", 0.0),
        ("ab_abracadabra", 1.0),
        ("pb_abracadabra", 0.0),
        ("si_abracadabra", 0.0),
    ],
)
def test_different_defect_type_predict(defect_type_model: DefectTypeModel, defect_type: str, expected: float) -> None:
    result = defect_type_model.predict([WEB_DRIVER_ERROR], defect_type)
    assert result[0][0] == expected, f"Invalid result for defect type: {defect_type}"


@pytest.mark.parametrize("defect_type", ["ndabc", "asdfcas", "pb00a", "nd_abc abc", "\n_asdcas", " _aab"])
def test_invalid_type_error(defect_type_model: DefectTypeModel, defect_type: str) -> None:
    with pytest.raises(KeyError):
        defect_type_model.predict([WEB_DRIVER_ERROR], defect_type)


def test_load_model_again_not_loads_model(defect_type_model: DefectTypeModel):
    models = defect_type_model.models
    count_vectorizer_models = defect_type_model.count_vectorizer_models
    defect_type_model.load_model()
    assert models is defect_type_model.models
    assert count_vectorizer_models is defect_type_model.count_vectorizer_models
