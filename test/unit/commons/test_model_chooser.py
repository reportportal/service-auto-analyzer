#  Copyright 2025 EPAM Systems
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

import tempfile
from typing import Generator

import pytest
from machine_learning.models import CustomDefectTypeModel, MlModel

import app.utils.utils as utils
from app.commons import object_saving
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.ml import ModelType
from app.commons.model_chooser import ModelChooser
from app.commons.object_saving import ObjectSaver
from app.machine_learning.models.defect_type_model import DefectTypeModel

# Test constants for deterministic project IDs
# Logic: if test_value > custom_model_prob -> use global model, else use custom model
# For 0.5 probability: hash % 100 > 50 -> global model, hash % 100 <= 50 -> custom model
PROJECT_ID_CUSTOM_MODEL = 100000  # hash % 100 = 39, <= 50 -> uses custom model
PROJECT_ID_GLOBAL_MODEL = 100051  # hash % 100 = 55, > 50 -> uses global model
HASH_SOURCE_CUSTOM = "test_hash_2"  # hash % 100 = 10, <= 50 -> uses custom model
HASH_SOURCE_GLOBAL = "test_hash_1"  # hash % 100 = 90, > 50 -> uses global model


class TestModelChooserChooseModel:
    """Test cases for ModelChooser.choose_model method."""

    @pytest.fixture(scope="session")
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture(scope="session")
    def app_config(self, temp_dir: str) -> ApplicationConfig:
        """Create test ApplicationConfig with filesystem storage."""
        return ApplicationConfig(
            binaryStoreType="filesystem",
            filesystemDefaultPath=temp_dir,
            bucketPrefix="test-",
        )

    @pytest.fixture(scope="session")
    def search_config(self, temp_dir: str) -> SearchConfig:
        """Create test SearchConfig with empty global model folders."""
        model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
        return SearchConfig(
            GlobalDefectTypeModelFolder=model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]
            .strip()
            .rstrip("/")
            .rstrip("\\"),
            SuggestBoostModelFolder="",
            BoostModelFolder="",
        )

    @pytest.fixture(scope="session")
    def object_saver_custom(self, app_config: ApplicationConfig) -> ObjectSaver:
        """Create a mock object_saver with a temporary directory."""
        saver = object_saving.create(
            app_config, project_id=PROJECT_ID_CUSTOM_MODEL, path=f"{ModelType.defect_type.name}_model/model"
        )
        return saver

    @pytest.fixture(scope="session")
    def object_saver_global(self, app_config: ApplicationConfig) -> ObjectSaver:
        """Create a mock object_saver with a temporary directory."""
        saver = object_saving.create(
            app_config, project_id=PROJECT_ID_GLOBAL_MODEL, path=f"{ModelType.defect_type.name}_model/model"
        )
        return saver

    @pytest.fixture(scope="session")
    def custom_model_test(self, object_saver_custom: ObjectSaver) -> DefectTypeModel:
        """Create a mock custom DefectTypeModel."""
        model = CustomDefectTypeModel(object_saver_custom)
        model.save_model()
        return model

    @pytest.fixture(scope="session")
    def global_model_test(self, object_saver_global: ObjectSaver) -> DefectTypeModel:
        """Create a mock custom DefectTypeModel."""
        model = CustomDefectTypeModel(object_saver_global)
        model.save_model()
        return model

    @pytest.fixture(scope="session")
    def model_chooser(self, app_config: ApplicationConfig, search_config: SearchConfig) -> ModelChooser:
        """Create ModelChooser instance with mocked models."""
        chooser = ModelChooser(app_config, search_config)
        return chooser

    def test_same_result_with_same_project_id_default_params(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel
    ) -> None:
        """Test 1.

        Method returns the same result (custom model) if hash_source and custom_model_prob are not set and project_id
        is the same.
        """
        # Call the method multiple times with the same project_id
        result1: MlModel = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type)
        result2: MlModel = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type)

        # Should return the same model instance
        assert result1.get_model_info() == custom_model_test.get_model_info()
        assert result1.get_model_info() == result2.get_model_info()

    def test_same_result_with_different_project_id_prob_1_0(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel, global_model_test: DefectTypeModel
    ):
        """Test 2.

        Method returns the same result if hash_source is not set, custom_model_prob is 1.0, and project_id is
        different.
        """
        # Call with different project IDs but custom_model_prob=1.0 (always try custom)
        result1 = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=1.0)
        result2 = model_chooser.choose_model(PROJECT_ID_GLOBAL_MODEL, ModelType.defect_type, custom_model_prob=1.0)

        assert result1.get_model_info() == custom_model_test.get_model_info()
        assert result2.get_model_info() == global_model_test.get_model_info()

    def test_same_result_with_same_project_id_prob_0_5(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel
    ):
        """Test 3.

        Method returns the same result if hash_source is not set, custom_model_prob is 0.5, but project_id is the same.
        """
        # Call multiple times with the same project_id and custom_model_prob=0.5
        result1 = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5)
        result2 = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5)

        # Should return the same model instance due to deterministic hash
        assert result1.get_model_info() == result2.get_model_info() == custom_model_test.get_model_info()

    def test_different_results_with_different_project_id_prob_0_5(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel, global_model_test: DefectTypeModel
    ):
        """Test 4.

        Method returns different results if hash_source is not set, custom_model_prob is 0.5 and project_id is
        different.
        """
        # Call with different project IDs that should produce different hash results
        result1 = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5)
        result2 = model_chooser.choose_model(PROJECT_ID_GLOBAL_MODEL, ModelType.defect_type, custom_model_prob=0.5)

        # Should return different model types based on hash
        # PROJECT_ID_CUSTOM_MODEL (hash % 100 = 39 <= 50) should use custom model
        # PROJECT_ID_GLOBAL_MODEL (hash % 100 = 55 > 50) should use global model
        assert result1.get_model_info() == custom_model_test.get_model_info()  # Should be custom model
        assert result2 != custom_model_test.get_model_info()
        assert result2 != global_model_test.get_model_info()

    def test_same_result_with_same_hash_source_default_params(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel
    ):
        """Test 5a: Method returns the same result when hash_source is the same and custom_model_prob is not set."""
        # Call multiple times with the same hash_source
        result1 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, hash_source=HASH_SOURCE_CUSTOM
        )
        result2 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, hash_source=HASH_SOURCE_CUSTOM
        )

        # Should return the same model instance
        assert result1.get_model_info() == result2.get_model_info() == custom_model_test.get_model_info()

    def test_same_result_with_different_hash_source_prob_1_0(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel
    ):
        """Test 5b: Method returns the same result when hash_source is different but custom_model_prob is 1.0."""
        # Call with different hash sources but custom_model_prob=1.0
        result1 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=1.0, hash_source=HASH_SOURCE_CUSTOM
        )
        result2 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=1.0, hash_source=HASH_SOURCE_GLOBAL
        )

        # Both should try to use custom model since prob=1.0
        assert result1.get_model_info() == result2.get_model_info() == custom_model_test.get_model_info()

    def test_same_result_with_same_hash_source_prob_0_5(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel
    ):
        """Test 5c: Method returns the same result when hash_source is the same and custom_model_prob is 0.5."""
        # Call multiple times with the same hash_source and custom_model_prob=0.5
        result1 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5, hash_source=HASH_SOURCE_CUSTOM
        )
        result2 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5, hash_source=HASH_SOURCE_CUSTOM
        )

        # Should return the same model instance
        assert result1.get_model_info() == result2.get_model_info() == custom_model_test.get_model_info()

    def test_different_results_with_different_hash_source_prob_0_5(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel, global_model_test: DefectTypeModel
    ):
        """Test 5d: Method returns different results when hash_source is different and custom_model_prob is 0.5."""
        # Call with different hash sources that should produce different hash results
        result1 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5, hash_source=HASH_SOURCE_CUSTOM
        )
        result2 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5, hash_source=HASH_SOURCE_GLOBAL
        )

        # Should return different model types based on hash
        # HASH_SOURCE_CUSTOM (hash % 100 = 10 <= 50) should use custom model
        # HASH_SOURCE_GLOBAL (hash % 100 = 90 > 50) should use global model
        assert result1.get_model_info() == custom_model_test.get_model_info()  # Should be custom model
        assert (
            result2.get_model_info() != global_model_test.get_model_info()
            and result2.get_model_info() != custom_model_test.get_model_info()
        )  # Should be global model

    def test_hash_source_overrides_project_id(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel, global_model_test: DefectTypeModel
    ):
        """Test 6: Test when project_id and hash_source are set and hash_source overrides project_id."""
        # Use PROJECT_ID_GLOBAL_MODEL (would use global) but with HASH_SOURCE_CUSTOM (would use custom)
        result1 = model_chooser.choose_model(
            PROJECT_ID_GLOBAL_MODEL, ModelType.defect_type, custom_model_prob=0.5, hash_source=HASH_SOURCE_CUSTOM
        )

        # Use PROJECT_ID_CUSTOM_MODEL (would use custom) but with HASH_SOURCE_GLOBAL (would use global)
        result2 = model_chooser.choose_model(
            PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.5, hash_source=HASH_SOURCE_GLOBAL
        )

        # hash_source should override project_id behavior
        assert result1.get_model_info() == custom_model_test.get_model_info()  # Should be custom model
        assert (
            result2.get_model_info() != global_model_test.get_model_info()
            and result2.get_model_info() != custom_model_test.get_model_info()
        )  # Should be global model

    def test_deterministic_behavior(self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel):
        """Test that hash-based selection is deterministic and repeatable."""
        # Test with specific values to ensure deterministic behavior
        test_cases = [
            (PROJECT_ID_CUSTOM_MODEL, None, 0.5),
            (PROJECT_ID_GLOBAL_MODEL, None, 0.5),
            (PROJECT_ID_CUSTOM_MODEL, HASH_SOURCE_CUSTOM, 0.5),
            (PROJECT_ID_CUSTOM_MODEL, HASH_SOURCE_GLOBAL, 0.5),
        ]

        for project_id, hash_source, prob in test_cases:
            # Call multiple times with the same parameters
            results: list[MlModel] = []
            for _ in range(5):
                result = model_chooser.choose_model(
                    project_id, ModelType.defect_type, custom_model_prob=prob, hash_source=hash_source
                )
                results.append(result)

            # All results should be identical
            result = all(r.get_model_info() == results[0].get_model_info() for r in results)
            assert result, f"Non-deterministic behavior for project_id={project_id}, hash_source={hash_source}"

    def test_edge_cases(
        self, model_chooser: ModelChooser, custom_model_test: DefectTypeModel, global_model_test: DefectTypeModel
    ):
        """Test edge cases with different probability values."""
        # With probability 0.0, should always return global model
        result = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=0.0)
        assert result.get_model_info() != custom_model_test.get_model_info()

        # With probability 100.0, should try to use custom model if available
        result = model_chooser.choose_model(PROJECT_ID_CUSTOM_MODEL, ModelType.defect_type, custom_model_prob=100.0)
        assert result.get_model_info() == custom_model_test.get_model_info()

        # Use a project ID that has no custom model saved - should fall back to global
        project_id_no_custom = 999999999
        result = model_chooser.choose_model(project_id_no_custom, ModelType.defect_type, custom_model_prob=1.0)
        assert result.get_model_info() != global_model_test.get_model_info()
