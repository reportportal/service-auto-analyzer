#  Copyright 2026 EPAM Systems
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

"""Startup settings which choose the queues an instance consumes."""

import json
from unittest import mock

import pytest

from app import main
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.coordinator.coordinator import DiscardingCoordinator, LaneCoordinator
from app.utils import file_utils

WORKER_VARIABLES = ["ANALYZER_WORKERS", "ANALYZER_TRAIN_WORKERS", "DISABLE_TRAIN"]


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in WORKER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "variables, workers, train_workers, disable_train",
    [
        ({}, 2, 1, False),
        ({"ANALYZER_WORKERS": "0"}, 0, 1, False),
        ({"ANALYZER_WORKERS": "-3"}, 0, 1, False),
        ({"ANALYZER_TRAIN_WORKERS": "0"}, 2, 0, False),
        ({"ANALYZER_TRAIN_WORKERS": "3"}, 2, 3, False),
        ({"DISABLE_TRAIN": "true"}, 2, 1, True),
        ({"DISABLE_TRAIN": "true", "ANALYZER_TRAIN_WORKERS": "0"}, 2, 0, True),
    ],
)
def test_worker_settings(
    environment: pytest.MonkeyPatch,
    variables: dict[str, str],
    workers: int,
    train_workers: int,
    disable_train: bool,
):
    for name, value in variables.items():
        environment.setenv(name, value)

    app_config = main.build_app_config()

    assert app_config.analyzerWorkers == workers
    assert app_config.analyzerTrainWorkers == train_workers
    assert app_config.disableTrain is disable_train


@pytest.mark.parametrize(
    "workers, train_workers, disable_train, expected",
    [
        (2, 1, False, [("all", LaneCoordinator), ("train", LaneCoordinator)]),
        (0, 1, False, [("train", LaneCoordinator)]),
        (2, 0, False, [("all", LaneCoordinator)]),
        (0, 0, False, []),
        # Training disabled: the queue is drained, whatever the number of train workers is
        (2, 1, True, [("all", LaneCoordinator), ("train", DiscardingCoordinator)]),
        (2, 0, True, [("all", LaneCoordinator), ("train", DiscardingCoordinator)]),
        (0, 0, True, [("train", DiscardingCoordinator)]),
    ],
)
def test_create_coordinators(workers: int, train_workers: int, disable_train: bool, expected: list[tuple[str, type]]):
    app_config = ApplicationConfig(
        analyzerWorkers=workers, analyzerTrainWorkers=train_workers, disableTrain=disable_train
    )

    coordinators = main.create_coordinators(app_config, SearchConfig())

    assert [(coordinator.name, type(coordinator)) for coordinator in coordinators] == expected


def test_main_exits_when_nothing_to_consume(environment: pytest.MonkeyPatch):
    environment.setenv("ANALYZER_WORKERS", "0")
    environment.setenv("ANALYZER_TRAIN_WORKERS", "0")

    with mock.patch.object(main, "run") as run, pytest.raises(SystemExit) as exit_info:
        main.main()

    assert exit_info.value.code == 1
    run.assert_not_called()


@pytest.mark.parametrize(
    "workers, disabled, description",
    [
        (0, False, "consumed by another instance"),
        (0, True, "messages are discarded, training is disabled"),
        (2, True, "messages are discarded, training is disabled"),
        (2, False, "2 worker processes"),
        (1, False, "1 worker process"),
    ],
)
def test_describe_queue(workers: int, disabled: bool, description: str):
    assert main.describe_queue(workers, disabled) == description


def test_build_app_config_reads_version(environment: pytest.MonkeyPatch):
    with open("VERSION") as file:
        version = file.read().strip()

    assert main.build_app_config().appVersion == version


def test_build_search_config_reads_model_settings():
    with open("res/model_settings.json") as file:
        model_settings = json.load(file)

    search_config = main.build_search_config()

    assert search_config.BoostModelFolder == file_utils.strip_path(model_settings["BOOST_MODEL_FOLDER"])
    assert search_config.SuggestBoostModelFolder == file_utils.strip_path(model_settings["SUGGEST_BOOST_MODEL_FOLDER"])
    assert search_config.GlobalDefectTypeModelFolder == file_utils.strip_path(
        model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]
    )
