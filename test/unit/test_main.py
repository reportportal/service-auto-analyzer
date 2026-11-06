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
import warnings
from typing import Any
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


DATASTORE_VARIABLES = [
    "DATASTORE_TYPE",
    "ANALYZER_BINSTORE_TYPE",
    "ANALYZER_BINARYSTORE_TYPE",
    "DATASTORE_ENDPOINT",
    "MINIO_SHORT_HOST",
    "MINIO_USE_TLS",
    "DATASTORE_REGION",
    "ANALYZER_BINSTORE_MINIO_REGION",
    "ANALYZER_BINARYSTORE_MINIO_REGION",
    "DATASTORE_DEFAULTBUCKETNAME",
    "DATASTORE_BUCKETPREFIX",
    "ANALYZER_BINSTORE_BUCKETPREFIX",
    "ANALYZER_BINARYSTORE_BUCKETPREFIX",
    "DATASTORE_BUCKETPOSTFIX",
    "DATASTORE_ACCESSKEY",
    "MINIO_ACCESS_KEY",
    "DATASTORE_SECRETKEY",
    "MINIO_SECRET_KEY",
]


@pytest.fixture
def datastore_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in DATASTORE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def datastore_settings(app_config: ApplicationConfig) -> dict[str, Any]:
    return {
        "type": app_config.datastoreType,
        "endpoint": app_config.datastoreEndpoint,
        "region": app_config.datastoreRegion,
        "prefix": app_config.datastoreBucketPrefix,
        "postfix": app_config.datastoreBucketPostfix,
        "access_key": app_config.datastoreAccessKey,
        "secret_key": app_config.datastoreSecretKey,
    }


def deprecation(old: str, new: str) -> str:
    return f"'{old}' configuration variable is deprecated, use '{new}' instead."


def test_datastore_defaults(datastore_environment: pytest.MonkeyPatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        app_config = main.build_app_config()

    assert datastore_settings(app_config) == {
        "type": "filesystem",
        "endpoint": None,
        "region": None,
        "prefix": "prj-",
        "postfix": "",
        "access_key": None,
        "secret_key": None,
    }


def test_datastore_new_variables_win_over_deprecated(datastore_environment: pytest.MonkeyPatch):
    new_values = {
        "DATASTORE_TYPE": "s3",
        "DATASTORE_ENDPOINT": "https://s3.example.com",
        "DATASTORE_REGION": "eu-west-1",
        "DATASTORE_BUCKETPREFIX": "new-",
        "DATASTORE_ACCESSKEY": "new-access",
        "DATASTORE_SECRETKEY": "new-secret",
    }
    old_values = {
        "ANALYZER_BINSTORE_TYPE": "minio",
        "ANALYZER_BINARYSTORE_TYPE": "minio",
        "MINIO_SHORT_HOST": "minio:9000",
        "MINIO_USE_TLS": "true",
        "ANALYZER_BINSTORE_MINIO_REGION": "old-region-1",
        "ANALYZER_BINARYSTORE_MINIO_REGION": "old-region-2",
        "ANALYZER_BINSTORE_BUCKETPREFIX": "old-1-",
        "ANALYZER_BINARYSTORE_BUCKETPREFIX": "old-2-",
        "MINIO_ACCESS_KEY": "old-access",
        "MINIO_SECRET_KEY": "old-secret",
    }
    for name, value in {**new_values, **old_values}.items():
        datastore_environment.setenv(name, value)

    with pytest.warns(DeprecationWarning) as record:
        app_config = main.build_app_config()

    assert datastore_settings(app_config) == {
        "type": "s3",
        "endpoint": "https://s3.example.com",
        "region": "eu-west-1",
        "prefix": "new-",
        "postfix": "",
        "access_key": "new-access",
        "secret_key": "new-secret",
    }
    assert [str(warning.message) for warning in record] == [
        deprecation("ANALYZER_BINSTORE_TYPE", "DATASTORE_TYPE"),
        deprecation("ANALYZER_BINARYSTORE_TYPE", "DATASTORE_TYPE"),
        deprecation("MINIO_SHORT_HOST", "DATASTORE_ENDPOINT"),
        deprecation("MINIO_USE_TLS", "DATASTORE_ENDPOINT"),
        deprecation("ANALYZER_BINSTORE_MINIO_REGION", "DATASTORE_REGION"),
        deprecation("ANALYZER_BINARYSTORE_MINIO_REGION", "DATASTORE_REGION"),
        deprecation("ANALYZER_BINSTORE_BUCKETPREFIX", "DATASTORE_BUCKETPREFIX"),
        deprecation("ANALYZER_BINARYSTORE_BUCKETPREFIX", "DATASTORE_BUCKETPREFIX"),
        deprecation("MINIO_ACCESS_KEY", "DATASTORE_ACCESSKEY"),
        deprecation("MINIO_SECRET_KEY", "DATASTORE_SECRETKEY"),
    ]


@pytest.mark.parametrize(
    "variables, expected",
    [
        ({"ANALYZER_BINSTORE_TYPE": "minio"}, {"type": "minio"}),
        ({"ANALYZER_BINARYSTORE_TYPE": "minio"}, {"type": "minio"}),
        ({"ANALYZER_BINSTORE_TYPE": "minio", "ANALYZER_BINARYSTORE_TYPE": "s3"}, {"type": "minio"}),
        ({"MINIO_SHORT_HOST": "minio:9000"}, {"endpoint": "http://minio:9000"}),
        ({"MINIO_SHORT_HOST": "minio:9000", "MINIO_USE_TLS": "true"}, {"endpoint": "https://minio:9000"}),
        ({"MINIO_USE_TLS": "true"}, {}),
        ({"ANALYZER_BINARYSTORE_MINIO_REGION": "region-2"}, {"region": "region-2"}),
        (
            {"ANALYZER_BINSTORE_MINIO_REGION": "region-1", "ANALYZER_BINARYSTORE_MINIO_REGION": "region-2"},
            {"region": "region-1"},
        ),
        ({"ANALYZER_BINARYSTORE_BUCKETPREFIX": "old-2-"}, {"prefix": "old-2-"}),
        (
            {"ANALYZER_BINSTORE_BUCKETPREFIX": "old-1-", "ANALYZER_BINARYSTORE_BUCKETPREFIX": "old-2-"},
            {"prefix": "old-1-"},
        ),
        (
            {"MINIO_ACCESS_KEY": "access", "MINIO_SECRET_KEY": "secret"},
            {"access_key": "access", "secret_key": "secret"},
        ),
    ],
)
def test_datastore_deprecated_variables_are_used(
    datastore_environment: pytest.MonkeyPatch, variables: dict[str, str], expected: dict[str, Any]
):
    for name, value in variables.items():
        datastore_environment.setenv(name, value)

    with pytest.warns(DeprecationWarning):
        app_config = main.build_app_config()

    settings = datastore_settings(app_config)
    assert {name: settings[name] for name in expected} == expected


@pytest.mark.parametrize(
    "variables, prefix",
    [
        ({"DATASTORE_DEFAULTBUCKETNAME": "bucket"}, "bucket"),
        ({"DATASTORE_DEFAULTBUCKETNAME": "bucket", "DATASTORE_BUCKETPREFIX": "prj-"}, "bucket/prj-"),
        ({"DATASTORE_BUCKETPREFIX": "custom-"}, "custom-"),
    ],
)
def test_datastore_default_bucket_name(
    datastore_environment: pytest.MonkeyPatch, variables: dict[str, str], prefix: str
):
    for name, value in variables.items():
        datastore_environment.setenv(name, value)

    assert main.build_app_config().datastoreBucketPrefix == prefix


def test_datastore_bucket_postfix(datastore_environment: pytest.MonkeyPatch):
    datastore_environment.setenv("DATASTORE_BUCKETPOSTFIX", "-data")

    assert main.build_app_config().datastoreBucketPostfix == "-data"
