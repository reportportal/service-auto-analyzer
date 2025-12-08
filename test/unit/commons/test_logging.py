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

from unittest.mock import Mock

import pytest

from app.commons.logging import CORRELATION_ID_PARAM, Logger, set_correlation_id


@pytest.fixture
def mock_logger():
    return Mock()


@pytest.fixture
def logger(mock_logger):
    return Logger(mock_logger)


@pytest.mark.parametrize(
    "log_method",
    ["debug", "info", "warning", "error", "critical"],
)
def test_log_methods_add_correlation_id_to_extra(logger, mock_logger, log_method):
    set_correlation_id("test-corr-id")
    getattr(logger, log_method)("test message")
    getattr(mock_logger, log_method).assert_called_once()
    call_kwargs = getattr(mock_logger, log_method).call_args[1]
    assert "extra" in call_kwargs
    extra = call_kwargs["extra"]
    assert CORRELATION_ID_PARAM in extra
    assert extra[CORRELATION_ID_PARAM] == "test-corr-id"
    assert CORRELATION_ID_PARAM not in call_kwargs


@pytest.mark.parametrize(
    "log_method",
    ["debug", "info", "warning", "error", "critical"],
)
def test_log_methods_generates_correlation_id(logger, mock_logger, log_method):
    # noinspection PyTypeChecker
    set_correlation_id(None)  # NOSONAR
    getattr(logger, log_method)("test message")
    getattr(mock_logger, log_method).assert_called_once()
    call_kwargs = getattr(mock_logger, log_method).call_args[1]
    assert "extra" in call_kwargs
    extra = call_kwargs["extra"]
    assert CORRELATION_ID_PARAM in extra
    assert len(extra[CORRELATION_ID_PARAM]) == 22
    assert CORRELATION_ID_PARAM not in call_kwargs


@pytest.mark.parametrize(
    "log_method",
    ["debug", "info", "warning", "error", "critical"],
)
def test_log_methods_remove_correlation_id_from_kwargs(logger, mock_logger, log_method):
    getattr(logger, log_method)("test message", correlation_id="custom-corr-id")
    getattr(mock_logger, log_method).assert_called_once()
    call_kwargs = getattr(mock_logger, log_method).call_args[1]
    assert CORRELATION_ID_PARAM not in call_kwargs
    assert "extra" in call_kwargs
    assert call_kwargs["extra"][CORRELATION_ID_PARAM] == "custom-corr-id"


@pytest.mark.parametrize(
    "log_method",
    ["debug", "info", "warning", "error", "critical"],
)
def test_log_methods_pass_other_kwargs(logger, mock_logger, log_method):
    set_correlation_id("test-corr-id")
    getattr(logger, log_method)("test message", stacklevel=2)
    getattr(mock_logger, log_method).assert_called_once()
    call_kwargs = getattr(mock_logger, log_method).call_args[1]
    assert call_kwargs["stacklevel"] == 2


def test_exception_method_with_exc_info(logger, mock_logger):
    set_correlation_id("test-corr-id")
    logger.exception("exception message", exc_info=False)
    mock_logger.error.assert_called_once()
    call_kwargs = mock_logger.error.call_args[1]
    assert call_kwargs["exc_info"] is False
    assert "extra" in call_kwargs
    assert call_kwargs["extra"][CORRELATION_ID_PARAM] == "test-corr-id"


def test_exception_method_default_exc_info(logger, mock_logger):
    set_correlation_id("test-corr-id")
    logger.exception("exception message")
    mock_logger.error.assert_called_once()
    call_kwargs = mock_logger.error.call_args[1]
    assert call_kwargs["exc_info"] is True
    assert "extra" in call_kwargs


def test_exception_method_with_custom_correlation_id(logger, mock_logger):
    logger.exception("exception message", correlation_id="exc-corr-id")
    mock_logger.error.assert_called_once()
    call_kwargs = mock_logger.error.call_args[1]
    assert CORRELATION_ID_PARAM not in call_kwargs
    assert call_kwargs["extra"][CORRELATION_ID_PARAM] == "exc-corr-id"
