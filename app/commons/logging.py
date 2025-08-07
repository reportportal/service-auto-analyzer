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

"""Logging adapter to add correlation id to each log entry which Analyzer outputs."""

import base64
import logging.config
import uuid
from threading import local
from typing import Any, Optional

from app.commons.model.launch_objects import ApplicationConfig

CORRELATION_ID_PARAM = "correlation_id"

__INSTANCES = local()


def _process_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Process the keyword arguments to ensure that the correlation ID is included in the extra field."""
    my_kwargs = kwargs.copy()
    correlation_id = None
    if CORRELATION_ID_PARAM in my_kwargs:
        correlation_id = my_kwargs[CORRELATION_ID_PARAM]
    if not correlation_id:
        correlation_id = get_correlation_id()
    my_kwargs["extra"] = {CORRELATION_ID_PARAM: correlation_id}
    return my_kwargs


class Logger:
    __logger: logging.Logger

    def __init__(self, logger: logging.Logger):
        self.__logger = logger

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Delegate a debug call to the underlying logger."""
        my_kwargs = _process_kwargs(kwargs)
        self.__logger.debug(msg, *args, **my_kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Delegate an info call to the underlying logger."""
        my_kwargs = _process_kwargs(kwargs)
        self.__logger.info(msg, *args, **my_kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Delegate a warning call to the underlying logger."""
        my_kwargs = _process_kwargs(kwargs)
        self.__logger.warning(msg, *args, **my_kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Delegate an error call to the underlying logger."""
        my_kwargs = _process_kwargs(kwargs)
        self.__logger.error(msg, *args, **my_kwargs)

    def exception(self, msg: Any, *args: Any, exc_info: Optional[bool | BaseException] = True, **kwargs: Any) -> None:
        """Delegate an exception call to the underlying logger."""
        my_kwargs = _process_kwargs(kwargs)
        self.__logger.error(msg, *args, exc_info=exc_info, **my_kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Delegate a critical call to the underlying logger."""
        my_kwargs = _process_kwargs(kwargs)
        self.__logger.critical(msg, *args, **my_kwargs)


def new_correlation_id() -> str:
    corr_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")
    __INSTANCES.correlation_id = corr_id
    return corr_id


def get_correlation_id() -> str:
    corr_id = getattr(__INSTANCES, CORRELATION_ID_PARAM, None)
    if corr_id is None:
        corr_id = new_correlation_id()
    return corr_id


def set_correlation_id(corr_id: str):
    __INSTANCES.correlation_id = corr_id


# Sonar complains about the name of this function, but it must have the same name as in standard library
# noinspection PyPep8Naming
def getLogger(logger_name: str) -> Logger:  # NOSONAR
    return Logger(logging.getLogger(logger_name))


def setup(app_config: ApplicationConfig):
    log_file_path = "res/logging.conf"
    logging.config.fileConfig(log_file_path, defaults={"logfilename": app_config.analyzerPathToLog})
    if app_config.logLevel.lower() == "debug":
        logging.disable(logging.NOTSET)
    elif app_config.logLevel.lower() == "info":
        logging.disable(logging.DEBUG)
    else:
        logging.disable(logging.INFO)
