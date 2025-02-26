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
import logging
import uuid
from threading import local

__INSTANCES = local()


class Logger:
    __logger: logging.Logger

    def __init__(self, logger: logging.Logger):
        self.__logger = logger

    def debug(self, msg, *args, **kwargs):
        """
        Delegate a debug call to the underlying logger.
        """
        kwargs['extra'] = {'correlation_id': get_correlation_id()}
        self.__logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Delegate an info call to the underlying logger.
        """
        kwargs['extra'] = {'correlation_id': get_correlation_id()}
        self.__logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Delegate a warning call to the underlying logger.
        """
        kwargs['extra'] = {'correlation_id': get_correlation_id()}
        self.__logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Delegate an error call to the underlying logger.
        """
        kwargs['extra'] = {'correlation_id': get_correlation_id()}
        self.__logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Delegate an exception call to the underlying logger.
        """
        kwargs['extra'] = {'correlation_id': get_correlation_id()}
        self.__logger.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Delegate a critical call to the underlying logger.
        """
        kwargs['extra'] = {'correlation_id': get_correlation_id()}
        self.__logger.critical(msg, *args, **kwargs)


def new_correlation_id() -> str:
    corr_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=')
    __INSTANCES.correlation_id = corr_id
    return corr_id


def get_correlation_id() -> str:
    corr_id = getattr(__INSTANCES, 'correlation_id', None)
    if corr_id is None:
        corr_id = new_correlation_id()
    return corr_id


# noinspection PyPep8Naming
def getLogger(logger_name: str) -> Logger:  # NOSONAR
    return Logger(logging.getLogger(logger_name))
