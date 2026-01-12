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

from time import time
from typing import Optional

from app.commons import logging
from app.commons.model.launch_objects import (
    ApplicationConfig,
    DeleteLaunchesRequest,
    DeleteLogsRequest,
    DeleteTestItemsRequest,
    RemoveByDatesRequest,
)
from app.commons.os_client import OsClient

LOGGER = logging.getLogger("analyzerApp.cleanIndexService")


class CleanIndexService:
    os_client: OsClient

    def __init__(
        self,
        app_config: ApplicationConfig,
        *,
        os_client: Optional[OsClient] = None,
    ):
        """Initialize CleanIndexService

        :param app_config: Application configuration object
        :param os_client: Optional OsClient instance. If not provided, a new one will be created.
        """
        self.app_config = app_config
        self.os_client = os_client or OsClient(app_config=self.app_config)

    def delete_logs(self, clean_index: DeleteLogsRequest) -> int:
        LOGGER.info("Started cleaning index")
        t_start = time()
        log_ids = [str(log_id) for log_id in clean_index.ids]
        deleted_logs_cnt = self.os_client.delete_logs_by_ids(clean_index.project, log_ids)
        LOGGER.info("Finished cleaning index %.2f s", time() - t_start)
        return deleted_logs_cnt

    def delete_test_items(self, remove_items_info: DeleteTestItemsRequest) -> int:
        LOGGER.info("Started removing test items")
        t_start = time()
        test_item_ids = [str(test_item_id) for test_item_id in remove_items_info.itemsToDelete]
        deleted_logs_cnt = self.os_client.delete_test_items(remove_items_info.project, test_item_ids)
        LOGGER.info("Finished removing test items %.2f s", time() - t_start)
        return deleted_logs_cnt

    def delete_launches(self, launch_remove_info: DeleteLaunchesRequest) -> int:
        LOGGER.info("Started removing launches")
        t_start = time()
        launch_ids = [str(launch_id) for launch_id in launch_remove_info.launch_ids]
        deleted_logs_cnt = self.os_client.delete_by_launch_ids(launch_remove_info.project, launch_ids)
        LOGGER.info("Finished removing launches %.2f s", time() - t_start)
        return deleted_logs_cnt

    def remove_by_launch_start_time(self, remove_by_launch_start_time_info: RemoveByDatesRequest) -> int:
        project: int | str = remove_by_launch_start_time_info.project
        start_date: str = remove_by_launch_start_time_info.interval_start_date
        end_date: str = remove_by_launch_start_time_info.interval_end_date
        LOGGER.info("Started removing logs by launch start time")
        t_start = time()
        deleted_logs_cnt = self.os_client.delete_by_launch_start_time_range(project, start_date, end_date)
        LOGGER.info("Finished removing logs by launch start time %.2f s", time() - t_start)
        return deleted_logs_cnt

    def remove_by_log_time(self, remove_by_log_time_info: RemoveByDatesRequest) -> int:
        project: int | str = remove_by_log_time_info.project
        start_date: str = remove_by_log_time_info.interval_start_date
        end_date: str = remove_by_log_time_info.interval_end_date
        LOGGER.info("Started removing logs by log time range")
        t_start = time()
        deleted_logs_cnt = self.os_client.delete_by_log_time_range(project, start_date, end_date)
        LOGGER.info("Finished removing logs by log time range %.2f s", time() - t_start)
        return deleted_logs_cnt
