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

from app.commons import logging
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import CleanIndexStrIds, ApplicationConfig
from app.service.suggest_info_service import SuggestInfoService
from app.utils import utils

logger = logging.getLogger("analyzerApp.cleanIndexService")


class CleanIndexService:
    es_client: EsClient
    suggest_info_service: SuggestInfoService

    def __init__(self, app_config: ApplicationConfig):
        self.app_config = app_config
        self.es_client = EsClient(app_config=self.app_config)
        self.suggest_info_service = SuggestInfoService(app_config=app_config)

    @utils.ignore_warnings
    def delete_logs(self, clean_index):
        logger.info("Started cleaning index")
        t_start = time()
        deleted_logs_cnt = self.es_client.delete_logs(clean_index)
        self.suggest_info_service.clean_suggest_info_logs(clean_index)
        logger.info("Finished cleaning index %.2f s", time() - t_start)
        return deleted_logs_cnt

    @utils.ignore_warnings
    def delete_test_items(self, remove_items_info):
        logger.info("Started removing test items")
        t_start = time()
        deleted_logs_cnt = self.es_client.remove_test_items(remove_items_info)
        self.suggest_info_service.clean_suggest_info_logs_by_test_item(remove_items_info)
        logger.info("Finished removing test items %.2f s", time() - t_start)
        return deleted_logs_cnt

    @utils.ignore_warnings
    def delete_launches(self, launch_remove_info):
        logger.info("Started removing launches")
        t_start = time()
        deleted_logs_cnt = self.es_client.remove_launches(launch_remove_info)
        self.suggest_info_service.clean_suggest_info_logs_by_launch_id(launch_remove_info)
        logger.info("Finished removing launches %.2f s", time() - t_start)
        return deleted_logs_cnt

    @utils.ignore_warnings
    def remove_by_launch_start_time(self, remove_by_launch_start_time_info: dict) -> int:
        project: int = remove_by_launch_start_time_info["project"]
        start_date: str = remove_by_launch_start_time_info["interval_start_date"]
        end_date: str = remove_by_launch_start_time_info["interval_end_date"]
        logger.info("Started removing logs by launch start time")
        t_start = time()
        launch_ids = self.es_client.get_launch_ids_by_start_time_range(
            project, start_date, end_date
        )
        deleted_logs_cnt = self.es_client.remove_by_launch_start_time_range(
            project, start_date, end_date
        )
        launch_remove_info = {
            "project": project,
            "launch_ids": launch_ids
        }
        self.suggest_info_service.clean_suggest_info_logs_by_launch_id(launch_remove_info)
        logger.info(
            "Finished removing logs by launch start time %.2f s", time() - t_start
        )
        return deleted_logs_cnt

    @utils.ignore_warnings
    def remove_by_log_time(self, remove_by_log_time_info: dict) -> int:
        project: int = remove_by_log_time_info["project"]
        start_date: str = remove_by_log_time_info["interval_start_date"]
        end_date: str = remove_by_log_time_info["interval_end_date"]
        logger.info("Started removing logs by log time range")
        t_start = time()
        log_ids = self.es_client.get_log_ids_by_log_time_range(
            project, start_date, end_date
        )
        deleted_logs_cnt = self.es_client.remove_by_log_time_range(
            project, start_date, end_date
        )
        clean_index = CleanIndexStrIds(ids=log_ids, project=project)
        self.suggest_info_service.clean_suggest_info_logs(clean_index)
        logger.info(
            "Finished removing logs by log time range %.2f s", time() - t_start
        )
        return deleted_logs_cnt
