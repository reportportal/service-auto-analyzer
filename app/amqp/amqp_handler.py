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

import json
import uuid
from typing import Callable, Any

from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from app.commons import launch_objects, logging

logger = logging.getLogger("analyzerApp.amqpHandler")


def prepare_launches(launches: list) -> list[launch_objects.Launch]:
    """Function for deserializing array of launches"""
    return [launch_objects.Launch(**launch) for launch in launches]


def prepare_suggest_info_list(suggest_info_list: list) -> list[launch_objects.SuggestAnalysisResult]:
    """Function for deserializing array of suggest info results"""
    return [launch_objects.SuggestAnalysisResult(**res) for res in suggest_info_list]


def prepare_search_logs(search_data: dict) -> launch_objects.SearchLogs:
    """Function for deserializing search logs object"""
    return launch_objects.SearchLogs(**search_data)


def prepare_launch_info(launch_info: dict) -> launch_objects.LaunchInfoForClustering:
    """Function for deserializing search logs object"""
    return launch_objects.LaunchInfoForClustering(**launch_info)


def prepare_clean_index(clean_index: dict) -> launch_objects.CleanIndex:
    """Function for deserializing clean index object"""
    return launch_objects.CleanIndex(**clean_index)


def prepare_delete_index(body: Any) -> int:
    """Function for deserializing index id object"""
    return int(body)


def prepare_test_item_info(test_item_info: Any) -> Any:
    """Function for deserializing test item info for suggestions"""
    return launch_objects.TestItemInfo(**test_item_info)


def prepare_search_response_data(response: list | dict) -> str:
    """Function for serializing response from search request"""
    return json.dumps(response)


def prepare_analyze_response_data(response: list) -> str:
    """Function for serializing response from analyze request"""
    return json.dumps([resp.dict() for resp in response])


def prepare_index_response_data(response: Any) -> str:
    """Function for serializing response from index request
    and other objects, which are pydantic objects"""
    return response.json()


def output_result(response: Any) -> str:
    """Function for serializing int object"""
    return str(response)


def __get_correlation_id() -> str:
    return str(uuid.uuid4())


def handle_amqp_request(channel: BlockingChannel, method: Basic.Deliver, props: BasicProperties, body: bytes,
                        request_handler: Callable[[Any], Any],
                        prepare_data_func: Callable[[Any], Any] = prepare_launches,
                        prepare_response_data: Callable[[Any], str] = prepare_search_response_data,
                        publish_result: bool = True) -> None:
    """Function for handling amqp request: index, search and analyze."""
    logging.new_correlation_id()
    logger.debug(f'Started message processing:\n--Method: {method}\n'
                 f'--Properties: {props}\n--Body: {body}')
    try:
        message = json.loads(body, strict=False)
    except Exception as exc:
        logger.error('Failed to parse message body to JSON')
        logger.exception(exc)
        return
    try:
        message = prepare_data_func(message)
    except Exception as exc:
        logger.error('Failed to prepare message body')
        logger.exception(exc)
        return
    try:
        response = request_handler(message)
    except Exception as exc:
        logger.error('Failed to handle message')
        logger.exception(exc)
        return

    try:
        response_body = prepare_response_data(response)
    except Exception as exc:
        logger.error('Failed to prepare response body')
        logger.exception(exc)
        return
    if publish_result:
        try:
            if props.reply_to:
                channel.basic_publish(exchange='', routing_key=props.reply_to, properties=BasicProperties(
                    correlation_id=props.correlation_id, content_type='application/json'), mandatory=False,
                                      body=bytes(response_body, 'utf-8'))
        except Exception as exc:
            logger.error('Failed to publish result')
            logger.exception(exc)
            return
    logger.debug(f'Finished processing message')


def handle_inner_amqp_request(channel, method, props, body, request_handler):
    """Function for handling inner amqp requests."""
    logging.new_correlation_id()
    logger.debug(f'Started message processing.\n--Method: {method}\n'
                 f'--Properties: {props}\n--Body: {body}')
    try:
        message = json.loads(body, strict=False)
    except Exception as exc:
        logger.error('Failed to parse message body to JSON')
        logger.exception(exc)
        return
    try:
        request_handler(message)
    except Exception as exc:
        logger.error('Failed to handle message')
        logger.exception(exc)
        return
    logger.debug(f'Finished processing message')
