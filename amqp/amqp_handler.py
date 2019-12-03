"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import logging
import json
import pika
import commons.launch_objects as launch_objects

logger = logging.getLogger("analyzerApp.amqpHandler")

def prepare_launches(launches):
    return [launch_objects.Launch(**launch) for launch in launches]

def prepare_search_logs(search_data):
    return launch_objects.SearchLogs(**search_data)

def prepare_search_response_data(response):
    return json.dumps(response)

def prepare_analyze_response_data(response):
    return json.dumps([resp.dict() for resp in response])

def prepare_index_response_data(response):
    return response.json()

def handle_amqp_request(channel, method, props, body,
        request_handler, prepare_data_func = prepare_launches,
        prepare_response_data = prepare_search_response_data):
    logger.debug("Started processing {} method".format(method))
    logger.debug("Started processing data {}".format(body))
    try:
        launches = json.loads(body, strict=False)
    except Exception as err:
        logger.error("Failed to load json from body")
        logger.error(err)
        return
    try:
        launches = prepare_data_func(launches)
    except Exception as err:
        logger.error("Failed to transform body into objects")
        logger.error(err)
        return
    try:
        response = request_handler(launches)
    except Exception as err:
        logger.error("Failed to process launches")
        logger.error(err)
        return

    try:
        response_body = prepare_response_data(response)
    except Exception as err:
        logger.error("Failed to dump launches result")
        logger.error(err)
        return
    try:
        channel.basic_publish(exchange = '',
                         routing_key = props.reply_to,
                         properties = pika.BasicProperties(
                            correlation_id = props.correlation_id,
                            content_type = "application/json"),
                         mandatory = False,
                         body = response_body)
    except Exception as err:
        logger.error("Failed to publish result")
        logger.error(err)
    logger.debug("Finished processing {} method".format(method))
    return True

def handle_delete_request(channel, method, props, body, request_handler):
    logger.debug("Started processing {} method".format(method))
    logger.debug("Started processing data {}".format(body))
    index_id = None
    try:
        index_id = int(body)
    except Exception as err:
        logger.error("Failed to transform index_id to int")
        logger.error(err)
        return
    try:
        response = request_handler(index_id)
    except Exception as err:
        logger.error("Failed to delete index")
        logger.error(err)
        return
    logger.debug("Finished processing {} method".format(method))
    return True

def handle_clean_request(channel, method, props, body, request_handler):
    logger.debug("Started processing {} method".format(method))
    logger.debug("Started processing data {}".format(body))
    clean_index = None
    try:
        clean_index = json.loads(body, strict=False)
    except Exception as err:
        logger.error("Failed to load json from body")
        logger.error(err)
        return
    try:
        clean_index = launch_objects.CleanIndex(**clean_index)
    except Exception as err:
        logger.error("Failed to transform clean index into object")
        logger.error(err)
        return
    try:
        response = request_handler(clean_index)
    except Exception as err:
        logger.error("Failed to clean index")
        logger.error(err)
        return
    logger.debug("Finished processing {} method".format(method))
    return True