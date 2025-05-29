#   Copyright 2023 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from http import HTTPStatus
from typing import Any

import httpretty


def get_index_call(index_name: str, status: HTTPStatus) -> dict:
    """
    Returns a dictionary representing an HTTP call that simulates getting index.
    """
    return {
        "method": httpretty.GET,
        "uri": f"/{index_name}",
        "status": status,
    }


def get_index_found_call(index_name: str) -> dict:
    """
    Returns a dictionary representing an HTTP call that simulates a successful index retrieval.
    """
    # Mute invalid Sonar's "Change this argument; Function "get_index_call" expects a different type"
    return get_index_call(index_name, HTTPStatus.OK)  # NOSONAR


def get_index_not_found_call(index_name: str) -> dict:
    """
    Returns a dictionary representing an HTTP call that simulates an index not found error.
    """
    # Mute invalid Sonar's "Change this argument; Function "get_index_call" expects a different type"
    return get_index_call(index_name, HTTPStatus.NOT_FOUND)  # NOSONAR


def get_search_for_logs_call(index_name: str, query_parameters: str, rq: Any, rs: Any) -> dict:
    uri = f"/{index_name}/_search"
    if query_parameters:
        uri += f"?{query_parameters}"
    return {
        "method": httpretty.GET,
        "uri": uri,
        "status": HTTPStatus.OK,
        "content_type": "application/json",
        "rq": rq,
        "rs": rs,
    }


def get_search_for_logs_call_no_parameters(index_name: str, rq: Any, rs: Any) -> dict:
    return get_search_for_logs_call(index_name, "", rq, rs)


def get_search_for_logs_call_with_parameters(index_name: str, rq: Any, rs: Any) -> dict:
    return get_search_for_logs_call(index_name, "scroll=5m&size=1000", rq, rs)


def get_bulk_call(rq: Any, rs: Any) -> dict:
    call = {
        "method": httpretty.POST,
        "uri": "/_bulk?refresh=true",
        "status": HTTPStatus.OK,
        "content_type": "application/json",
        "rs": rs,
    }
    if rq is not None:
        call["rq"] = rq
    return call
