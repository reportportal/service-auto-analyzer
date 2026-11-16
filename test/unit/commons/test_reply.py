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

"""The reply which services return."""

import json

import pytest

from app.commons.model.launch_objects import AnalysisResult
from app.commons.model.processing import Reply, ReplyType, to_json_list


def test_reply_is_direct_by_default():
    reply = Reply("[]")

    assert reply.type is ReplyType.DIRECT_REPLY
    assert reply.routing_key is None


@pytest.mark.parametrize("routing_key", [None, ""])
def test_queue_send_requires_routing_key(routing_key):
    with pytest.raises(ValueError, match="requires a routing key"):
        Reply("{}", ReplyType.QUEUE_SEND, routing_key)


def test_queue_send_with_routing_key():
    assert Reply("{}", ReplyType.QUEUE_SEND, "train_models").routing_key == "train_models"


def test_to_json_list_of_no_models():
    assert to_json_list([]) == "[]"


def test_to_json_list_uses_model_dump_json():
    results = [
        AnalysisResult(testItem=1, issueType="pb001", relevantItem=7),
        AnalysisResult(testItem=2, issueType="ab001", relevantItem=8),
    ]

    payload = to_json_list(results)

    assert payload == f"[{results[0].model_dump_json()},{results[1].model_dump_json()}]"
    assert json.loads(payload) == [result.model_dump() for result in results]


def test_to_json_list_keeps_non_ascii_text():
    payload = to_json_list([AnalysisResult(testItem=1, issueType="тест", relevantItem=2)])

    assert "тест" in payload
    assert json.loads(payload)[0]["issueType"] == "тест"


def test_to_json_list_excludes_fields():
    results = [AnalysisResult(testItem=1, issueType="pb001", relevantItem=7)]

    assert json.loads(to_json_list(results, exclude={"relevantItem"})) == [{"testItem": 1, "issueType": "pb001"}]
