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

"""Helpers to read the replies of services in tests."""

import json
from typing import Optional, TypeVar

from pydantic import BaseModel

from app.commons.model.processing import Reply, ReplyType

ModelType = TypeVar("ModelType", bound=BaseModel)


def payload(replies: Optional[list[Reply]], reply_type: ReplyType = ReplyType.DIRECT_REPLY) -> str:
    """Get the payload of the only reply, and check its type.

    :param replies: Replies of a service
    :param reply_type: The expected reply type
    :return: The payload
    """
    assert replies is not None
    assert [reply.type for reply in replies] == [reply_type]
    return replies[0].payload


def reply_model(replies: Optional[list[Reply]], model: type[ModelType]) -> ModelType:
    """Parse the payload of the only direct reply into a model."""
    return model.model_validate_json(payload(replies))


def reply_models(
    replies: Optional[list[Reply]], model: type[ModelType], reply_type: ReplyType = ReplyType.DIRECT_REPLY
) -> list[ModelType]:
    """Parse the payload of the only reply, a JSON list, into models."""
    return [model.model_validate(item) for item in json.loads(payload(replies, reply_type))]
