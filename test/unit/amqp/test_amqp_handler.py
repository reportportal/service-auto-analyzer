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

import pytest
from opensearchpy import ConflictError, NotFoundError

from app.amqp.amqp_handler import retry
from app.commons.model.processing import ProcessingItem


def _make_processing_item(routing_key: str) -> ProcessingItem:
    return ProcessingItem(
        priority=10,
        number=1,
        routing_key=routing_key,
        reply_to=None,
        log_correlation_id="log-id",
        msg_correlation_id="msg-id",
        item={"payload": "value"},
    )


@pytest.mark.parametrize(
    ("exc", "routing_key", "expected"),
    [
        (ConflictError(409, "conflict but no document was found", {}), "item_remove", False),
        (ConflictError(409, "conflict occurred", {}), "item_remove", True),
        (ConflictError(409, "conflict but no document was found", {}), "other_key", True),
        (NotFoundError(404, "resource not found: no such index", {}), "remove_by_launch_start_time", False),
        (NotFoundError(404, "resource not found", {}), "remove_by_launch_start_time", True),
        (NotFoundError(404, "resource not found: no such index", {}), "other_key", True),
        (ValueError("Input X contains NaN, check your data"), "train_models", False),
        (ValueError("unexpected value"), "train_models", True),
        (ValueError("Input X contains NaN, check your data"), "other_key", True),
        (RuntimeError("failure"), "any_key", True),
    ],
)
def test_retry_returns_expected_value(exc: Exception, routing_key: str, expected: bool) -> None:
    item = _make_processing_item(routing_key)
    result = retry(item, exc)

    assert result is expected
