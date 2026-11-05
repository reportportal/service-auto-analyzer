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

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.commons.model.launch_objects import ApplicationConfig
from app.coordinator.health import HealthServer, OpenSearchHealth


class FakeCoordinator:
    def __init__(self, connected: bool = True) -> None:
        self.connected = connected

    def snapshot(self) -> dict[str, Any]:
        return {"name": "all", "connected": self.connected, "workers": [], "queued": 0}


def make_opensearch(healthy: bool, interval: float = 20.0) -> tuple[OpenSearchHealth, Mock]:
    os_client = Mock()
    os_client.is_healthy = Mock(return_value=healthy)
    return OpenSearchHealth(ApplicationConfig(), interval=interval, os_client=os_client), os_client


async def get_status(server: HealthServer) -> tuple[int, dict[str, Any], str]:
    async with TestClient(TestServer(server.make_app())) as client:
        response = await client.get("/")
        return response.status, await response.json(), response.headers.get("Access-Control-Allow-Origin", "")


@pytest.mark.asyncio
async def test_healthy_status():
    opensearch, os_client = make_opensearch(True)
    await opensearch.check()
    server = HealthServer(ApplicationConfig(appVersion="1.2.3"), [FakeCoordinator()], opensearch)  # type: ignore

    status, body, cors = await get_status(server)

    assert status == 200
    assert body["status"] == "healthy"
    assert body["version"] == "1.2.3"
    assert body["opensearch"]["healthy"] is True
    assert body["coordinators"] == [{"name": "all", "connected": True, "workers": [], "queued": 0}]
    assert cors == "*"
    os_client.is_healthy.assert_called_once()


@pytest.mark.asyncio
async def test_endpoint_uses_cached_opensearch_status():
    opensearch, os_client = make_opensearch(True)
    await opensearch.check()
    server = HealthServer(ApplicationConfig(), [FakeCoordinator()], opensearch)  # type: ignore

    await get_status(server)
    await get_status(server)

    os_client.is_healthy.assert_called_once()


@pytest.mark.asyncio
async def test_unhealthy_opensearch():
    opensearch, _ = make_opensearch(False)
    await opensearch.check()
    server = HealthServer(ApplicationConfig(), [FakeCoordinator()], opensearch)  # type: ignore

    status, body, _ = await get_status(server)

    assert status == 503
    assert body["status"] == "OpenSearch is not healthy"


@pytest.mark.asyncio
async def test_stale_opensearch_status_is_unhealthy():
    opensearch, _ = make_opensearch(True, interval=0.01)
    await opensearch.check()
    await asyncio.sleep(0.05)
    server = HealthServer(ApplicationConfig(), [FakeCoordinator()], opensearch)  # type: ignore

    status, body, _ = await get_status(server)

    assert status == 503
    assert body["opensearch"]["healthy"] is False


@pytest.mark.asyncio
async def test_pending_first_check_is_unhealthy():
    opensearch, _ = make_opensearch(True)
    server = HealthServer(ApplicationConfig(), [FakeCoordinator()], opensearch)  # type: ignore

    status, body, _ = await get_status(server)

    assert status == 503
    assert body["status"] == "OpenSearch is not healthy"
    assert body["opensearch"] == {"healthy": None, "checked_seconds_ago": None}


@pytest.mark.asyncio
async def test_disconnected_consumer():
    opensearch, _ = make_opensearch(True)
    await opensearch.check()
    server = HealthServer(ApplicationConfig(), [FakeCoordinator(), FakeCoordinator(False)], opensearch)  # type: ignore

    status, body, _ = await get_status(server)

    assert status == 503
    assert body["status"] == "AMQP is not connected"


@pytest.mark.asyncio
async def test_background_check_runs_and_stops():
    opensearch, os_client = make_opensearch(True)

    opensearch.start()
    for _ in range(100):
        if os_client.is_healthy.called:
            break
        await asyncio.sleep(0.01)
    await opensearch.stop()
    await asyncio.sleep(0.05)

    os_client.is_healthy.assert_called_once()
    assert opensearch.status()["healthy"] is True
