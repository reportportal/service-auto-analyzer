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

"""Health endpoint, served on the coordinator's event loop."""

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Final, Optional

from aiohttp import web

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.os_client import OsClient
from app.coordinator.coordinator import Coordinator

LOGGER = logging.getLogger("analyzerApp.health")

OPENSEARCH_CHECK_INTERVAL: Final[float] = 20.0
# A result older than this many check intervals counts as unhealthy, so a hung check shows up
STALE_AFTER_INTERVALS: Final[int] = 3
STATUS_HEALTHY: Final[str] = "healthy"
STATUS_OPENSEARCH_UNHEALTHY: Final[str] = "OpenSearch is not healthy"
STATUS_AMQP_DISCONNECTED: Final[str] = "AMQP is not connected"

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


class OpenSearchHealth:
    """Checks OpenSearch health in the background, so the endpoint answers at once and doesn't block the loop."""

    _client: OsClient
    _interval: float
    _healthy: Optional[bool]
    _checked_at: Optional[float]
    _started_at: float
    _task: Optional[asyncio.Task[None]]

    def __init__(
        self,
        app_config: ApplicationConfig,
        *,
        interval: float = OPENSEARCH_CHECK_INTERVAL,
        os_client: Optional[OsClient] = None,
    ) -> None:
        """Create the checker.

        :param app_config: Application configuration
        :param interval: Seconds between checks
        :param os_client: Optional client for tests; by default a client without retries, so a slow OpenSearch is
            reported within the request timeout
        """
        self._client = os_client or OsClient(app_config, max_retries=0)
        self._interval = interval
        self._healthy = None
        self._checked_at = None
        self._started_at = time.monotonic()
        self._task = None

    async def check(self) -> bool:
        """Check OpenSearch once, in a thread of the default executor.

        :return: Whether OpenSearch is healthy
        """
        healthy = await asyncio.to_thread(self._client.is_healthy)
        self._healthy = healthy
        self._checked_at = time.monotonic()
        return healthy

    async def _run(self) -> None:
        while True:
            try:
                await self.check()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("OpenSearch health check failed", exc_info=exc)
                self._healthy = False
                self._checked_at = time.monotonic()
            await asyncio.sleep(self._interval)

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._task = asyncio.create_task(self._run(), name="opensearch-health")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def status(self) -> dict[str, Any]:
        """Last check result and its age.

        :return: "healthy" is True, False, or None while the first check is running and not overdue
        """
        now = time.monotonic()
        age = now - (self._checked_at if self._checked_at is not None else self._started_at)
        healthy = self._healthy
        if age > self._interval * STALE_AFTER_INTERVALS:
            healthy = False
        return {"healthy": healthy, "checked_seconds_ago": round(age, 1) if self._checked_at is not None else None}


@web.middleware
async def cors_middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
    """Allow any origin to read the endpoint, as the previous Flask app did."""
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


class HealthServer:
    """HTTP server with the health endpoint at "/"."""

    _app_config: ApplicationConfig
    _coordinators: list[Coordinator]
    _opensearch: OpenSearchHealth
    _runner: Optional[web.AppRunner]

    def __init__(
        self, app_config: ApplicationConfig, coordinators: list[Coordinator], opensearch: OpenSearchHealth
    ) -> None:
        self._app_config = app_config
        self._coordinators = coordinators
        self._opensearch = opensearch
        self._runner = None

    def make_app(self) -> web.Application:
        """Create the aiohttp application."""
        app = web.Application(middlewares=[cors_middleware])
        app.router.add_get("/", self.get_health_status)
        return app

    async def get_health_status(self, _: web.Request) -> web.Response:
        opensearch = self._opensearch.status()
        coordinators = [coordinator.snapshot() for coordinator in self._coordinators]

        opensearch_healthy = opensearch["healthy"]
        amqp_connected = all(coordinator["connected"] for coordinator in coordinators)
        if not opensearch_healthy or not amqp_connected:
            errors = []
            if not opensearch_healthy:
                errors.append(STATUS_OPENSEARCH_UNHEALTHY)
            if not amqp_connected:
                errors.append(STATUS_AMQP_DISCONNECTED)
            status = ". ".join(errors)
        else:
            status = STATUS_HEALTHY
        body: dict[str, Any] = {
            "status": status,
            "version": self._app_config.appVersion,
            "opensearch": opensearch,
            "coordinators": coordinators,
        }
        if status != STATUS_HEALTHY:
            LOGGER.error(f"Analyzer health check status failed: {status}")
        return web.Response(
            text=json.dumps(body), status=200 if status == STATUS_HEALTHY else 503, content_type="application/json"
        )

    async def start(self) -> None:
        """Start serving on `ANALYZER_HTTP_PORT` and start the OpenSearch checks."""
        self._opensearch.start()
        runner = web.AppRunner(self.make_app(), access_log=None)
        self._runner = runner
        await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=self._app_config.analyzerHttpPort)  # NOSONAR
        await site.start()
        LOGGER.info(f"Started http server on port {self._app_config.analyzerHttpPort}")

    async def stop(self) -> None:
        await self._opensearch.stop()
        if self._runner is not None:
            await self._runner.cleanup()
