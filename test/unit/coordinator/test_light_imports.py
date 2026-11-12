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

"""The coordinator process must not load the ML libraries or the services: they cost about 150 MB of RAM."""

import json
import subprocess
import sys

HEAVY_MODULES = ["sklearn", "nltk", "scipy", "numpy", "xgboost", "gensim", "app.service"]
COORDINATOR_MODULES = [
    "app.main",
    "app.amqp.async_amqp",
    "app.commons.os_client",
    "app.commons.processing",
    "app.coordinator.coordinator",
    "app.coordinator.health",
    "app.coordinator.pool",
]


def test_coordinator_imports_are_light():
    imports = "; ".join(f"import {module}" for module in COORDINATOR_MODULES)
    code = f"import json, sys; {imports}; print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
