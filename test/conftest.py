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

"""Pytest configuration and hooks."""

import inspect


def pytest_pycollect_makeitem(collector, name, obj):
    """
    Hook to prevent pytest from collecting classes that are defined in the app module,
    even when they are imported into test files.
    """
    # Check if this is a class (not a function)
    if inspect.isclass(obj):
        # Check if the class is defined in the app module (not in test module)
        if hasattr(obj, "__module__") and obj.__module__.startswith("app."):
            # Return empty list to skip collection
            return []
    # Return None to let pytest continue with default behavior
    return None
