#  Copyright 2024 EPAM Systems
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

from collections import defaultdict as _defaultdict
from typing import TypeVar, Callable

_KT = TypeVar("_KT")
_RT = TypeVar("_RT")


class DefaultDict(_defaultdict):
    _default_factory: Callable[[_KT], _RT]

    def __init__(self, default_factory: Callable[[_KT], _RT], **kwargs):
        super().__init__(**kwargs)
        self._default_factory = default_factory

    def __missing__(self, key: _KT) -> _RT:
        if self._default_factory is None:
            raise KeyError(key)
        self[key] = value = self._default_factory(key)
        return value