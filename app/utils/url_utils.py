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

"""URL helpers without heavy dependencies, safe to import from the coordinator process."""

import re
from urllib.parse import urlparse


def remove_credentials_from_url(url: str) -> str:
    """Remove the user and password part from a URL, to make it safe for logging.

    :param url: URL which may contain credentials
    :return: The URL without credentials
    """
    parsed_url = urlparse(url)
    new_netloc = re.sub("^[^:]+:[^@]*@", "", parsed_url.netloc)
    if parsed_url.netloc == new_netloc:
        return url
    return url.replace(parsed_url.netloc, new_netloc)
