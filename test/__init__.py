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

import os
import random
import string
from typing import List

from app.commons.launch_objects import SearchConfig, ApplicationConfig
from app.utils.utils import read_json_file

DEFAULT_ES_CONFIG = {'esHost': 'http://localhost:9200', 'esVerifyCerts': False, 'esUseSsl': False,
                     'esSslShowWarn': False, 'esCAcert': None, 'esClientCert': None, 'esClientKey': None,
                     'esUser': None, 'turnOffSslVerification': True, 'esProjectIndexPrefix': '', 'esChunkNumber': 1000}
DEFAULT_BOOST_LAUNCH = 8.0
DEFAULT_SEARCH_CONFIG = SearchConfig(BoostLaunch=DEFAULT_BOOST_LAUNCH)

APP_CONFIG = ApplicationConfig(
    esHost="http://localhost:9200",
    esUser="",
    esPassword="",
    esVerifyCerts=False,
    esUseSsl=False,
    esSslShowWarn=False,
    turnOffSslVerification=True,
    esCAcert="",
    esClientCert="",
    esClientKey="",
    appVersion="",
    minioRegion="",
    minioBucketPrefix="",
    filesystemDefaultPath="",
    esChunkNumber=1000,
    binaryStoreType="filesystem",
    minioHost="",
    minioAccessKey="",
    minioSecretKey="",
    esProjectIndexPrefix="rp_"
)


def get_fixture(fixture_name, to_json=False):
    return read_json_file("test_res/fixtures", fixture_name, to_json)


def read_file_lines(folder: str, filename: str) -> List[str]:
    with open(os.path.join(folder, filename), "r") as file:
        return file.readlines()


def random_alphanumeric(num: int):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=num))
