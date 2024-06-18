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

import pytest
from test import read_file
from app.commons.prepared_log import PreparedLogMessage


@pytest.mark.parametrize(
    'test_file, expected_file',
    [
        ('stacktraces/log_stacktrace_js.txt',
         'stacktraces/log_stacktrace_js_exception_message_no_params_and_brackets.txt'),
    ]
)
def test_exception_message_no_params_and_brackets(test_file, expected_file):
    log = read_file('test_res/test_logs', test_file)
    expected_log = read_file('test_res/test_logs', expected_file)
    assert PreparedLogMessage(log, -1).exception_message_no_params_and_brackets == expected_log.strip()
