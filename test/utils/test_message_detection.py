"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import unittest
import logging
from app.utils import utils


class TestMessageDetection(unittest.TestCase):
    """Tests detecting messages in logs"""
    @utils.ignore_warnings
    def setUp(self):
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def test_detecting_messages(self):
        example_logs = utils.get_fixture("example_logs.json", to_json=True)
        for idx, example in enumerate(example_logs):
            det_message, stacktrace =\
                utils.detect_log_description_and_stacktrace(example["log"])

            try:
                assert det_message == example["detected_message"]
                assert stacktrace == example["stacktrace"]
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}').\
                    with_traceback(err.__traceback__)
