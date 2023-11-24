#  Copyright 2023 EPAM Systems
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

from unittest import mock

import pytest

from app.service.analyzer_service import AnalyzerService
from test import DEFAULT_SEARCH_CONFIG, DEFAULT_BOOST_LAUNCH

DEFAULT_LAUNCH_NAME = 'Test Launch'
DEFAULT_LAUNCH_ID = 3


def get_empty_bool_query():
    return {
        'query': {
            'bool': {
                'must': [],
                'should': []
            }
        }
    }


DEFAULT_LAUNCH_NAME_SEARCH = {'must': [{'term': {'launch_name': {'value': DEFAULT_LAUNCH_NAME}}}],
                              'should': [{'term': {'launch_id': {'value': DEFAULT_LAUNCH_ID,
                                                                 'boost': DEFAULT_BOOST_LAUNCH}}}]}


@pytest.mark.parametrize(
    'previous_launch_id, launch_mode, expected_query',
    [
        (2, 'LAUNCH_NAME', DEFAULT_LAUNCH_NAME_SEARCH),
        (2, 'CURRENT_AND_THE_SAME_NAME', DEFAULT_LAUNCH_NAME_SEARCH),
        (2, 'CURRENT_LAUNCH', {'must': [{'term': {'launch_id': {'value': DEFAULT_LAUNCH_ID}}}], 'should': []}),
        (2, 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_id': {'value': 2}}}], 'should': []}),
        (None, 'PREVIOUS_LAUNCH', {'must': [], 'should': []}),
        ('3', 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_id': {'value': 3}}},], 'should': []}),
        (2, None, {'must': [], 'should': [{'term': {'launch_name': {'value': DEFAULT_LAUNCH_NAME,
                                                                    'boost': DEFAULT_BOOST_LAUNCH}}}]})
    ]
)
def test_add_constraints_for_launches_into_query(previous_launch_id, launch_mode, expected_query):
    launch = mock.Mock()
    launch.launchId = DEFAULT_LAUNCH_ID
    launch.previousLaunchId = previous_launch_id
    launch.launchName = DEFAULT_LAUNCH_NAME
    launch.analyzerConfig.analyzerMode = launch_mode
    analyzer = AnalyzerService(None, DEFAULT_SEARCH_CONFIG)
    result = analyzer.add_constraints_for_launches_into_query(get_empty_bool_query(), launch)
    assert result['query']['bool'] == expected_query
