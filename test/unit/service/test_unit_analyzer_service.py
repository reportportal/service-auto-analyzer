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

DEFAULT_LAUNCH_NAME_SEARCH = {'must': [{'term': {'launch_name': DEFAULT_LAUNCH_NAME}}],
                              'should': [{'term': {'launch_id': {'value': DEFAULT_LAUNCH_ID,
                                                                 'boost': DEFAULT_BOOST_LAUNCH}}}]}
DEFAULT_LAUNCH_BOOST = {'should': [
    {'term': {'launch_id': {'value': 3, 'boost': DEFAULT_BOOST_LAUNCH}}},
    {'term': {'launch_name': {'value': DEFAULT_LAUNCH_NAME, 'boost': DEFAULT_BOOST_LAUNCH}}}
]}


@pytest.mark.parametrize(
    'previous_launch_id, launch_mode, expected_query',
    [
        (2, 'LAUNCH_NAME', {'must': [{'term': {'launch_name': DEFAULT_LAUNCH_NAME}}],
                            'must_not': [{'term': {'launch_id': DEFAULT_LAUNCH_ID}}]}),
        (2, 'CURRENT_AND_THE_SAME_NAME', DEFAULT_LAUNCH_NAME_SEARCH),
        (2, 'CURRENT_LAUNCH', {'must': [{'term': {'launch_id': DEFAULT_LAUNCH_ID}}]}),
        (2, 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_id': 2}}]}),
        (None, 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_id': 0}}]}),
        ('3', 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_id': 3}}]}),
        (2, None, DEFAULT_LAUNCH_BOOST),
        (2, 'ALL', {'must_not': [{'term': {'launch_id': DEFAULT_LAUNCH_ID}}]})
    ]
)
def test_add_constraints_for_launches_into_query(previous_launch_id, launch_mode, expected_query):
    launch = mock.Mock()
    launch.launchId = DEFAULT_LAUNCH_ID
    launch.previousLaunchId = previous_launch_id
    launch.launchName = DEFAULT_LAUNCH_NAME
    launch.analyzerConfig.analyzerMode = launch_mode
    analyzer = AnalyzerService(None, DEFAULT_SEARCH_CONFIG)
    result = analyzer.add_constraints_for_launches_into_query({'query': {'bool': {}}}, launch)
    assert result['query']['bool'] == expected_query


DEFAULT_LAUNCH_BOOST_SUGGEST = {'should': [
    {'term': {'launch_name': {'value': DEFAULT_LAUNCH_NAME, 'boost': DEFAULT_BOOST_LAUNCH}}},
    {'term': {'launch_id': {'value': DEFAULT_LAUNCH_ID, 'boost': 1 / DEFAULT_BOOST_LAUNCH}}}
]}


@pytest.mark.parametrize(
    'previous_launch_id, launch_mode, expected_query',
    [
        (2, 'LAUNCH_NAME', DEFAULT_LAUNCH_BOOST_SUGGEST),
        (2, 'CURRENT_AND_THE_SAME_NAME', DEFAULT_LAUNCH_BOOST),
        (2, 'CURRENT_LAUNCH', DEFAULT_LAUNCH_BOOST),
        (2, 'PREVIOUS_LAUNCH', {'should': [{'term': {'launch_id': {'value': 2, 'boost': DEFAULT_BOOST_LAUNCH}}}]}),
        (None, 'PREVIOUS_LAUNCH', {}),
        ('2', 'PREVIOUS_LAUNCH', {'should': [{'term': {'launch_id': {'value': 2, 'boost': DEFAULT_BOOST_LAUNCH}}}]}),
        (2, 'ALL', DEFAULT_LAUNCH_BOOST_SUGGEST)
    ]
)
def test_add_constraints_for_launches_into_query_suggest(previous_launch_id, launch_mode, expected_query):
    launch = mock.Mock()
    launch.launchId = DEFAULT_LAUNCH_ID
    launch.previousLaunchId = previous_launch_id
    launch.launchName = DEFAULT_LAUNCH_NAME
    launch.analyzerConfig.analyzerMode = launch_mode
    analyzer = AnalyzerService(None, DEFAULT_SEARCH_CONFIG)
    result = analyzer.add_constraints_for_launches_into_query_suggest({'query': {'bool': {}}}, launch)
    assert result['query']['bool'] == expected_query
