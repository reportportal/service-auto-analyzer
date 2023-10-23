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

DEFAULT_ES_CONFIG = {'esHost': 'http://localhost:9200', 'esVerifyCerts': False, 'esUseSsl': False,
                     'esSslShowWarn': False, 'esCAcert': None, 'esClientCert': None, 'esClientKey': None,
                     'esUser': None, 'turnOffSslVerification': True}

DEFAULT_BOOST_LAUNCH = 8.0
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


DEFAULT_LAUNCH_NAME_SEARCH = {'must': [{'term': {'launch_name': {'value': DEFAULT_LAUNCH_NAME}}}], 'should': []}


@pytest.mark.parametrize(
    'launch_number, launch_mode, expected_query',
    [
        (2, 'LAUNCH_NAME', DEFAULT_LAUNCH_NAME_SEARCH),
        (2, 'CURRENT_AND_THE_SAME_NAME', DEFAULT_LAUNCH_NAME_SEARCH),
        (2, 'CURRENT_LAUNCH', {'must': [{'term': {'launch_id': {'value': DEFAULT_LAUNCH_ID}}}], 'should': []}),
        (2, 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_number': {'value': 1}}}], 'should': []}),
        (None, 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_number': {'value': -1}}}], 'should': []}),
        ('3', 'PREVIOUS_LAUNCH', {'must': [{'term': {'launch_number': {'value': 2}}}], 'should': []}),
        (2, None, {'must': [], 'should': [{'term': {'launch_name': {'value': DEFAULT_LAUNCH_NAME,
                                                                    'boost': DEFAULT_BOOST_LAUNCH}}}]})
    ]
)
def test_add_constraints_for_launches_into_query(launch_number, launch_mode, expected_query):
    launch = mock.Mock()
    launch.launchId = DEFAULT_LAUNCH_ID
    launch.launchNumber = launch_number
    launch.launchName = DEFAULT_LAUNCH_NAME
    launch.analyzerConfig.analyzerMode = launch_mode
    analyzer = AnalyzerService(None, DEFAULT_ES_CONFIG, {'SimilarityWeightsFolder': '',
                                                         'BoostLaunch': DEFAULT_BOOST_LAUNCH})
    result = analyzer.add_constraints_for_launches_into_query(get_empty_bool_query(), launch)
    assert result['query']['bool'] == expected_query
