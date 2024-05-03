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
from utils import text_processing


def basic_prepare(message):
    cleaned_message = message.strip()
    cleaned_message = text_processing.unify_line_endings(cleaned_message)
    cleaned_message = text_processing.remove_starting_datetime(cleaned_message)
    cleaned_message = text_processing.remove_starting_log_level(cleaned_message)
    cleaned_message = text_processing.remove_starting_datetime(cleaned_message)  # Sometimes log level goes first
    cleaned_message = text_processing.replace_tabs_for_newlines(cleaned_message)
    cleaned_message = text_processing.fix_big_encoded_urls(cleaned_message)
    cleaned_message = text_processing.remove_generated_parts(cleaned_message)
    cleaned_message = text_processing.remove_guid_uuids_from_text(cleaned_message)
    cleaned_message = text_processing.clean_html(cleaned_message)
    cleaned_message = text_processing.delete_empty_lines(cleaned_message)
    cleaned_message = text_processing.leave_only_unique_lines(cleaned_message)
    return cleaned_message
