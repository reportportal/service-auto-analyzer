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

import re
import string
import nltk
import logging

logger = logging.getLogger("analyzerApp.utils")


def sanitize_text(text):
    """Sanitize text by deleting all numbers"""
    return re.sub(r"\d+", "", text)


def calculate_line_number(text):
    """Calculate line numbers in the text"""
    return len([line for line in text.split("\n") if line.strip() != ""])


def first_lines(log_str, n_lines):
    """Take n first lines"""
    return "\n".join((log_str.split("\n")[:n_lines])) if n_lines >= 0 else log_str


def build_url(main_url, url_params):
    """Build url by concating url and url_params"""
    return main_url + "/" + "/".join(url_params)


def split_words(text, min_word_length=0, only_unique=True, split_urls=True):
    all_words = set() if only_unique else []
    stopwords = set(nltk.corpus.stopwords.words("english"))
    replace_symbols = r"[<>\{:,!?\}\[\];=\(\)\'\"]|\.\.\."
    text = re.sub(replace_symbols, " ", text)
    res = text.split()
    translate_map = {}
    for punct in string.punctuation:
        if punct != "." and (split_urls or punct not in ["/", "\\"]):
            translate_map[punct] = " "
    for word_part in res:
        word_part = re.sub(r"\s+", " ",
                           word_part.translate(word_part.maketrans(translate_map))).strip().lower()
        word_part = re.sub(r"\.+\b|\b\.+", "", word_part)
        for w in word_part.split():
            if w != "" and w not in stopwords and len(w) >= min_word_length and re.search(r"\w", w):
                if only_unique:
                    all_words.add(w)
                else:
                    all_words.append(w)
    return list(all_words)


def find_query_words_count_from_explanation(elastic_res, field_name="message"):
    """Find information about matched words in elasticsearch query"""
    index_query_words_details = None
    all_words = set()
    try:
        for idx, field in enumerate(elastic_res["_explanation"]["details"]):
            if "weight(%s:" % field_name in field["description"].lower():
                word = re.search(r"weight\(%s:(.+) in" % field_name, field["description"]).group(1)
                all_words.add(word)
                break
            for detail in field["details"]:
                if "weight(%s:" % field_name in detail["description"].lower():
                    index_query_words_details = idx
                    break
        if index_query_words_details is not None:
            field_explaination = elastic_res["_explanation"]["details"]
            for detail in field_explaination[index_query_words_details]["details"]:
                word = re.search(r"weight\(%s:(.+) in" % field_name, detail["description"]).group(1)
                all_words.add(word)
    except Exception as e:
        logger.error(e)
        return []
    return list(all_words)


def transform_string_feature_range_into_list(text):
    """Converts features from string to list of ids"""
    values = []
    for part in text.split(","):
        if part.strip() == "":
            continue
        if "-" in part:
            start, end = part.split("-")[:2]
            values.extend(list(range(int(start), int(end) + 1)))
        else:
            values.append(int(part))
    return values
