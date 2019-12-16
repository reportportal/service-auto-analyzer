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


def split_words(text, min_word_length=0):
    """Splits words by space and punctuation marks"""
    all_words = set()
    stopwords = set(nltk.corpus.stopwords.words("english"))
    replace_symbols = r"[<>\{:,!?\}\[\];=\(\)\'\"]|\.\.\."
    text = re.sub(replace_symbols, " ", text)
    res = text.split()
    translate_map = {}
    for punct in string.punctuation:
        translate_map[punct] = " "
    for w in res:
        w = re.sub(r"\s+", " ", w.translate(w.maketrans(translate_map))).strip().lower()
        if w != "" and w not in stopwords and len(w) >= min_word_length and re.search(r"\w", w):
            all_words.add(w)
    return list(all_words)


def find_query_words_count_from_explanation(elastic_res):
    """Find information about matched words in elasticsearch query"""
    index_query_words_details = None
    all_words = set()
    try:
        for idx, field in enumerate(elastic_res["_explanation"]["details"]):
            for detail in field["details"]:
                if "weight(message:" in detail["description"].lower():
                    index_query_words_details = idx
                    break
        for detail in elastic_res["_explanation"]["details"][index_query_words_details]["details"]:
            word = re.search(r"weight\(message:(.+) in", detail["description"]).group(1)
            all_words.add(word)
    except Exception as err:
        logger.error(err)
        return []
    return list(all_words)
