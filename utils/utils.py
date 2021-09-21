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
from dateutil.parser import parse
import urllib
from urllib.parse import urlparse
import warnings
import os
import json
import requests
import commons
from collections import Counter
import random
import numpy as np
import traceback

logger = logging.getLogger("analyzerApp.utils")
file_extensions = ["java", "php", "cpp", "cs", "c", "h", "js", "swift", "rb", "py", "scala"]
stopwords = set(nltk.corpus.stopwords.words("english"))
ERROR_LOGGING_LEVEL = 40000


def ignore_warnings(method):
    """Decorator for ignoring warnings"""
    def _inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method(*args, **kwargs)
        return result
    return _inner


def compress(text):
    """compress sentence to consist of only unique words"""
    return " ".join(split_words(text, only_unique=True))


def sanitize_text(text):
    """Sanitize text by deleting all numbers"""
    return re.sub(r"\d+", "", text)


def calculate_line_number(text):
    """Calculate line numbers in the text"""
    return len([line for line in text.split("\n") if line.strip()])


def first_lines(log_str, n_lines):
    """Take n first lines"""
    return "\n".join((log_str.split("\n")[:n_lines])) if n_lines >= 0 else log_str


def build_url(main_url, url_params):
    """Build url by concating url and url_params"""
    return main_url + "/" + "/".join(url_params)


def filter_empty_lines(log_lines):
    return [line for line in log_lines if line.strip()]


def delete_empty_lines(log):
    """Delete empty lines"""
    return "\n".join(filter_empty_lines(log.split("\n")))


def reverse_log(log):
    """Concatenates lines in reverse order"""
    return "\n".join(log.split("\n")[::-1])


def split_words(text, min_word_length=0, only_unique=True, split_urls=True, to_lower=True):
    all_unique_words = set()
    all_words = []
    translate_map = {}
    for punct in string.punctuation + "<>{}[];=()'\"":
        if punct != "." and (split_urls or punct not in ["/", "\\"]):
            translate_map[punct] = " "
    text = text.translate(text.maketrans(translate_map)).strip().strip(".")
    for word_part in text.split():
        word_part = word_part.strip().strip(".")
        for w in word_part.split():
            if to_lower:
                w = w.lower()
            if w != "" and len(w) >= min_word_length:
                if w in stopwords:
                    continue
                if not only_unique or w not in all_unique_words:
                    all_unique_words.add(w)
                    all_words.append(w)
    return all_words


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


def remove_starting_datetime(text, remove_first_digits=False):
    """Removes datetime at the beginning of the text"""
    log_date = ""
    idx_text_start = 0
    for idx, str_part in enumerate(text.split(" ")):
        try:
            parsed_info = re.sub(r"[\[\]\{\},;!#\"$%&\'\(\)*<=>?@^_`|~]", "", log_date + " " + str_part)
            parse(parsed_info)
            log_date = parsed_info
            log_date = log_date.strip()
        except Exception as e: # noqa
            idx_text_start = idx
            break
    log_date = log_date.replace("'", "").replace("\"", "")
    found_regex_log_date = re.search(r"\d{1,7}", log_date)
    if found_regex_log_date and found_regex_log_date.group(0) == log_date:
        idx_text_start = 0

    text_split = text.split(" ")
    if remove_first_digits:
        if idx_text_start == 0:
            for idx in range(len(text_split)):
                rs = text_split[idx].translate(text_split[idx].maketrans("", "", string.punctuation))
                if not re.search(r"\d+", rs.strip()):
                    idx_text_start = idx
                    break

    return " ".join(text_split[idx_text_start:])


def is_starting_message_pattern(text):
    processed_text = text
    res = re.search(r"\w*\s*\(\s*.*\.%s:\d+\s*\)" % "|".join(file_extensions), processed_text)
    if res and processed_text.startswith(res.group(0)):
        return True
    return False


def does_stacktrace_need_words_reweighting(log):
    found_file_extensions = []
    all_extensions_to_find = "|".join(
        ["py", "java", "php", "cpp", "cs", "c", "h", "js", "swift", "rb", "scala"])
    for m in re.findall(r"\.(%s)(?!\.)\b" % all_extensions_to_find, log):
        found_file_extensions.append(m)
    if len(found_file_extensions) == 1 and found_file_extensions[0] in ["js", "c", "h", "rb", "cpp"]:
        return True
    return False


def is_line_from_stacktrace(text):
    """Deletes line numbers in the stacktrace"""
    if is_starting_message_pattern(text):
        return False

    text = re.sub(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)", "", text)
    res = re.sub(r"(?<=:)\d+(?=\)?\]?(\n|\r\n|$))", " ", text)
    if res != text:
        return True
    res = re.sub(r"((?<=line )|(?<=line))\s*\d+\s*((?=, in)|(?=,in)|(?=\n)|(?=\r\n)|(?=$))",
                 " ", res, flags=re.I)
    if res != text:
        return True
    res = re.sub("|".join([r"\.%s(?!\.)\b" % ext for ext in file_extensions]), " ", res, flags=re.I)
    if res != text:
        return True
    result = re.search(r"^\s*at\s+.*\(.*?\)[\s]*$", res)
    if result and result.group(0) == res:
        return True
    else:
        result = re.search(r"^\s*\w+([\.\/]\s*\w+)+\s*\(.*?\)[\s]*$", res)
        if result and result.group(0) == res:
            return True
    return False


def find_only_numbers(detected_message_with_numbers):
    """Removes all non digit symbols and concatenates unique numbers"""
    detected_message_only_numbers = re.sub(r"[^\d \._]", "", detected_message_with_numbers)
    return " ".join(split_words(detected_message_only_numbers, only_unique=True))


def is_python_log(log):
    """Tries to find whether a log was for the python language"""
    found_file_extensions = []
    for m in re.findall(r"\.(%s)(?!\.)\b" % "|".join(file_extensions), log):
        found_file_extensions.append(m)
    found_file_extensions = list(set(found_file_extensions))
    if len(found_file_extensions) == 1 and found_file_extensions[0] == "py":
        return True
    return False


def has_stacktrace_keywords(line):
    normalized_line = line.lower()
    for key_word in ["stacktrace", "stack trace", "stack-trace", "traceback"]:
        if re.search(r"\s*%s\s*:\s*$" % key_word, normalized_line):
            return True
        if "end of " in normalized_line and key_word in normalized_line:
            return True
    return False


def has_more_lines_pattern(line):
    normalized_line = line.lower().strip()
    result = re.search(r"^\s*\.+\s*\d+\s+more\s*$", normalized_line)
    if result and result.group(0) == normalized_line:
        return True
    return False


def detect_log_parts_python(message, default_log_number=1):
    detected_message_lines = []
    stacktrace_lines = []
    traceback_begin = False
    detected_message_begin = True
    skip_exceptions_finding = False
    for idx, line in enumerate(message.split("\n")):
        for key_word in ["stacktrace", "stack trace", "stack-trace", "traceback", "trace back"]:
            if key_word in line.lower():
                traceback_begin = True
                detected_message_begin = False
                skip_exceptions_finding = True
                break

        if not skip_exceptions_finding and get_found_exceptions(line):
            detected_message_begin = True
            traceback_begin = False
        if traceback_begin:
            stacktrace_lines.append(line)
        elif detected_message_begin:
            detected_message_lines.append(line)
        skip_exceptions_finding = False
    if len(detected_message_lines) == 0:
        detected_message_lines = stacktrace_lines[-default_log_number:]
        stacktrace_lines = stacktrace_lines[:-default_log_number]
    return "\n".join(detected_message_lines), "\n".join(stacktrace_lines)


def detect_log_description_and_stacktrace(message):
    """Split a log into a log message and stacktrace"""
    message = remove_starting_datetime(message)
    message = delete_empty_lines(message)
    if calculate_line_number(message) > 2:
        if is_python_log(message):
            return detect_log_parts_python(message)
        split_lines = message.split("\n")
        detected_message_lines = []
        stacktrace_lines = []
        for idx, line in enumerate(split_lines):
            if is_line_from_stacktrace(line):
                stacktrace_lines.append(line)
            else:
                detected_message_lines.append(line)

        if not detected_message_lines:
            detected_message_lines = stacktrace_lines[:1]
            stacktrace_lines = stacktrace_lines[1:]

        return "\n".join(detected_message_lines), "\n".join(stacktrace_lines)
    return message, ""


def detect_log_description_and_stacktrace_light(message):
    """Split a log into a log message and stacktrace in a light way"""
    message = delete_empty_lines(message)
    if calculate_line_number(message) > 2:
        if is_python_log(message):
            return detect_log_parts_python(message)
        split_lines = message.split("\n")
        stacktrace_start_idx = len(split_lines)
        for idx, line in enumerate(split_lines):
            if is_line_from_stacktrace(line):
                stacktrace_start_idx = idx
                break

        if stacktrace_start_idx == 0:
            stacktrace_start_idx = 1

        return "\n".join(
            split_lines[:stacktrace_start_idx]), "\n".join(
            split_lines[stacktrace_start_idx:])
    return message, ""


def fix_big_encoded_urls(message):
    """Decodes urls encoded with %12 and etc. and removes brackets to separate url"""
    try:
        new_message = urllib.parse.unquote(message)
    except: # noqa
        pass
    if new_message != message:
        return re.sub(r"[\(\)\{\}#%]", " ", new_message)
    return message


def leave_only_unique_lines(message):
    all_unique = set()
    all_lines = []
    for idx, line in enumerate(message.split("\n")):
        # To remove lines with 'For documentation on this error please visit ...url'
        if "documentation" in line.lower() and "error" in line.lower() and "visit" in line.lower():
            continue
        if line.strip() not in all_unique:
            all_unique.add(line.strip())
            all_lines.append(line)
    return "\n".join(all_lines)


def remove_generated_parts(message):
    """Removes lines with '<generated>' keyword and removes parts, like $ab24b, @c321e from words"""
    all_lines = []
    for line in message.split("\n"):
        if "<generated>" in line.lower():
            continue
        if has_stacktrace_keywords(line) or has_more_lines_pattern(line):
            continue
        for symbol in [r"\$", "@"]:
            all_found_parts = set()
            for m in re.finditer(r"%s+(.+?)\b" % symbol, line):
                try:
                    found_part = m.group(1).strip().strip(symbol).strip()
                    if found_part != "":
                        all_found_parts.add((found_part, m.group(0).strip()))
                except Exception as err:
                    logger.error(err)
            sorted_parts = sorted(list(all_found_parts), key=lambda x: len(x[1]), reverse=True)
            for found_part in sorted_parts:
                whole_found_part = found_part[1].replace("$", r"\$")
                found_part = found_part[0]
                part_to_replace = ""
                if re.search(r"\d", found_part):
                    part_with_numbers_in_the_end = re.search(r"[a-zA-z]{5,}\d+", found_part)
                    if part_with_numbers_in_the_end and part_with_numbers_in_the_end.group(0) == found_part:
                        part_to_replace = " %s" % found_part
                    else:
                        part_to_replace = ""
                else:
                    part_to_replace = ".%s" % found_part
                try:
                    line = re.sub(whole_found_part, part_to_replace, line)
                except: # noqa
                    pass

        line = re.sub(r"\.+", ".", line)
        all_lines.append(line)
    return "\n".join(all_lines)


def clean_text_from_html_tags(message):
    """Removes style and script tags together with inner text and removes html tags"""
    regex_style_tag = re.compile('<style.*?>[\\s\\S]*?</style>')
    message = re.sub(regex_style_tag, " ", message)
    regex_script_tag = re.compile('<script.*?>[\\s\\S]*?</script>')
    message = re.sub(regex_script_tag, " ", message)
    regex_html_tags = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    message = re.sub(regex_html_tags, " ", message)
    return message


def clean_html(message):
    """Removes html tags inside the parts with <.*?html.*?>...</html>"""
    all_lines = []
    started_html = False
    finished_with_html_tag = False
    html_part = []
    for idx, line in enumerate(message.split("\n")):
        if re.search(r"<.*?html.*?>", line):
            started_html = True
            html_part.append(line)
        else:
            if started_html:
                html_part.append(line)
            else:
                all_lines.append(line)
        if "</html>" in line:
            finished_with_html_tag = True
        if finished_with_html_tag:
            all_lines.append(clean_text_from_html_tags("\n".join(html_part)))
            html_part = []
            finished_with_html_tag = False
            started_html = False
    if len(html_part) > 0:
        all_lines.extend(html_part)
    return delete_empty_lines("\n".join(all_lines))


def replace_tabs_for_newlines(message):
    return message.replace("\t", "\n")


def remove_credentials_from_url(url):
    parsed_url = urlparse(url)
    new_netloc = re.sub("^.+?:.+?@", "", parsed_url.netloc)
    return url.replace(parsed_url.netloc, new_netloc)


def clean_colon_stacking(text):
    return text.replace(":", " : ")


def leave_only_unique_logs(logs):
    unique_logs = set()
    all_logs = []
    for log in logs:
        if log.message.strip() not in unique_logs:
            all_logs.append(log)
            unique_logs.add(log.message.strip())
    return all_logs


def read_json_file(folder, filename, to_json=False):
    """Read fixture from file"""
    with open(os.path.join(folder, filename), "r") as file:
        return file.read() if not to_json else json.loads(file.read())


def get_found_exceptions(text, to_lower=False):
    """Extract exception and errors from logs"""
    unique_exceptions = set()
    found_exceptions = []
    for word in split_words(text, to_lower=to_lower):
        for key_word in ["error", "exception", "failure"]:
            if re.search(r"[^\s]{3,}%s(\s|$)" % key_word, word.lower()) is not None:
                if word not in unique_exceptions:
                    found_exceptions.append(word)
                    unique_exceptions.add(word)
                break
    return " ".join(found_exceptions)


def clean_from_params(text):
    text = re.sub(r"(?<=[^\w])('.+?'|\".+?\")(?=[^\w]|$)|(?<=^)('.+?'|\".+?\")(?=[^\w]|$)", " ", text)
    return re.sub(r" +", " ", text).strip()


def clean_from_urls(text):
    text = re.sub(r"(http|https|ftp):[^\s]+|\bwww\.[^\s]+", " ", text)
    return re.sub(r" +", " ", text).strip()


def clean_from_paths(text):
    text = re.sub(r"(^|(?<=[^\w:\\\/]))(\w:)?([\w\d\.\-_]+)?([\\\/]+[\w\d\.\-_]+){2,}", " ", text)
    return re.sub(r" +", " ", text).strip()


def extract_urls(text):
    all_unique = set()
    all_urls = []
    for param in re.findall(r"((http|https|ftp):[^\s]+|\bwww\.[^\s]+)", text):
        url = param[0].strip()
        if url not in all_unique:
            all_unique.add(url)
            all_urls.append(url)
    return all_urls


def extract_paths(text):
    all_unique = set()
    all_paths = []
    for param in re.findall(r"((^|(?<=[^\w:\\\/]))(\w:)?([\w\d\.\-_ ]+)?([\\\/]+[\w\d\.\-_ ]+){2,})", text):
        path = param[0].strip()
        if path not in all_unique:
            all_unique.add(path)
            all_paths.append(path)
    return all_paths


def extract_message_params(text):
    all_unique = set()
    all_params = []
    for param in re.findall(r"(^|[^\w])('.+?'|\".+?\")([^\w]|$|\n)", text):
        param = re.search(r"[^\'\"]+", param[1].strip())
        if param is not None:
            param = param.group(0).strip()
            if param not in all_unique:
                all_unique.add(param)
                all_params.append(param)
    return all_params


def enrich_found_exceptions(text):
    unique_words = set()
    new_words = []
    for word in text.split(" "):
        word_parts = word.split(".")
        new_words.append(word)
        unique_words.add(word)
        for i in [2, 1]:
            new_word_part = ".".join(word_parts[-i:])
            if new_word_part not in unique_words:
                new_words.append(new_word_part)
                unique_words.add(new_word_part)
    return " ".join(new_words)


def enrich_text_with_method_and_classes(text):
    new_lines = []
    for line in text.split("\n"):
        new_line = line
        found_values = []
        for w in split_words(line, min_word_length=0, only_unique=True, split_urls=True, to_lower=False):
            if len(w.split(".")) > 2:
                last_word = w.split(".")[-1]
                if len(last_word) > 3:
                    found_values.append(w)
        for val in sorted(found_values, key=lambda x: len(x.split(".")), reverse=False):
            words = val.split(".")
            full_path = val
            for i in [2, 1]:
                full_path = full_path + " " + ".".join(words[-i:])
            full_path = full_path + " "
            new_line = re.sub(r"\b(?<!\.)%s(?!\.)\b" % val, full_path, new_line)
        new_lines.append(new_line)
    return "\n".join(new_lines)


def extract_real_id(elastic_id):
    real_id = str(elastic_id)
    if real_id[-2:] == "_m":
        return int(real_id[:-2])
    return int(real_id)


def jaccard_similarity(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0


def get_fixture(fixture_name, to_json=False):
    with open(os.path.join("fixtures", fixture_name), "r") as file:
        return json.loads(file.read()) if to_json else file.read()


def clean_from_brackets(text):
    for pattern in [r"\[[\s\S]+\]", r"\{[\s\S]+?\}", r"\([\s\S]+?\)"]:
        text = re.sub(pattern, "", text)
    return text


def get_potential_status_codes(text):
    potential_codes = set()
    for line in text.split("\n"):
        line = clean_from_brackets(line)
        patterns_to_check = [r"\bcode[^\w\d\.]+(\d+)[^\d;\.]*(\d*)|\bcode[^\w\d\.]+(\d+?)$",
                             r"\w+_code[^\w\d\.]+(\d+)[^\d;\.]*(\d*)|\w+_code[^\w\d\.]+(\d+?)$"]
        for pattern in patterns_to_check:
            result = re.search(pattern, line, flags=re.IGNORECASE)
            for i in range(1, 4):
                try:
                    found_code = result.group(i)
                    if found_code and found_code.strip():
                        potential_codes.add(found_code)
                except: # noqa
                    pass
    return list(potential_codes)


def choose_issue_type(predicted_labels, predicted_labels_probability,
                      issue_type_names, scores_by_issue_type):
    predicted_issue_type = ""
    max_prob = 0.0
    max_val_start_time = None
    global_idx = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1:
            issue_type = issue_type_names[i]
            chosen_type = scores_by_issue_type[issue_type]
            start_time = chosen_type["mrHit"]["_source"]["start_time"]
            predicted_prob = round(predicted_labels_probability[i][1], 2)
            if (predicted_prob > max_prob) or\
                    ((predicted_prob == max_prob) and # noqa
                        (max_val_start_time is None or start_time > max_val_start_time)):
                max_prob = predicted_prob
                predicted_issue_type = issue_type
                global_idx = i
                max_val_start_time = start_time
    return predicted_issue_type, max_prob, global_idx


def prepare_message_for_clustering(message, number_of_log_lines):
    message = first_lines(message, number_of_log_lines)
    words = split_words(message, min_word_length=2, only_unique=False)
    if len(words) < 2:
        return ""
    return " ".join(words)


def send_request(url, method, username, password):
    """Send request with specified url and http method"""
    try:
        if username.strip() and password.strip():
            response = requests.get(url, auth=(username, password)) if method == "GET" else {}
        else:
            response = requests.get(url) if method == "GET" else {}
        data = response._content.decode("utf-8")
        content = json.loads(data, strict=False)
        return content
    except Exception as err:
        logger.error("Error with loading url: %s",
                     remove_credentials_from_url(url))
        logger.error(err)
    return []


def extract_all_exceptions(bodies):
    logs_with_exceptions = []
    for log_body in bodies:
        exceptions = [
            exc.strip() for exc in log_body["_source"]["found_exceptions"].split()]
        logs_with_exceptions.append(
            commons.launch_objects.LogExceptionResult(
                logId=int(log_body["_id"]),
                foundExceptions=exceptions))
    return logs_with_exceptions


def calculate_proportions_for_labels(labels):
    counted_labels = Counter(labels)
    if len(counted_labels.keys()) >= 2:
        min_val = min(counted_labels.values())
        max_val = max(counted_labels.values())
        if max_val > 0:
            return np.round(min_val / max_val, 3)
    return 0.0


def rebalance_data(train_data, train_labels, due_proportion):
    one_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 1]
    zero_data = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 0]
    zero_count = len(zero_data)
    one_count = len(one_data)
    all_data = []
    all_data_labels = []
    real_proportion = 0.0 if zero_count < 0.001 else np.round(one_count / zero_count, 3)
    if zero_count > 0 and real_proportion < due_proportion:
        all_data.extend(one_data)
        all_data_labels.extend([1] * len(one_data))
        random.seed(1763)
        random.shuffle(zero_data)
        zero_size = int(one_count * (1 / due_proportion) - 1)
        all_data.extend(zero_data[:zero_size])
        all_data_labels.extend([0] * zero_size)
        real_proportion = calculate_proportions_for_labels(all_data_labels)
        if real_proportion / due_proportion >= 0.9:
            real_proportion = due_proportion

    random.seed(1257)
    random.shuffle(all_data)
    random.seed(1257)
    random.shuffle(all_data_labels)
    return all_data, all_data_labels, real_proportion


def preprocess_words(text):
    all_words = []
    for w in re.finditer(r"[\w\._]+", text):
        word_normalized = re.sub(r"^[\w]\.", "", w.group(0))
        word = word_normalized.replace("_", "")
        if len(word) >= 3:
            all_words.append(word.lower())
        split_parts = word_normalized.split("_")
        split_words = []
        if len(split_parts) > 2:
            for idx in range(len(split_parts)):
                if idx != len(split_parts) - 1:
                    split_words.append("".join(split_parts[idx:idx + 2]).lower())
            all_words.extend(split_words)
        if "." not in word_normalized:
            split_words = []
            split_parts = [s.strip() for s in re.split("([A-Z][^A-Z]+)", word) if s.strip()]
            if len(split_parts) > 2:
                for idx in range(len(split_parts)):
                    if idx != len(split_parts) - 1:
                        if len("".join(split_parts[idx:idx + 2]).lower()) > 3:
                            split_words.append("".join(split_parts[idx:idx + 2]).lower())
            all_words.extend(split_words)
    return all_words


def topological_sort(feature_graph):
    visited = {}
    for key_ in feature_graph:
        visited[key_] = 0
    stack = []

    for key_ in feature_graph:
        if visited[key_] == 0:
            stack_vertices = [key_]
            while len(stack_vertices):
                vert = stack_vertices[-1]
                if vert not in visited:
                    continue
                if visited[vert] == 1:
                    stack_vertices.pop()
                    visited[vert] = 2
                    stack.append(vert)
                else:
                    visited[vert] = 1
                    for key_i in feature_graph[vert]:
                        if key_i not in visited:
                            continue
                        if visited[key_i] == 0:
                            stack_vertices.append(key_i)
    return stack


def to_number_list(features_list):
    feature_numbers_list = []
    for feature_name in features_list.split(";"):
        feature_name = feature_name.split("_")[0]
        try:
            feature_numbers_list.append(int(feature_name))
        except: # noqa
            try:
                feature_numbers_list.append(float(feature_name))
            except: # noqa
                pass
    return feature_numbers_list


def fill_prevously_gathered_features(feature_list, feature_ids):
    previously_gathered_features = {}
    try:
        if type(feature_ids) == str:
            feature_ids = to_number_list(feature_ids)
        for i in range(len(feature_list)):
            for idx, feature in enumerate(feature_ids):
                if feature not in previously_gathered_features:
                    previously_gathered_features[feature] = []
                if len(previously_gathered_features[feature]) <= i:
                    previously_gathered_features[feature].append([])
                previously_gathered_features[feature][i].append(feature_list[i][idx])
    except Exception as err:
        logger.error(err)
    return previously_gathered_features


def gather_feature_list(gathered_data_dict, feature_ids, to_list=False):
    features_array = None
    axis_x_size = max(map(lambda x: len(x), gathered_data_dict.values()))
    if axis_x_size <= 0:
        return []
    for idx, feature in enumerate(feature_ids):
        if feature not in gathered_data_dict or len(gathered_data_dict[feature]) == 0:
            gathered_data_dict[feature] = [[0.0] for i in range(axis_x_size)]
        if features_array is None:
            features_array = np.asarray(gathered_data_dict[feature])
        else:
            features_array = np.concatenate([features_array, gathered_data_dict[feature]], axis=1)
    return features_array.tolist() if to_list else features_array


def unite_project_name(project_id, prefix):
    return prefix + project_id


def get_project_id(full_project, prefix):
    return re.sub("^%s" % prefix, "", full_project) if prefix.strip() else full_project


def extract_exception(err):
    err_message = traceback.format_exception_only(type(err), err)
    if len(err_message):
        err_message = err_message[-1]
    else:
        err_message = ""
    return err_message


def find_test_methods_in_text(text):
    test_methods = set()
    for m in re.findall(
            r"([^ \(\)\/\\\\:]+(Test|Step)[s]*\.[^ \(\)\/\\\\:]+)|([^ \(\)\/\\\\:]+\.spec\.js)", text):
        if m[0].strip():
            test_methods.add(m[0].strip())
        if m[2].strip():
            test_methods.add(m[2].strip())
    final_test_methods = set()
    for method in test_methods:
        exceptions = get_found_exceptions(method)
        if not exceptions:
            final_test_methods.add(method)
    return final_test_methods


def replace_text_pieces(text, text_pieces):
    for w in sorted(text_pieces, key=lambda x: len(x), reverse=True):
        text = text.replace(w, " ")
    return text


def split_and_filter_empty_words(text):
    return [_s.strip() for _s in text.split() if _s.strip()]


def remove_guid_uids_from_text(text):
    for pattern in [
            r"[0-9a-fA-F]{16,48}|[0-9a-fA-F]{10,48}\.\.\.",
            "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}" # noqa
            ]:
        strings_to_replace = set()
        for m in re.findall(pattern, text):
            if not m.isdigit() and m.strip():
                strings_to_replace.add(m)
        for _str in sorted(strings_to_replace, key=lambda x: (len(x), x), reverse=True):
            text = text.replace(_str, " ")
    return text


def preprocess_test_item_name(text):
    text = text.replace("-", " ").replace("_", " ")
    all_words = []
    words = split_words(text, to_lower=False, only_unique=False)
    for w in words:
        if "." not in w:
            all_words.extend([s.strip() for s in re.split("([A-Z][^A-Z]+)", w) if s.strip()])
        else:
            all_words.extend(
                [s.strip() for s in enrich_text_with_method_and_classes(w).split(" ") if s.strip()])
            all_words.extend(
                [s.strip() for s in re.split("([A-Z][^A-Z]+)", w.split(".")[-1]) if s.strip()])
    return " ".join(all_words)


def preprocess_found_test_methods(text):
    all_words = []
    words = split_words(text, to_lower=False, only_unique=False)
    for w in words:
        if "." not in w:
            all_words.extend([s.strip() for s in re.split("([A-Z][^A-Z]+)", w) if s.strip()])
        else:
            all_words.append(w)
    return " ".join(all_words)
