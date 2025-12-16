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

import re
import string
import urllib.parse
from typing import Iterable
from urllib.parse import urlparse

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.commons.model.launch_objects import Log, SimilarityResult

try:
    from app.commons import logging
except ImportError:
    import logging  # type: ignore

logger = logging.getLogger("analyzerApp.utils.textProcessing")

STOPWORDS = nltk.corpus.stopwords.words("english")
STOPWORDS_ALL = set(STOPWORDS)
FILE_EXTENSIONS = ["java", "php", "cpp", "cs", "c", "h", "js", "swift", "rb", "py", "scala"]


def create_punctuation_map(split_urls: bool) -> dict[str, str | int | None]:
    translate_map: dict[str, str | int | None] = {}
    for punct in string.punctuation + "<>{}[];=()'\"":
        if punct != "." and (split_urls or punct not in ["/", "\\"]):
            translate_map[punct] = " "
    return translate_map


PUNCTUATION_MAP_NO_SPLIT_URLS = create_punctuation_map(False)
PUNCTUATION_MAP_SPLIT_URLS = create_punctuation_map(True)


def replace_patterns(text: str, patterns: Iterable[tuple[re.Pattern, str]]) -> str:
    """Removes starting patterns from the text."""
    result = text
    for p, repl in patterns:
        result = p.sub(repl, result)
    return result


def remove_patterns(text: str, patterns: Iterable[re.Pattern]) -> str:
    """Removes starting patterns from the text."""
    return replace_patterns(text, [(p, "") for p in patterns])


EU_DATE: str = r"\d+-\d+-\d+"
EU_TIME: str = r"\d+:\d+:\d+(?:[.,]\d+)?"
US_DATE: str = r"\d+/\d+/\d+"
US_TIME: str = EU_TIME

EU_DATETIME: str = rf"{EU_DATE}\s+{EU_TIME}"
US_DATETIME: str = rf"{US_DATE}\s+{US_TIME}"

DELIM: str = r"(?:\s*-\s*)|(?:\s*\|\s*)"

DATETIME_PATTERNS: Iterable[re.Pattern] = [
    re.compile(rf"^{EU_DATETIME}(?:{DELIM})?\s*"),
    re.compile(rf"^{US_DATETIME}(?:{DELIM})?\s*"),
    re.compile(rf"^{EU_TIME}(?:{DELIM})?\s*"),
    re.compile(rf"^\[{EU_TIME}](?:{DELIM})?\s*"),
    re.compile(rf"^\[{EU_DATETIME}](?:{DELIM})?\s*"),
]


def remove_starting_datetime(text: str) -> str:
    """Removes datetime at the beginning of the text."""
    return remove_patterns(text, DATETIME_PATTERNS)


LOG_LEVEL: str = r"(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\s?"
LOG_LEVEL_PATTERNS: Iterable[re.Pattern] = [
    re.compile(rf"^{LOG_LEVEL}(?:{DELIM})?\s+"),
    re.compile(rf"^\[{LOG_LEVEL}](?:{DELIM})?\s+"),
    re.compile(rf"^\({LOG_LEVEL}\)(?:{DELIM})?\s+"),
]


def remove_starting_log_level(text: str) -> str:
    """Removes log level at the beginning of the text."""
    return remove_patterns(text, LOG_LEVEL_PATTERNS)


THREAD_ID_PATTERN: str = r"\d+\s+-+\s*"
THREAD_ID_PATTERNS: Iterable[re.Pattern] = [
    re.compile(rf"^{THREAD_ID_PATTERN}(?:{DELIM})?\s+"),
]


def remove_starting_thread_id(text: str) -> str:
    """Removes thread id at the beginning of the text."""
    return remove_patterns(text, THREAD_ID_PATTERNS)


THREAD_NAME_PATTERN: str = r"\[[^\]]*]"
THREAD_NAME_PATTERNS: Iterable[re.Pattern] = [re.compile(rf"^{THREAD_NAME_PATTERN}(?:{DELIM})?\s+")]


def remove_starting_thread_name(text: str) -> str:
    """Removes thread name at the beginning of the text."""
    return remove_patterns(text, THREAD_NAME_PATTERNS)


def filter_empty_lines(log_lines: list[str]) -> list[str]:
    return [line for line in log_lines if line.strip()]


def delete_empty_lines(log: str) -> str:
    """Delete empty lines"""
    return "\n".join(filter_empty_lines(log.split("\n")))


def calculate_line_number(text):
    """Calculate line numbers in the text"""
    return len([line for line in text.split("\n") if line.strip()])


def is_python_log(log):
    """Tries to find whether a log was for the python language"""
    found_file_extensions = []
    for m in re.findall(r"\.(%s)(?!\.)\b" % "|".join(FILE_EXTENSIONS), log):
        found_file_extensions.append(m)
    found_file_extensions = list(set(found_file_extensions))
    if len(found_file_extensions) == 1 and found_file_extensions[0] == "py":
        return True
    return False


def is_starting_message_pattern(text):
    processed_text = text
    res = re.search(r"\w*\s*\(\s*.*\.%s:\d+\s*\)" % "|".join(FILE_EXTENSIONS), processed_text)
    if res and processed_text.startswith(res.group(0)):
        return True
    return False


def get_found_exceptions(text: str, to_lower: bool = False) -> str:
    """Extract exception and errors from logs"""
    unique_exceptions = set()
    found_exceptions = []
    for word in split_words(text, to_lower=to_lower):
        for key_word in ["error", "exception", "failure"]:
            if re.search(r"\S{3,}%s(\s|$)" % key_word, word.lower()) is not None:
                if word not in unique_exceptions:
                    found_exceptions.append(word)
                    unique_exceptions.add(word)
                break
    return " ".join(found_exceptions)


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


def is_line_from_stacktrace(text: str) -> bool:
    """Detects if the line is a stacktrace part"""
    if is_starting_message_pattern(text):
        return False

    res = re.sub(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)", "", text)
    res = re.sub(r"(?<=:)\d+(?=\)?]?(\n|$))", " ", res)
    if res != text:
        return True
    res = re.sub(r"line\s*\d+\s*(?:(?=, in)|(?=,in)|(?=\n)|(?=$))", "line ", res, flags=re.I)
    if res != text:
        # C# stacktrace line
        return True
    res = re.sub("|".join([r"\.%s(?!\.)\b" % ext for ext in FILE_EXTENSIONS]), " ", res, flags=re.I)
    if res != text:
        return True
    result = re.search(r"^\s*at\s[^(]*\([^)]*\)\s*$", res)
    if result and result.group(0) == res:
        # Java stacktrace line
        return True
    else:
        result = re.search(r"^\s*\w+([./]\s*\w+)+\s*\(.*?\)\s*$", res)
        if result and result.group(0) == res:
            return True
    return False


def detect_log_description_and_stacktrace(message: str) -> tuple[str, str]:
    """Split a log into a log message and stacktrace"""
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


SQR_BRCKTS = r"\[[^]]*]"
RND_BRCKTS = r"\([^)]*\)"
CRL_BRCKTS = r"\{[^}]*}"
BRCKTS_TXT = re.compile(rf"{SQR_BRCKTS}|{RND_BRCKTS}|{CRL_BRCKTS}")


def clean_from_brackets(text: str) -> str:
    """Removes all brackets and text inside them from the given text."""
    return BRCKTS_TXT.sub("", text)


SPECIAL_CHARACTER_PATTERN = re.compile(r'[/?&=#@:.*!$%^+~\\|,;<>\[\]{}()`"\'_]')


def clean_special_chars(text: str) -> str:
    """Removes all special characters in the given text."""
    return SPECIAL_CHARACTER_PATTERN.sub(" ", text)


# Mute Sonar's "Group parts of the regex together to make the intended operator precedence explicit." which is invalid
STATUS_CODES_PATTERNS = [
    re.compile(r"\bcode[^\w.]+(\d+)\D*(\d*)|\bcode[^\w.]+(\d*)$", flags=re.IGNORECASE),  # NOSONAR
    re.compile(r"\w+_code[^\w.]+(\d+)\D*(\d*)|\w+_code[^\w.]+(\d*)$", flags=re.IGNORECASE),  # NOSONAR
    re.compile(r"\bstatus[^\w.]+(\d+)\D*(\d*)|\bstatus[^\w.]+(\d*)$", flags=re.IGNORECASE),  # NOSONAR
    re.compile(r"\w+_status[^\w.]+(\d+)\D*(\d*)|\w+_status[^\w.]+(\d*)$", flags=re.IGNORECASE),  # NOSONAR
]


def get_potential_status_codes(text: str) -> list[str]:
    potential_codes_list = []
    for line in text.split("\n"):
        line = clean_from_brackets(line)
        for pattern in STATUS_CODES_PATTERNS:
            result = pattern.search(line)
            if not result:
                continue
            for i in range(1, 4):
                found_code = (result.group(i) or "").strip()
                if found_code:
                    potential_codes_list.append(found_code)
    return potential_codes_list


def get_unique_strings(strings: list[str]) -> list[str]:
    """Get unique strings from the list.

    :param strings: List of strings to filter.
    :return: List of unique strings.
    """
    if not strings:
        return []
    unique_strings = set()
    result = []
    for code in strings:
        if code not in unique_strings:
            unique_strings.add(code)
            result.append(code)
    return result


def get_unique_potential_status_codes(text: str) -> list[str]:
    """Get unique potential status codes from the text."""
    return get_unique_strings(get_potential_status_codes(text))


NUMBER_PATTERN = re.compile(r"\b\d+\b")
NUMBER_PART_PATTERN = re.compile(r"\d+")
NUMBER_TAG = "SPECIALNUMBER"


def remove_numbers(text: str) -> str:
    """Sanitize text by deleting all numbers"""
    result = NUMBER_PATTERN.sub(NUMBER_TAG, text)
    result = NUMBER_PART_PATTERN.sub("", result)
    return result


def first_lines(log_str: str, n_lines: int) -> str:
    """Take n first lines."""
    return "\n".join((log_str.split("\n")[:n_lines])) if n_lines >= 0 else log_str


def prepare_message_for_clustering(
    message: str, number_of_log_lines: int, clean_numbers: bool, leave_log_structure: bool = False
) -> str:
    potential_status_codes = get_unique_potential_status_codes(message)
    message = remove_starting_datetime(message)
    if clean_numbers:
        status_codes_replaced = {}
        for idx, code in enumerate(potential_status_codes):
            replaced_code = "#&#" * (idx + 1)
            status_codes_replaced[replaced_code] = code
            message = re.sub(rf"\b{code}\b", replaced_code, message)
        message = remove_numbers(message)
        for code_replaced in sorted(status_codes_replaced.keys(), reverse=True):
            message = re.sub(code_replaced, str(status_codes_replaced[code_replaced]), message)
    message = delete_empty_lines(message)
    message = first_lines(message, number_of_log_lines)
    if leave_log_structure:
        return message
    words = split_words(message, min_word_length=2, only_unique=False)
    if len(words) == 1:
        return " ".join(words) + " error"
    return " ".join(words)


REGEX_STYLE_TAG = re.compile(r'<style(?:\s+\S+\s*=\s*"[^"]*")*\s*>[^<]*</style>')
REGEX_SCRIPT_TAG = re.compile(r'<script(?:\s+\S+\s*=\s*"[^"]*")*\s*>[^<]*</script>')
REGEX_HTML_TAGS = re.compile(
    r'</?\w+(?:\s+[^>\s]+\s*=\s*"[^"]*"\s?|\s+[^>\s]+\s*)*>'
    r"|&([a-zA-Z0-9]+"
    r"|#[0-9]{1,6}"
    r"|#x[0-9a-fA-F]{1,6});"
)


def clean_text_from_html_tags(message: str) -> str:
    """Removes style and script tags together with inner text and removes html tags"""
    message = re.sub(REGEX_STYLE_TAG, " ", message)
    message = re.sub(REGEX_SCRIPT_TAG, " ", message)
    message = re.sub(REGEX_HTML_TAGS, " ", message)
    return message


def clean_html(message):
    """Removes html tags inside the parts with <.*?html.*?>...</html>"""
    all_lines = []
    started_html = False
    finished_with_html_tag = False
    html_part = []
    for idx, line in enumerate(message.split("\n")):
        if re.search(r'<html(?:\s+\S+\s*=\s*"[^"]*")*\s*>', line):
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


def split_text_on_words(text: str, min_word_length: int, only_unique: bool) -> list[str]:
    all_unique_words = set()
    all_words = []
    for w in text.split():
        w = w.strip().strip(".")
        if w != "" and len(w) >= min_word_length:
            if w in STOPWORDS_ALL:
                continue
            if only_unique:
                if w in all_unique_words:
                    continue
                all_unique_words.add(w)
            all_words.append(w)
    return all_words


def split_words(
    text: str, min_word_length: int = 0, only_unique: bool = True, split_urls: bool = True, to_lower: bool = True
) -> list[str]:
    if not text:
        return []

    if split_urls:
        result = text.translate(str.maketrans(PUNCTUATION_MAP_SPLIT_URLS))
    else:
        result = text.translate(str.maketrans(PUNCTUATION_MAP_NO_SPLIT_URLS))
    result = result.strip().strip(".").strip()
    if to_lower:
        result = result.lower()

    return split_text_on_words(result, min_word_length, only_unique)


def normalize_message(message: str) -> str:
    return " ".join(sorted(split_words(message, to_lower=True)))


def find_only_numbers(detected_message_with_numbers: str) -> str:
    """Removes all non digit symbols and concatenates unique numbers"""
    detected_message_only_numbers = re.sub(r"[^\d ._]", "", detected_message_with_numbers)
    return " ".join(split_words(detected_message_only_numbers, only_unique=False))


def enrich_text_with_method_and_classes(text: str) -> str:
    new_lines = []
    for line in text.split("\n"):
        new_line = line
        found_values = []
        for w in split_words(line, split_urls=True, to_lower=False):
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
            new_line = re.sub(rf"\b(?<!\.){val}(?!\.)\b", full_path, new_line)
        new_lines.append(new_line)
    return "\n".join(new_lines)


SPLIT_WORDS_PATTERN = "([A-Z][^A-Z]+)"


def preprocess_test_item_name(text: str) -> str:
    result = text.replace("-", " ").replace("_", " ")
    all_words = []
    words = split_words(result, to_lower=False, only_unique=False)
    for w in words:
        if "." not in w:
            all_words.extend([s.strip() for s in re.split(SPLIT_WORDS_PATTERN, w) if s.strip()])
        else:
            all_words.extend([s.strip() for s in enrich_text_with_method_and_classes(w).split(" ") if s.strip()])
            all_words.extend([s.strip() for s in re.split(SPLIT_WORDS_PATTERN, w.split(".")[-1]) if s.strip()])
    return " ".join(all_words)


def find_test_methods_in_text(text: str) -> set[str]:
    test_methods = set()
    residual = text
    while True:
        match = re.search(r"\b[^\s()/\\:]+(?:Test|Step)s?\.", residual)
        if not match:
            break
        match_str = residual[match.start() : match.end()]
        residual = residual[match.end() :]
        match = re.search(r"^[^\s()/\\:]+", residual)
        if match:
            match_str += residual[match.start() : match.end()]
            residual = residual[match.end() :]
        test_methods.add(match_str)

    for m in re.findall(r"(\b[^\s()/\\:]+\.(?:spec|cy)\.[jt]s\b)", text):
        if m[0].strip():
            test_methods.add(m[0].strip())
        if m[1].strip():
            test_methods.add(m[1].strip())
    final_test_methods = set()
    for method in test_methods:
        exceptions = get_found_exceptions(method)
        if not exceptions:
            final_test_methods.add(method)
    return final_test_methods


def preprocess_found_test_methods(text: str) -> str:
    all_words = []
    words = split_words(text, to_lower=False, only_unique=False)
    for w in words:
        if "." not in w:
            all_words.extend([s.strip() for s in re.split(SPLIT_WORDS_PATTERN, w) if s.strip()])
        else:
            all_words.append(w)
    return " ".join(all_words)


def compress(text):
    """compress sentence to consist of only unique words"""
    return " ".join(split_words(text, to_lower=False))


def preprocess_words(text):
    all_words = []
    for w in re.finditer(r"[\w.]+", text):
        word_normalized = re.sub(r"^\w\.", "", w.group(0))
        word = word_normalized.replace("_", "")
        if len(word) >= 3:
            all_words.append(word.lower())
        split_parts = word_normalized.split("_")
        split_words_list = []
        if len(split_parts) > 2:
            for idx in range(len(split_parts)):
                if idx != len(split_parts) - 1:
                    split_words_list.append("".join(split_parts[idx : idx + 2]).lower())
            all_words.extend(split_words_list)
        if "." not in word_normalized:
            split_words_list = []
            split_parts = [s.strip() for s in re.split(SPLIT_WORDS_PATTERN, word) if s.strip()]
            if len(split_parts) > 2:
                for idx in range(len(split_parts)):
                    if idx != len(split_parts) - 1:
                        if len("".join(split_parts[idx : idx + 2]).lower()) > 3:
                            split_words_list.append("".join(split_parts[idx : idx + 2]).lower())
            all_words.extend(split_words_list)
    return all_words


UUID = r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}"
TRUNCATED_UUID = r"[0-9a-fA-F]{16,48}|[0-9a-fA-F]{10,48}\.\.\."
NAMED_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-(\w+)"
UUID_TAG = "SPECIALUUID"
GUID_UUID_PATTERNS: Iterable[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{UUID}\b"), UUID_TAG),
    (re.compile(rf"\b{TRUNCATED_UUID}\b"), UUID_TAG),
    (re.compile(rf"\b{NAMED_UUID}\b"), rf"{UUID_TAG} \1"),
]


def remove_guid_uuids_from_text(text: str) -> str:
    return replace_patterns(text, GUID_UUID_PATTERNS)


def replace_tabs_for_newlines(message: str) -> str:
    return message.replace("\t", "\n")


HORIZONTAL_WHITESPACE = (
    r" \t\u00A0\u1680\u180E\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000"
)
LINE_ENDING_PATTERN = re.compile(rf"[{HORIZONTAL_WHITESPACE}]*\r?\n")


def unify_line_endings(message: str) -> str:
    return LINE_ENDING_PATTERN.sub(r"\n", message)


SPACE_PATTERN = re.compile(rf"[{HORIZONTAL_WHITESPACE}]+")
NEWLINE_SPACE_PATTERN = re.compile(rf"[{HORIZONTAL_WHITESPACE}]*\n[{HORIZONTAL_WHITESPACE}]*")
SPACE_REPLACEMENT = " "
NEWLINE_SPACE_REPLACEMENT = "\n"
SPACE_PATTERNS: Iterable[tuple[re.Pattern, str]] = [
    (SPACE_PATTERN, SPACE_REPLACEMENT),
    (NEWLINE_SPACE_PATTERN, NEWLINE_SPACE_REPLACEMENT),
]


def unify_spaces(message: str) -> str:
    return replace_patterns(message, SPACE_PATTERNS)


def fix_big_encoded_urls(message):
    """Decodes urls encoded with %12, etc. and removes brackets to separate url"""
    new_message = message
    try:
        new_message = urllib.parse.unquote(message)
    except:  # noqa
        pass
    if new_message != message:
        return re.sub(r"[(){}#%]", " ", new_message)
    return message


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


INNER_CLASS_EXTERNAL_PATTERN = re.compile(r"\b((?:[a-zA-Z0-9_-]+/|\\)+)([a-zA-Z0-9_-]+)\$([a-zA-Z0-9_-]+\.class)\b")
INNER_CLASS_INTERNAL_PATTERN = re.compile(r"(?<=[.$])([a-zA-Z0-9_-]+)\$(?=[a-zA-Z0-9_-]+[.$(@])")
GENERATED_LINE_PATTERN = re.compile(
    (
        r"\s*(?:at\s*)?(?:[a-zA-Z0-9_-]+\.)+(?:[a-zA-Z0-9_-]+\$\$)+[0-9a-f]+\."
        r"(?:[a-zA-Z0-9_-]+\$|\.)*[a-zA-Z0-9_-]+\(<generated>\).*"
    )
)
CLASS_NAME_WITH_MEMORY_REFERENCE_PATTERN = re.compile(r"\b((?:[a-zA-Z0-9_-]+\.)+)([a-zA-Z0-9_-]+)@[0-9a-f]+\b")
TRUNCATED_STACKTRACE_PATTERN = re.compile(r"\s*\.\.\. \d+ more.*")
STACKTRACE_PATTERNS: Iterable[tuple[re.Pattern, str]] = [
    (GENERATED_LINE_PATTERN, r""),
    (INNER_CLASS_EXTERNAL_PATTERN, r"\1\2.\3"),
    (INNER_CLASS_INTERNAL_PATTERN, r"\1."),
    (CLASS_NAME_WITH_MEMORY_REFERENCE_PATTERN, r"\1\2"),
    (TRUNCATED_STACKTRACE_PATTERN, r""),
]


def remove_generated_parts(message):
    """Removes lines with '<generated>' keyword and removes parts, like $ab24b, @c321e from words"""
    return replace_patterns(message, STACKTRACE_PATTERNS)


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


def leave_only_unique_logs(logs: list[Log]) -> list[Log]:
    unique_logs = set()
    all_logs = []
    for log in logs:
        stripped_message = log.message.strip()
        if stripped_message not in unique_logs:
            all_logs.append(log)
            unique_logs.add(stripped_message)
    return all_logs


def clean_colon_stacking(text: str) -> str:
    return text.replace(":", " : ")


def clean_from_params(text: str) -> str:
    return clean_special_chars(text)


URL_PATTERN = re.compile(r"\b[\w+]+:/\S+\b", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    """Extracts URLs from the given text.

    :param text: The input text from which to extract URLs.
    :return: A list of extracted URLs.
    """
    all_urls = []
    for param in URL_PATTERN.findall(text):
        url = param.strip()
        all_urls.append(url)
    return all_urls


WINDOWS_PATHS = r"""
        (?:^|\s|"|'|`)      # start of line, space, quote
        [A-Za-z]:\\         # drive letter + back-slash
        (?:[^:\\\r\n]+\\)*  # zero or more sub-dirs — NO colon allowed!
        [^\s\\:\r\n]+       # final component: no space, back-slash or colon
    """
POSIX_PATHS = r"""
        (?:^|\s|"|'|`)             # start of line, space, quote
        /                          # leading slash
        (?:                        # 0+ "segment/" blocks
            (?:\\.|[^/\s\r\n])+ /  # seg may contain backslash-escapes
        )*                         # 0+ segments
        (?:\\.|[^/\s\r\n])+        # final segment
    """
PATH_REGEX = re.compile(rf"{WINDOWS_PATHS}|{POSIX_PATHS}", re.VERBOSE)


def extract_paths(text_without_urls: str) -> list[str]:
    all_unique = set()
    all_paths = []
    for param in PATH_REGEX.findall(text_without_urls):
        path = param.strip()
        if path and path not in all_unique:
            all_unique.add(path)
            all_paths.append(path)
    return all_paths


def extract_message_params(text: str) -> list[str]:
    all_unique = set()
    all_params = []
    for param in re.findall(r"(^|\W)('.*'|\".*\")(\W|$)", text):
        param = re.search(r"[^\'\"]+", param[1].strip())
        if param is not None:
            param = param.group(0).strip()
            if param not in all_unique:
                all_unique.add(param)
                all_params.append(param)
    return all_params


def build_url(main_url: str, url_params: list) -> str:
    """Build url by concatenating url and url_params"""
    return main_url + "/" + "/".join(url_params)


def remove_credentials_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    new_netloc = re.sub("^[^:]+:[^@]*@", "", parsed_url.netloc)
    if parsed_url.netloc == new_netloc:
        return url
    return url.replace(parsed_url.netloc, new_netloc)


def does_stacktrace_need_words_reweighting(log: str) -> bool:
    found_file_extensions = []
    all_extensions_to_find = "|".join(["py", "java", "php", "cpp", "cs", "c", "h", "js", "swift", "rb", "scala"])
    for m in re.findall(r"\.(%s)(?!\.)\b" % all_extensions_to_find, log):
        found_file_extensions.append(m)
    if len(found_file_extensions) == 1 and found_file_extensions[0] in {"js", "c", "h", "rb", "cpp"}:
        return True
    return False


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


def transform_string_feature_range_into_list(text: str) -> list[int]:
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


def unite_project_name(project_id: str | int, prefix: str) -> str:
    return f"{prefix}{project_id}"


def replace_text_pieces(text: str, text_pieces: Iterable[str]) -> str:
    result = text
    for w in sorted(text_pieces, key=lambda x: len(x), reverse=True):
        result = result.replace(w, " ")
    return result


def prepare_es_min_should_match(min_should_match: float) -> str:
    return str(int(min_should_match * 100)) + "%"


ACCESS_OR_REFRESH_TOKEN_PATTERN = r"(?:access|refresh|biometric|jwt)_?token"
JSON_ACCESS_TOKEN = rf'("{ACCESS_OR_REFRESH_TOKEN_PATTERN}"\s*:\s*")[^"]+'
HTTP_ACCESS_TOKEN = (
    r"(Authorization\s*:\s*(?:Bearer|Basic|Digest|HOBA|Mutual|Negotiate|NTLM|VAPID|SCRAM|AWS4-HMAC-SHA256)) .*"
)
TOKEN_TAG = "SPECIALTOKEN"
TOKEN_REPLACEMENT = rf"\1{TOKEN_TAG}"
ACCESS_TOKEN_PATTERNS: Iterable[tuple[re.Pattern, str]] = [
    (re.compile(JSON_ACCESS_TOKEN, re.RegexFlag.IGNORECASE), TOKEN_REPLACEMENT),
    (re.compile(HTTP_ACCESS_TOKEN, re.RegexFlag.IGNORECASE), TOKEN_REPLACEMENT),
]


def remove_access_tokens(text: str) -> str:
    return replace_patterns(text, ACCESS_TOKEN_PATTERNS)


MARKDOWN_MODE_PATTERN = re.compile(r"!!!MARKDOWN_MODE!!!\s*")
MARKDOWN_MODE_REPLACEMENT = ""
MARKDOWN_MODE_PATTERNS: Iterable[tuple[re.Pattern, str]] = [(MARKDOWN_MODE_PATTERN, MARKDOWN_MODE_REPLACEMENT)]


def remove_markdown_mode(text: str) -> str:
    return replace_patterns(text, MARKDOWN_MODE_PATTERNS)


MARKDOWN_CODE_SEPARATOR: str = r"`{3}"
FANCY_TEXT_SEPARATOR_START: str = r"-{3,}=+"
FANCY_TEXT_SEPARATOR_END: str = r"={3,}-+"
MARKDOWN_TEXT_SEPARATOR: str = r"-{3,}"
EQUALITY_TEXT_SEPARATOR: str = r"={3,}"
UNDERSCORE_TEXT_SEPARATOR: str = r"_{3,}"
TEXT_SEPARATORS_PATTERN: str = (
    rf"(?:{FANCY_TEXT_SEPARATOR_START}|{FANCY_TEXT_SEPARATOR_END}"
    rf"|{MARKDOWN_CODE_SEPARATOR}|{MARKDOWN_TEXT_SEPARATOR}|{EQUALITY_TEXT_SEPARATOR}"
    rf"|{UNDERSCORE_TEXT_SEPARATOR})"
)
CODE_SEPARATOR_REPLACEMENT: str = "TEXTDELIMITER"
CODE_SEPARATOR_PATTERNS: Iterable[tuple[re.Pattern, str]] = [
    (re.compile(rf"\n{TEXT_SEPARATORS_PATTERN}\n"), rf" {CODE_SEPARATOR_REPLACEMENT}\n"),
    (re.compile(rf"^{TEXT_SEPARATORS_PATTERN}\n"), rf" {CODE_SEPARATOR_REPLACEMENT}\n"),
    (re.compile(rf"\s*{TEXT_SEPARATORS_PATTERN}\n"), rf" {CODE_SEPARATOR_REPLACEMENT}\n"),
    (re.compile(rf"\n{TEXT_SEPARATORS_PATTERN}\s+"), rf"\n{CODE_SEPARATOR_REPLACEMENT} "),
    (re.compile(rf"\s+{TEXT_SEPARATORS_PATTERN}\s+"), rf" {CODE_SEPARATOR_REPLACEMENT} "),
    (re.compile(rf"^{TEXT_SEPARATORS_PATTERN}\s*"), rf"{CODE_SEPARATOR_REPLACEMENT} "),
    (re.compile(rf"\s+{TEXT_SEPARATORS_PATTERN}$"), rf" {CODE_SEPARATOR_REPLACEMENT}"),
    (re.compile(rf"{TEXT_SEPARATORS_PATTERN}$"), rf" {CODE_SEPARATOR_REPLACEMENT}"),
]


def replace_code_separators(text: str) -> str:
    return replace_patterns(text, CODE_SEPARATOR_PATTERNS)


WEBDRIVER_SCREENSHOT_PATTERN = re.compile(r"(?:\s*-*>\s*)?Webdriver screenshot captured: [^/\0\n.]+\.\w+")
WEBDRIVER_SCREENSHOT_REFERENCE_PATTERN = re.compile(r"\s*Screenshot: file:/(?:[^/\0\n]+/)*[^/\0\n]+")
WEBDRIVER_PAGE_SOURCE_REFERENCE_PATTERN = re.compile(r"\s*Page source: file:/(?:[^/\0\n]+/)*[^/\0\n]+")
WEBDRIVER_BUILD_INFO_PATTERN = re.compile(r"\s*Build info: version: '[^']+', revision: '[^']+'")
WEBDRIVER_DRIVER_INFO_PATTERN = re.compile(r"\s*Driver info: [\w.]+")
WEBDRIVER_SYSTEM_INFO_PATTERN = re.compile(r"\s*System info: (?:[\w.]+: '[^']+', )+[\w.]+: '[^']+'")
WEBDRIVER_DRIVER_CAPABILITIES_PATTERN = re.compile(r"\s*Capabilities {\w+: [^\n]+")

WEBDRIVER_AUXILIARY_INFO_REPLACEMENT = ""
WEBDRIVER_AUXILIARY_PATTERNS: Iterable[tuple[re.Pattern, str]] = [
    (WEBDRIVER_SCREENSHOT_PATTERN, WEBDRIVER_AUXILIARY_INFO_REPLACEMENT),
    (WEBDRIVER_SCREENSHOT_REFERENCE_PATTERN, WEBDRIVER_AUXILIARY_INFO_REPLACEMENT),
    (WEBDRIVER_PAGE_SOURCE_REFERENCE_PATTERN, WEBDRIVER_AUXILIARY_INFO_REPLACEMENT),
    (WEBDRIVER_BUILD_INFO_PATTERN, WEBDRIVER_AUXILIARY_INFO_REPLACEMENT),
    (WEBDRIVER_DRIVER_INFO_PATTERN, WEBDRIVER_AUXILIARY_INFO_REPLACEMENT),
    (WEBDRIVER_SYSTEM_INFO_PATTERN, WEBDRIVER_AUXILIARY_INFO_REPLACEMENT),
    (WEBDRIVER_DRIVER_CAPABILITIES_PATTERN, WEBDRIVER_AUXILIARY_INFO_REPLACEMENT),
]


def remove_webdriver_auxiliary_info(text: str) -> str:
    return replace_patterns(text, WEBDRIVER_AUXILIARY_PATTERNS)


READABLE_NUMBER = "[EXCLUDED NUMBER]"


def replace_tokens_with_readable_text(text: str) -> str:
    return text.replace(NUMBER_TAG, READABLE_NUMBER)


URL_TAG = "SPECIALURL"


def remove_urls(message: str, urls_list: list[str]) -> str:
    """
    Remove URLs from exception message.

    Replace URLs with a special tag to normalize exception messages.

    :param str message: Exception message to process
    :param list[str] urls_list: List of URLs to be removed

    :return: Exception message with URLs replaced by special tag
    :rtype: str
    """
    result = message
    if not message or not urls_list:
        return result

    # Sort URLs by length in descending order to avoid partial replacements
    for url in sorted(urls_list, key=len, reverse=True):
        result = result.replace(url, URL_TAG)
        # Also try with URL-encoded version
        # noinspection PyBroadException
        try:
            encoded_url = urllib.parse.quote(url)
            if encoded_url != url:
                result = result.replace(encoded_url, URL_TAG)
        except Exception:
            logger.warning("Failed to encode URL: %s", url)
    return result


SPECIAL_CHARACTERS_PATTERN = re.compile(r"[.:/\\{}()\[\]\"',\-+=!@#$%^&*<>?|~`;_]")
CAMEL_CASE_PATTERN = re.compile(r"([a-z])([A-Z])")
UPPER_LOWER_CASE_PATTERN = re.compile(r"([A-Z])([A-Z][a-z])")


def preprocess_text_for_similarity(text: str) -> str:
    """
    Preprocess text for similarity calculation following the specified requirements.

    :param text: Input text to preprocess
    :return: Preprocessed text ready for similarity calculation
    """
    if not text:
        return ""

    result = text

    # 1. Text delimited with dots, slashes, colons, braces, quotes, commas, etc. -> separate words
    result = SPECIAL_CHARACTERS_PATTERN.sub(" ", result)

    # 2. Handle CamelCase and camelCase - insert space before uppercase letters that follow lowercase
    result = CAMEL_CASE_PATTERN.sub(r"\1 \2", result)

    # 3. Handle sequences of uppercase letters followed by lowercase (like XMLHttpRequest -> XML Http Request)
    result = UPPER_LOWER_CASE_PATTERN.sub(r"\1 \2", result)

    # 4. Lowercase everything
    result = result.lower()

    # 5. Replace excessive spaces with one space
    result = re.sub(r"\s+", " ", result).strip()

    # 6. Use lemmatizer and text normalization
    lemmatizer = WordNetLemmatizer()

    # 7. Remove English stopwords and lemmatize
    words = result.split()
    processed_words = []

    for word in words:
        if not word or word in STOPWORDS_ALL:
            continue

        # Lemmatize the word
        lemmatized_word = lemmatizer.lemmatize(word)
        processed_words.append(lemmatized_word)

    return " ".join(processed_words)


def calculate_text_similarity(base_text: str, *other_texts: str) -> list[SimilarityResult]:
    """
    Calculate similarity between a base text and multiple other texts using TF-IDF vectorization and cosine similarity.

    This function preprocesses all texts according to specified requirements and then
    computes their similarity using computationally efficient methods.

    :param base_text: Base text to compare against
    :param other_texts: Variable number of texts to compare with the base text
    :return: List of SimilarityResult objects where `similarity` is in [0.0, 1.0]
             and `both_empty` indicates both texts were empty
    """
    if not other_texts:
        return []

    # Preprocess the base text
    processed_base_text = preprocess_text_for_similarity(base_text)

    # Preprocess all other texts
    processed_other_texts = [preprocess_text_for_similarity(text) for text in other_texts]

    # Handle empty texts and identical texts first
    similarity_scores: list[SimilarityResult] = []
    valid_texts = []
    valid_indices = []

    for i, processed_other_text in enumerate(processed_other_texts):
        both_empty = not processed_base_text.strip() and not processed_other_text.strip()
        if both_empty:
            # The next variable will be False if base texts contain only stop words
            base_both_empty = not base_text.strip() and not other_texts[i].strip()
            similarity_scores.append(
                SimilarityResult(
                    similarity=0.0 if base_both_empty else float(base_text == other_texts[i]),
                    both_empty=base_both_empty,
                )
            )
        elif not processed_base_text.strip() or not processed_other_text.strip():
            # If one of the texts is empty after preprocessing, append 0
            similarity_scores.append(SimilarityResult(similarity=0.0, both_empty=both_empty))
        elif processed_base_text == processed_other_text:
            # If both texts are identical after preprocessing, append 1
            similarity_scores.append(SimilarityResult(similarity=1.0, both_empty=both_empty))
        else:
            # Store valid texts for batch processing
            valid_texts.append(processed_other_text)
            valid_indices.append(i)
            # Placeholder, will be replaced
            similarity_scores.append(SimilarityResult(similarity=0.0, both_empty=both_empty))

    if not valid_texts:
        return similarity_scores

    # If we have valid texts to process, use single TF-IDF vectorizer

    # Create all texts list: base text + all valid other texts
    all_texts = [processed_base_text] + valid_texts

    # Use TF-IDF vectorization and cosine similarity for efficient computation
    vectorizer = TfidfVectorizer(
        lowercase=False,  # Already lowercased in preprocessing
        token_pattern=r"\b\w+\b",  # Simple word tokenization
        min_df=1,  # Include all terms
        max_df=1.0,  # Include all terms
        ngram_range=(1, 2),  # Use unigrams and bigrams
        use_idf=False,
    )

    # Fit and transform all texts at once
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Calculate cosine similarity between base text (index 0) and all other texts
    base_vector = tfidf_matrix[0:1]  # Base text vector
    other_vectors = tfidf_matrix[1:]  # All other text vectors

    similarity_matrix = cosine_similarity(base_vector, other_vectors)

    # Update similarity scores for valid texts
    for i, valid_index in enumerate(valid_indices):
        similarity_scores[valid_index] = SimilarityResult(
            similarity=float(similarity_matrix[0][i]),
            both_empty=False,
        )

    return similarity_scores


def calculate_similarity_by_values(first_field: str, second_field: str) -> float:
    """Calculate overlap-based similarity for two space-separated value strings.

    The function splits both input strings on whitespace, normalizes tokens by trimming and
    lowercasing, and performs one-to-one greedy matching between tokens from the longer list
    (base) and the shorter list (compared). Each token in the compared list can be matched at most
    once. The similarity is defined as:

        similarity = (common / M) * (1 - different / M)

    where M is max(len(first_tokens), len(second_tokens)); ``common`` is the number of matched
    tokens, and ``different`` is the number of base tokens that had no match. If either input has no
    non-empty tokens, the function returns 0.0. The result is in the range [0.0, 1.0].

    :param str first_field: Space-separated values from the first input string.
    :param str second_field: Space-separated values from the second input string.
    :return: Overlap similarity penalized by unmatched tokens, in [0.0, 1.0].
    :rtype: float

    Examples:
    - "1 2 3" vs "2 3 4" -> common = 2, different = 1, M = 3 -> (2/3) * (1 - 1/3) = 4/9 ≈ 0.444.
    - "" vs "a" -> 0.0.

    Notes:
    - Comparison is case-insensitive and ignores empty tokens.
    - Duplicate tokens are matched greedily at most once (multiset-like behavior).
    """
    rq_values = [f for f in first_field.split(" ") if f.strip()]
    rs_values = [f for f in second_field.split(" ") if f.strip()]
    if not rq_values or not rs_values:
        return 0.0
    max_value_number = max(len(rq_values), len(rs_values))
    if len(rq_values) > len(rs_values):
        base_values = rq_values
        compared_values = rs_values
    else:
        base_values = rs_values
        compared_values = rq_values
    common_value_number = 0
    different_value_number = 0
    for base_value in base_values:
        found = False
        for i, compare_value in enumerate(compared_values):
            if base_value.strip().lower() == compare_value.strip().lower():
                common_value_number += 1
                del compared_values[i]
                found = True
                break
        if not found:
            different_value_number += 1
    return (common_value_number / max_value_number) * (1 - (different_value_number / max_value_number))
