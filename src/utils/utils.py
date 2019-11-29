import re

def sanitize_text(text):
    return re.sub("\d+", "", text)

def calculate_line_number(text):
    return len([line for line in text.split("\n") if line.strip() != ""])

def first_lines(log_str, n):
    return "\n".join((log_str.split("\n")[:n])) if n >= 0 else log_str

def build_url(main_url, url_params):
    return main_url + "/" + "/".join(url_params)