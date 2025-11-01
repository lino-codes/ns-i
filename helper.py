import re

def extract_number(s):
    return int(re.sub(r"\D", "", s))
