import pandas as pd
import re
from pathlib import Path

def extract_number(s):
    return int(re.sub(r"\D", "", s))


def parse_prize_ids(text: str, raw_data_file_name) -> dict[int, list[str]]:
    # The Prize
    header_re = re.compile(
        r"Part.*?each",  # prize with commas
        re.IGNORECASE | re.MULTILINE
    )
    prize_dict: dict[int, list[str]] = {}

    # Find all header matches with their span
    headers = list(header_re.finditer(text))

    # NOTE: for one million:
    header_m = re.compile(
        r"Part.*?1,000,000",  # prize with commas
        re.IGNORECASE | re.MULTILINE
    )

    m_headers = list(header_m.finditer(text))
    for i, header_match in enumerate(m_headers):
        header_text = header_match.group(0).strip()
        start_pos = header_match.end()

        # End at next header or end of text
        end_pos = headers[0].start()
        section_text = text[start_pos:end_pos]

        # Extract IDs: 3 digits + 2 letters + 6 digits
        id_pattern = r'\b\d{1,3}[A-Z]{2}\d{6}\b'
        ids = re.findall(id_pattern, section_text)
        # print('HEADER TEXT:', header_text)
        if ids:
            prize_dict.setdefault(int(re.search(r'£([\d,]+)',
                                                str(header_text)).group(1).replace(',', '')),
                                  []).extend(ids)


    for i, header_match in enumerate(headers):
        header_text = header_match.group(0).strip()
        start_pos = header_match.end()

        # End at next header or end of text
        end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section_text = text[start_pos:end_pos]

        # Extract IDs: 3 digits + 2 letters + 6 digits
        id_pattern = r'\b\d{1,3}[A-Z]{2}\d{6}\b'
        ids = re.findall(id_pattern, section_text)
        # print('HEADER TEXT:', header_text)
        if ids:
            prize_dict.setdefault(int(re.search(r'£([\d,]+)',
                                                str(header_text)).group(1).replace(',', '')),
                                  []).extend(ids)

    par_file_name = str(Path(raw_data_file_name).with_suffix(".parquet"))
    df = pd.DataFrame([prize_dict])
    df.to_parquet(f'./bond/{par_file_name}')
    return prize_dict
