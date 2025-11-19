import re
import pandas as pd
import requests
import os
import datetime
import logging
from bs4 import BeautifulSoup

import zipfile
from io import BytesIO
from bond_record import win_record
from helper import extract_number
from helpers.dataframe import pandas_show_all


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()  # also prints to console
    ]
)
logger = logging.getLogger(__name__)

# NOTE: essential files
base_url = 'https://www.nsandi.com/prize-checker/winners'
pandas_show_all()

class PremiumBonds(object):
    def __init__(self, bond_record):
        self.base_url = base_url
        self.bond_record = bond_record
        self.target_file_path = ''

    def download_prize_winner(self):
        response = requests.get(self.base_url)
        soup = BeautifulSoup(response.text, "html.parser")

        href_downloaded = ''  # NOTE: define what has already been downloaded

        target_directory = r'./bond_files'
        target_file = f'prize-{datetime.date.today().strftime("%B-%Y")}.xlsx'
        self.target_file_path = os.path.join(target_directory, target_file)

        if os.path.isfile(self.target_file_path):
            logging.info(f"Latest winner file is already downloaded {target_file}")
        else:
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href != href_downloaded:
                    if href.endswith(".xlsx"):
                        href_downloaded = href
                        full_url = f"https://www.nsandi.com{href}" if href.startswith("/") else href
                        print("Found file:", full_url)
                        logging.info(f"Identified winner excel file from NS&I: {full_url}")

                        # Download the file
                        file_name = full_url.split("/")[-1]

                        with open(fr'./bond_files/{file_name}', "wb") as f:
                            file_data = requests.get(full_url).content
                            f.write(file_data)
                        logging.info(f"Winner excel file downloaded: {file_name}")

    def check_prize(self):
        pandas_show_all()
        try:
            winner_df = pd.read_excel(self.target_file_path, sheet_name=0, engine='openpyxl', skiprows=2,
                                      index_col='Winning Bond NO.')
        except FileNotFoundError:
            logging.warning(f"Winner excel file not found. Double check url.")
            return None
        winner_df = winner_df.loc[:, ~winner_df.columns.str.contains('^Unnamed')]
        for bond_bundle, bond_details in self.bond_record.items():
            start_id = bond_details['start_id']
            end_id = bond_details['end_id']
            start_match = re.match(r'(\d*[A-Z]+)(\d+)', start_id, re.IGNORECASE)
            end_match = re.match(r'(\d*[A-Z]+)(\d+)', end_id, re.IGNORECASE)

            bond_start_str, bond_start_int = start_match.group(1), start_match.group(2)
            bond_start_int = int(bond_start_int)

            bond_end_str, bond_end_int = end_match.group(1), end_match.group(2)
            bond_end_int = int(bond_end_int)

            if start_match and end_match:
                if bond_start_str != bond_end_str:
                    logging.warning('Inconsistent bond purchase ids format')
                    break

                else:
                    prize_df = winner_df[winner_df.index.str.contains(bond_start_str, na=False)]
                    if not prize_df.empty:
                        prize_df = prize_df.reset_index()
                        prize_df['bond_id'] = prize_df['Winning Bond NO.'].str.extract(r'(\d*[A-Z]+)(\d+)')[1].astype(
                            int)
                        prize_df['in_purchase_bond'] = prize_df['bond_id'].between(bond_start_int, bond_end_int)
                        prize_df = prize_df[prize_df['in_purchase_bond']]
                        if not prize_df.empty:
                            for index, row in prize_df.iterrows():
                                print(f'You have WON £{row["Prize Value"]}')
                                print(f'Are you from {row["Area"]} with a total holding of £{row["Total V of Holding"]}?')
                                # print(prize_df)
                        else:
                            print(f'{bond_bundle}: You have the same winning prefix {bond_start_str}. '
                                  f'Better luck next time!')
                    else:
                        print(f'{bond_bundle}: Not even close this time :(')


    def bond_analysis(self):
        df_bond = pd.DataFrame.from_dict(self.bond_record, 'index')
        df_bond['number_of_bonds'] = df_bond.apply(lambda row: extract_number(row["end_id"]) -
                                                               extract_number(row["start_id"]) + 1, axis=1)

        df_win = pd.DataFrame.from_dict(win_record, orient='index')
        df_bond['start_num'] = df_bond['start_id'].apply(extract_number)
        df_bond['end_num'] = df_bond['end_id'].apply(extract_number)
        df_win['id_num'] = df_win['id'].apply(extract_number)

        df_bond['winnings'] = 0
        for _, win_row in df_win.iterrows():
            mask = (df_bond['start_num'] <= win_row['id_num']) & (df_bond['end_num'] >= win_row['id_num'])
            df_bond.loc[mask, 'winnings'] += win_row['amount']
        df_bond = df_bond.drop(columns=['start_num', 'end_num'])
        df_win = df_win.drop(columns=['id_num'])

        def annualised_bond_yield(df_bond, start_cols):
            as_of = datetime.date.today()
            df = df_bond.copy()

            for start_col in start_cols:
                start = pd.to_datetime(df_bond[f'{start_col}_date']).dt.date
                days = pd.Series((as_of - start).map(lambda d: d.days), index=df.index).clip(lower=1)
                principal = df['number_of_bonds'].astype(float)
                # NOTE: Compound Interest
                # df[f'{start_col}_annualised_rate'] = (1+ df['winnings'].astype(float) / principal)**(365 / days) - 1
                # NOTE: Simple Interest
                df[f'{start_col}_annualised_rate'] = (df['winnings'].astype(float) / principal)*(365 / days)
                df[f'{start_col}_days_held'] = days
                df['principal'] = principal
            return df

        df_out = annualised_bond_yield(df_bond, start_cols=['eligible', 'deposit'])
        total_bonds = df_out['number_of_bonds'].sum()

        weighted_rate = (df_out['eligible_annualised_rate'] * df_out['number_of_bonds']).sum() / total_bonds * 100
        print(f'Annualised Interest Rate for the entire portfolio based on eligible_date: {weighted_rate:.2f} ')
        weighted_rate = (df_out['deposit_annualised_rate'] * df_out['number_of_bonds']).sum() / total_bonds * 100
        print(f'Annualised Interest Rate for the entire portfolio based on deposit date: {weighted_rate:.2f} ')

    def historical_prize_winner(self):
        url = "https://www.nsandi.com/get-to-know-us/winning-bonds-downloads"

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        extract_dir = "./bond"
        download_exts = ['.zip', '.pdf', '.xls', '.xlsx', '.csv', '.doc', '.docx', '.ppt', '.pptx']
        download_links = []

        files = [f for f in os.listdir(extract_dir) if os.path.isfile(os.path.join(extract_dir, f))]
        month_years = set()
        for file_name in files:
            # Match patterns like PWREP_DDMMYYYY.txt
            match = re.search(r'_(\d{2})(\d{2})(\d{4})\.txt', file_name)
            if match:
                month = match.group(2)
                year = match.group(3)
                month_years.add(f"{month}-{year}")

        files_mm_yy = sorted(month_years)

        pattern = re.compile(r'({})$'.format('|'.join([re.escape(ext) for ext in download_exts])))
        # Find all anchor tags
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Check if link ends with one of the downloadable extensions
            if pattern.search(href.lower()):
                download_links.append(href)
            # Check if anchor has a 'download' attribute
            elif a.get('download') is not None:
                download_links.append(href)

        bond_links = [link for link in download_links if 'unclaimed' not in link]
        month_years = set()

        for file in bond_links:
            match = re.search(r'-(\d{2})-(\d{4})\.zip$', file)
            if match:
                month = match.group(1)
                year = match.group(2)
                month_years.add(f"{month}-{year}")

        bond_mm_yy = sorted(month_years)
        target_mm_yy = [item for item in bond_mm_yy if item not in files_mm_yy]

        if target_mm_yy:
            for link in bond_links:
                print(f"Processing {link} ...")
                resp = requests.get(f"https://www.nsandi.com/{link}")
                with zipfile.ZipFile(BytesIO(resp.content)) as zf:
                    for member in zf.namelist():
                        out_path = os.path.join(extract_dir, member)
                        if os.path.exists(out_path):
                            print(f"Extracted file already exists: {out_path} -- Skipping extraction for this file.")
                            continue
                        # Ensure subdirectories exist
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with zf.open(member) as source, open(out_path, "wb") as target:
                            target.write(source.read())
                        print(f"Extracted: {out_path}")
        else:
            print('All Files have already been extracted.')

    def read_historical_winners(self):
        files = os.listdir(f'./bond')
        matched_files = [f for f in files if f.startswith('PWREP') and f.endswith('.txt')]
        all_df = pd.DataFrame()
        for matched_file in matched_files:
            with open(f'./bond/{matched_file}', 'r', encoding='cp1252') as f:
                text = f.read()
            pattern = r'([A-Z]+\.)\s+([\d,]+) prize.*?£([\d,]+)'

            matches = re.findall(pattern, text)

            # Build the records for DataFrame
            rows = []
            for part, count, prize in matches:
                rows.append({
                    'part': part.replace('.', ''),
                    'number_of_prizes': int(count.replace(',', '')),
                    'prize_amount': int(prize.replace(',', ''))
                })

            df = pd.DataFrame(rows)
            df['total_prize_amount'] = df['number_of_prizes'] * df['prize_amount']
            file_date = matched_file.replace('PWREP_', '')
            file_date = file_date.replace('.txt', '')
            df['file_date'] = file_date

            all_df = pd.concat([all_df, df])

        all_df.to_csv(f"./bond/prize_distribution_{}.csv")


