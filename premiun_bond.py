import re
import pandas as pd
import requests
import os
import datetime
import logging
from bs4 import BeautifulSoup

from helpers.dataframe import pandas_show_all


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()  # also prints to console
    ]
)

# NOTE: essential files
base_url = 'https://www.nsandi.com/prize-checker/winners'


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
        winner_df = pd.read_excel(self.target_file_path, sheet_name=0, engine='openpyxl', skiprows=2,
                                  index_col='Winning Bond NO.')
        winner_df = winner_df.loc[:, ~winner_df.columns.str.contains('^Unnamed')]
        for bond_bundle, bond_details in self.bond_record.items():
            # print(bond_bundle)
            # print(bond_details)
            start_id = bond_details['bond_id_start']
            end_id = bond_details['bond_id_end']
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
                            print('Better luck next time!')


