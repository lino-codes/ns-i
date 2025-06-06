import re
import requests
import os
import datetime
from bs4 import BeautifulSoup

# NOTE: essential files
base_url = 'https://www.nsandi.com/prize-checker/winners'


class PremiumBonds(object):
    def __init__(self, bond_record):
        self.base_url = base_url
        self.bond_record = bond_record

    def download_prize_winner(self):
        response = requests.get(self.base_url)
        soup = BeautifulSoup(response.text, "html.parser")

        href_downloaded = ''  # NOTE: define what has already been downloaded

        target_directory = r'./bond_files'
        target_file = f'prize-{datetime.date.today().strftime("%B-%Y")}.xlsx'
        target_file_path = os.path.join(target_directory, target_file)

        if os.path.isfile(target_file_path):
            print(f'Latest winner file is available: {target_file}')
        else:
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href != href_downloaded:
                    if href.endswith(".xlsx"):
                        href_downloaded = href
                        full_url = f"https://www.nsandi.com{href}" if href.startswith("/") else href
                        print("Found file:", full_url)

                        # Download the file
                        file_name = full_url.split("/")[-1]

                        with open(f'C:/Users/LN129546/pycharm/premium_bonds/bond_files/{file_name}', "wb") as f:
                            file_data = requests.get(full_url).content
                            f.write(file_data)
                        print(f"Downloaded {file_name}")

    def check_prize(self):
        print(self.bond_record)
        for bond_bundle, bond_details in self.bond_record.items():
            match = re.match(r"([A-Za-z]+)(\d+)|(\d+)([A-Za-z]+)", bond_details['start_id'])
            if match:
                groups = match.groups()
                # Filter out None values
                parts = [g for g in groups if g]
                print("Part 1:", parts[0])
                print("Part 2:", parts[1])
            else:
                print("Pattern not matched")





