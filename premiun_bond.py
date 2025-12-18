import re
import pandas as pd
import numpy as np
import requests
import os
import datetime
import logging
from bs4 import BeautifulSoup

import zipfile
from scipy.stats import norm
from io import BytesIO
from bond_record import win_record, bonds_history, total_eligible_bonds
from helper import extract_number, parse_prize_ids
from helpers.dataframe import pandas_show_all, dict_to_df

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
        self.text_files = [f for f in os.listdir(f'./bond') if f.startswith('PWREP') and f.endswith('.txt')]
        self.parquet_files = [f for f in os.listdir(f'./bond') if f.startswith('PWREP') and f.endswith('.parquet')]
        self.current_YYYYMM = datetime.datetime.now().strftime("%Y%m")

    def download_high_value_winner(self):
        """Get and download high value winners from NS&I. For unofficial check on the first day of the month."""
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

    def check_high_value_prize(self):
        """Check if any holding bond is a high value winning one."""
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


    def get_complete_historical_prize_winners(self):
        """Download the complete prize winners."""
        logger.info('Getting complete prize winners from NS&I.')
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
            logger.info(f'All Files have already been extracted, latest being {bond_links[0]}')


    def format_complete_historical_prize_to_parquet(self):
        self.get_complete_historical_prize_winners()
        for matched_file in self.text_files:
            if not matched_file.replace('.txt', '.parquet') in self.parquet_files:
                logger.info(f'Creating parquet file from {matched_file}')
                with open(f'./bond/{matched_file}', 'r', encoding='cp1252') as f:
                    text = f.read()
                    # print(text)
                    parse_prize_ids(text, matched_file)
        self.parquet_files = [f for f in os.listdir(f'./bond') if f.startswith('PWREP') and f.endswith('.parquet')]

    def get_full_history_winning(self):
        """Only run this when if the entire history is available."""
        current_YYYYMM = datetime.datetime.now().strftime("%Y%m")

        if not os.path.exists(f'./bond/winning_record_{current_YYYYMM}.parquet'):
            all_winning_bonds = {}
            file_mm_yy_pattern = r'(\d{2})(\d{4})\.parquet$'
            for file in self.parquet_files:
                df = pd.read_parquet(f'./bond/{file}')
                match = re.search(file_mm_yy_pattern, file)
                month = match.group(1)
                year = match.group(2)
                # print(f"Month: {month}, Year: {year}")
                yyyy_mm = year + month
                prizes_raw = df.to_dict(orient='list')

                prizes = {
                    amount: [bond for arr in arr_list for bond in arr.tolist()]
                    for amount, arr_list in prizes_raw.items()
                }

                def in_any_history_range(bond_id, history):
                    win_bool = False
                    deposit_date = None
                    eligible_date = None
                    for record in history.values():
                        if record['start_id'] <= bond_id <= record['end_id']:
                            win_bool = True
                            deposit_date = record['deposit_date']
                            eligible_date = record['eligible_date']
                            return win_bool, deposit_date, eligible_date
                    return win_bool, deposit_date, eligible_date

                total_winnings = 0
                winning_bonds = {}  # optional: to see which of your bonds have won

                for amount, bond_ids in prizes.items():
                    for bond_id in bond_ids:
                        win, deposit_dt, eligible_dt = in_any_history_range(bond_id, bonds_history)
                        if win:
                            total_winnings += amount
                            winning_bonds[bond_id] = {'amount': amount, 'deposit_date': deposit_dt, 'eligible_date': eligible_dt}
                all_winning_bonds[f'{int(yyyy_mm)}'] = winning_bonds

            # Usage
            all_winning_df = dict_to_df(all_winning_bonds)
            all_winning_df.to_parquet(f'./bond/winning_record_{current_YYYYMM}.parquet')
        else:
            logger.info('Parquet file already generated, no need to download all winners')


    def bond_return_analysis(self):
        df_win = pd.read_parquet(f'./bond/winning_record_{self.current_YYYYMM}.parquet')
        df_legacy_win = pd.DataFrame.from_dict(win_record, orient='index')
        df_win = pd.concat([df_win, df_legacy_win])
        df_win = df_win.sort_values(by=['winning_period', 'winning_amount'], ascending=[True, False])
        df_bond = pd.DataFrame.from_dict(self.bond_record, 'index')
        df_bond['number_of_bonds'] = df_bond.apply(lambda row: extract_number(row["end_id"]) -
                                                               extract_number(row["start_id"]) + 1, axis=1)
        df_bond['eligible_date'] = pd.to_datetime(df_bond['eligible_date']).dt.date
        df_bond['winnings'] = 0
        for _, win_row in df_win.iterrows():
            mask = (df_bond['start_id'] <= win_row['bond_id']) & (df_bond['end_id'] >= win_row['bond_id'])
            df_bond.loc[mask, 'winnings'] += win_row['winning_amount']

        # NOTE: Configure some dates
        today = datetime.datetime.today()
        first_day_of_the_month = today.replace(day=1)
        total_bonds = df_bond['number_of_bonds'].sum()

        print(f'Total Bond: {total_bonds}')
        df_bond['eligible_date'] = pd.to_datetime(df_bond['eligible_date'])
        df_bond['deposit_date'] = pd.to_datetime(df_bond['deposit_date'])

        df_bond['YTD_deposit'] = (today - df_bond["deposit_date"]).dt.days / 365.25
        df_bond['YTM_deposit'] = (first_day_of_the_month - df_bond["deposit_date"]).dt.days / 365.25

        df_bond['YTD_eligible'] = (today - df_bond["eligible_date"]).dt.days / 365.25
        df_bond['YTM_eligible'] = (first_day_of_the_month - df_bond["eligible_date"]).dt.days / 365.25

        df_bond.loc[df_bond["YTD_deposit"] < 0, "YTD_deposit"] = 0
        df_bond.loc[df_bond["YTM_deposit"] < 0, "YTM_deposit"] = 0

        df_bond.loc[df_bond["YTD_eligible"] < 0, "YTD_eligible"] = 0
        df_bond.loc[df_bond["YTM_eligible"] < 0, "YTM_eligible"] = 0

        df_bond["number_of_bonds"] = df_bond["number_of_bonds"].astype(float)
        total_winnings = df_bond['winnings'].sum()

        weighted_principal = (df_bond["number_of_bonds"] * df_bond["YTD_deposit"]).sum()
        portfolio_rate = total_winnings/weighted_principal*100
        print(f'The portfolio rate using deposit date as start date and ending on today is {portfolio_rate:.2f}.')

        weighted_principal = (df_bond["number_of_bonds"] * df_bond["YTM_deposit"]).sum()
        portfolio_rate = total_winnings/weighted_principal*100
        print(f'The portfolio rate using deposit date as start date and ending on first day of the month is {portfolio_rate:.2f}.')


        weighted_principal = (df_bond["number_of_bonds"] * df_bond["YTD_eligible"]).sum()
        portfolio_rate = total_winnings/weighted_principal*100
        print(f'The portfolio rate using eligible date as start date and ending on today is {portfolio_rate:.2f}.')

        weighted_principal = (df_bond["number_of_bonds"] * df_bond["YTM_eligible"]).sum()
        portfolio_rate = total_winnings/weighted_principal*100
        print(f'The portfolio rate using eligible date as eligible date and ending on first day of the month is {portfolio_rate:.2f}.')
        return df_bond


    def get_prize_distribution(self):
        """This is to get the historical prize distribution, the number of prizes for each prize category"""
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
            df['file_date'] = str(file_date)
            all_df = pd.concat([all_df, df])

        all_df.to_csv(f"./bond/prize_distribution_{self.current_YYYYMM}.csv")
        all_df.to_parquet(f"./bond/prize_distribution_{self.current_YYYYMM}.parquet")
        return all_df

    def bond_odds_analysis_new(self):
        prize_distribution_parquet = f'./bond/prize_distribution_{self.current_YYYYMM}.parquet'
        if os.path.isfile(prize_distribution_parquet):
            prize_distribution = pd.read_parquet(f'./bond/prize_distribution_{self.current_YYYYMM}.parquet')
        else:
            prize_distribution = self.get_prize_distribution()
        print(prize_distribution)
        df_bond = self.bond_return_analysis()
        print(df_bond)
        df_win = pd.read_parquet(f'./bond/winning_record_{self.current_YYYYMM}.parquet')
        print(df_win)

        return None


    def bond_odds_analysis(self):
        logger.info('Running bond odds analysis')
        return_df = pd.read_parquet('./bond/bond_return.parquet')
        prize_df = pd.read_parquet('./bond/prize_distribution.parquet')
        win_df = pd.read_parquet('./bond/win_record.parquet')
        prize_df["file_date"] = pd.to_datetime(prize_df["file_date"], format="%d%m%Y")
        prize_df["prize_date"] = prize_df["file_date"].dt.strftime("%Y%m")
        prize_df['prize_date'] = prize_df['prize_date'].astype(int)
        prize_sum = prize_df.groupby(["prize_date"])[["number_of_prizes", "total_prize_amount"]].sum().reset_index()

        prize_sum['winning_per_prize'] = prize_sum['total_prize_amount'] / prize_sum['number_of_prizes']
        print('prize sum')
        print(prize_sum)

        return_df['eligible_date'] = pd.to_datetime(return_df['eligible_date'])
        return_df['eligible_ym'] = return_df['eligible_date'].dt.strftime('%Y%m')
        return_df['eligible_ym'] = return_df['eligible_ym'].astype(int)
        bonds_monthly = return_df.groupby('eligible_ym')[['number_of_bonds']].sum()

        df2_sorted = bonds_monthly.sort_index()  # sort by eligible_ym [web:9]
        df2_sorted['cum_bonds'] = df2_sorted['number_of_bonds'].cumsum()  # [web:7][web:10]

        # 2. For each prize_date, find latest eligible_ym <= prize_date
        #    and take its cumulative sum as eligible_bonds
        eligible = (
            df2_sorted
            .reindex(df2_sorted.index.union(prize_sum['prize_date']))  # fill intermediate dates [web:4]
            .sort_index()
        )
        eligible['cum_bonds'] = eligible['number_of_bonds'].fillna(0).cumsum()  # [web:7][web:10]

        # take cum_bonds at the prize_date rows
        lookup = eligible.loc[prize_sum['prize_date'], 'cum_bonds'].to_numpy()

        # 3. Assign to df1
        prize_sum['eligible_bonds'] = lookup
        prize_win = pd.merge(win_df, prize_sum, how='outer', left_on='winning_period', right_on='prize_date')

        total_number_of_bonds = {int(k.strftime('%Y%m')): v for k, v in total_eligible_bonds.items()}
        prize_df['total_no_of_prizes'] = prize_df['prize_date'].map(total_number_of_bonds)

        prize_win['total_number_of_bonds'] = prize_win['prize_date'].map(total_number_of_bonds)

        prize_win['expected_number_of_prize'] = prize_win['number_of_prizes'] / prize_win['total_number_of_bonds'] * prize_win['eligible_bonds']
        prize_win['expected_winning'] = prize_win['expected_number_of_prize'] * prize_win['winning_per_prize']
        prize_stats = prize_win[['prize_date', 'number_of_prizes', 'total_prize_amount', 'winning_per_prize',
                                 'eligible_bonds', 'total_number_of_bonds', 'expected_number_of_prize',
                                 'expected_winning']]

        own_wins = prize_win[['winning_period', 'bond_id', 'winning_amount', 'eligible_date', 'prize_date']]
        print(prize_stats)
        own_wins['total_winning_amount_per_period'] = (
            own_wins.groupby('winning_period')['winning_amount']
            .transform('sum')
        )
        own_wins['total_winning_bonds_per_period'] = (
            own_wins.groupby('winning_period')['bond_id']
            .transform('count')
        )

        own_wins['total_winning_amount_per_period'] = own_wins['total_winning_amount_per_period'].fillna(0)
        own_wins['total_winning_bonds_per_period'] = own_wins['total_winning_bonds_per_period'].fillna(0)
        basic_luck = own_wins[['prize_date', 'total_winning_amount_per_period', 'total_winning_bonds_per_period']]
        basic_luck = basic_luck.drop_duplicates().dropna()
        unique_prize_stats = prize_stats.drop_duplicates().dropna()
        basic_luck_df = pd.merge(basic_luck, unique_prize_stats, how='left', on='prize_date')
        basic_luck_df['luck_amount'] = basic_luck_df['total_winning_amount_per_period'] - basic_luck_df['expected_winning']
        basic_luck_df['luck_prizes'] = basic_luck_df['total_winning_bonds_per_period'] - basic_luck_df['expected_number_of_prize']
        months = max(basic_luck_df['prize_date']) - min(basic_luck_df['prize_date'])
        exp_amt = basic_luck_df['expected_winning'].sum()
        won_amt = basic_luck_df['total_winning_amount_per_period'].sum()
        exp_prizes = basic_luck_df['expected_number_of_prize'].sum()
        won_prizes = basic_luck_df['total_winning_bonds_per_period'].sum()
        luck_amt = basic_luck_df['luck_amount'].sum()
        luck_prizes = basic_luck_df['luck_prizes'].sum()

        print(
            f"In the past {months:.1f} months, "
            f"you were expected to win £{exp_amt:.1f}, "
            f"and you have won £{won_amt:.1f}. "
            f"You were expected to win {exp_prizes:.1f} prizes, "
            f"and you have won {won_prizes:.1f}."
        )

        print(
            f"In the past {months:.1f} months, "
            f"your net prize amount is £{luck_amt:.1f} "
            f"and net number of prizes is {luck_prizes:.1f}."
        )
        # print(basic_luck_df)
        print(prize_df)
        prize_df['prize_date'] = prize_df['prize_date'].astype(int)


        # We'll store percentiles and quantiles in new columns
        month_df = basic_luck_df.copy()
        month_df['theoretical_mean'] = np.nan
        month_df['theoretical_sd'] = np.nan
        month_df['z_score_amount'] = np.nan  # how many SDs your £ winnings are from the mean

        for idx, row in month_df.iterrows():
            m = int(row['prize_date'])
            N = float(row['eligible_bonds'])  # your number of £1 bonds that month

            grp = prize_df[prize_df['prize_date'] == m]
            if grp.empty or N <= 0:
                continue

            total_bonds_in_issue = float(grp['total_no_of_prizes'].iloc[0])

            v = grp['prize_amount'].to_numpy(dtype=float)  # prize sizes
            k = grp['number_of_prizes'].to_numpy(dtype=float)  # counts

            p = k / total_bonds_in_issue  # per-bond probabilities

            mu_bond = np.sum(p * v)
            m2_bond = np.sum(p * v * v)
            var_bond = m2_bond - mu_bond ** 2

            mu = N * mu_bond
            var = N * var_bond
            sd = np.sqrt(var)

            month_df.at[idx, 'theoretical_mean'] = mu
            month_df.at[idx, 'theoretical_sd'] = sd

            actual = float(row['total_winning_amount_per_period'])
            if sd > 0:
                z = (actual - mu) / sd
                month_df.at[idx, 'z_score_amount'] = z


        month_df['luck_percentile'] = norm.cdf(month_df['z_score_amount']) * 100

        print(month_df)
        return None


    def run_premium_bond_analysis(self):
        self.format_complete_historical_prize_to_parquet()
        self.get_full_history_winning()
        # self.bond_return_analysis()
        self.bond_odds_analysis_new()




    def check_if_high_value_winner(self):
        self.download_high_value_winner()
        self.check_high_value_prize()


