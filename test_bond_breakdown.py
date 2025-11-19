import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from helpers.dataframe import pandas_show_all
from scipy.stats import norm
import re
import zipfile
from io import BytesIO
import os


total_bonds = 133692841902
# NOTE: Getting total bonds

# pandas_show_all()
# # print(total_bonds)
#
# url = "https://nsandi-corporate.com/news-research/news/ps1-million-jackpot-win-two-premium-bonds-holders-shropshire-and-york"
#
# res = requests.get(url, verify=False)
# soup = BeautifulSoup(res.text, "html.parser")
#
# # Find all tables
# tables = soup.find_all("table")
#
# # Extract and save each table
# for i, table in enumerate(tables):
#     df = pd.read_html(str(table))[0]
#
# # print(df)
#
# # Table 1: rows 1 to 11 (actual prizes breakdown)
# prize_breakdown = df.iloc[1:12].reset_index(drop=True)
# prize_breakdown.columns = ['prize_value', 'number_of_prize']
# prize_breakdown['prize_value'] = (
#     prize_breakdown['prize_value']
#     .str.replace('£', '', regex=False)
#     .str.replace(',', '', regex=False)
#     .astype(int)
# )
# prize_breakdown['number_of_prize'] = prize_breakdown['number_of_prize'].astype(int)
#
# # Table 2: only the summary totals row (row 13)
# prize_totals = df.iloc[[13]].reset_index(drop=True)
# prize_totals.columns = ['total_prize_value', 'total_number_of_prize']
#
#
# prize_breakdown['total_prize_value'] = prize_breakdown['prize_value'] * prize_breakdown['number_of_prize']
# print(prize_breakdown)
# print(total_bonds)
# total_prize = prize_breakdown.number_of_prize.sum()
# total_value = prize_breakdown.total_prize_value.sum()
# print(f'The total number of prize: {total_prize}')
# print(f'The total prize fund: {total_value}')
#
# #NOTE: Expected winning disregarding the bond values
# your_investment = 50000
# winning_per_prize = total_value/total_prize
# winning_odds = your_investment*(total_prize/total_bonds)
#
# # NOTE:
# prize_breakdown['winning_odd'] = prize_breakdown['number_of_prize'] / total_bonds
# prize_breakdown['winning_odd_based_on_user'] = prize_breakdown['number_of_prize'] / total_bonds * your_investment
# print(prize_breakdown)
#
# variance = np.sum((prize_breakdown['prize_value']**2)
#                   * prize_breakdown['winning_odd_based_on_user']
#                   * (1 - prize_breakdown['winning_odd_based_on_user']))
#
# # Step 3: Take square root of variance to get standard deviation
# std_dev = np.sqrt(variance)
# print(f'Winning per prize: {winning_per_prize}')
# print(f'Your Odds of winning a price: {winning_odds:.2F}')
# print(f'Your expected winning every month: {(winning_odds*winning_per_prize):.2F}')
# print("Standard deviation:", std_dev)
#
# actual_win = 500
#
# # 2. Calculate the percentile for £200
#
# z_score = (actual_win - (winning_odds*winning_per_prize)) / std_dev
# percentile = norm.cdf(z_score) * 100
#
# print(f'Winning £{actual_win} is at the {percentile:.2f} percentile.')
# # NOTE: A z-score is a statistical measure that tells you how many standard deviations a data point is away from the
# #  mean of dataset
# print(f'Z-score: {z_score:.2f}')
#
#
#
