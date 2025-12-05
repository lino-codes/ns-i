import pandas as pd


def pandas_show_all():
    """Call to show more columns and rows"""
    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

def dict_to_df(data):
    rows = []
    for period, accounts in data.items():
        for bond_id, details in accounts.items():
            rows.append({
                'winning_period': period,
                'bond_id': bond_id,
                'winning_amount': details['amount'],
                'deposit_date': details['deposit_date'],
                'eligible_date': details['eligible_date']
            })
    return pd.DataFrame(rows)
