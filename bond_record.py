import datetime

# NOTE:
bonds_history = {0: {'eligible_date': datetime.date(2024, 12, 1),
                      'deposit_date': datetime.date(2024, 10, 17),
                      'start_id': '598DB544814',
                      'end_id': '598DB549813'},
                  1: {'eligible_date': datetime.date(2024, 12, 1),
                      'deposit_date': datetime.date(2024, 10, 31),
                      'start_id': '600QB262524',
                      'end_id': '600QB263523'},
                  2: {'eligible_date': datetime.date(2025, 3, 1),
                      'deposit_date': datetime.date(2025, 1, 3),
                      'start_id': '608ZF437404',
                      'end_id': '608ZF443403'},
                  3: {'eligible_date': datetime.date(2025, 4, 1),
                      'deposit_date': datetime.date(2025, 2, 19),
                      'start_id': '615AF256808',
                      'end_id': '615AF259807'},
                 4: {'eligible_date': datetime.date(2025, 11, 1),
                     'deposit_date': datetime.date(2025, 9, 18),
                     'start_id': '641WC926902',
                     'end_id': '641WC929901'},
                 5: {'eligible_date': datetime.date(2025, 11, 1),
                     'deposit_date': datetime.date(2025, 9, 26),
                     'start_id': '642WS000911',
                     'end_id': '642WS002910'},
                 6: {'eligible_date': datetime.date(2026, 1, 1),
                     'deposit_date': datetime.date(2025, 11, 24),
                     'start_id': '651HD648032',
                     'end_id': '651HD650531'},

                  }

win_record = {0: {'bond_id': '598DB548967',
                  'winning_period': 202502,
                  'winning_amount': 25,
                  'eligible_date': datetime.date(2024, 12, 1),
                  'deposit_date': datetime.date(2024, 10, 17)},
              1: {'bond_id': '615AF259643',
                  'winning_period': 202504,
                  'winning_amount': 25,
                  'eligible_date': datetime.date(2025, 4, 1),
                  'deposit_date': datetime.date(2025, 2, 19)
                  },
              2: {'bond_id': '608ZF439137',
                  'winning_period': 202505,
                  'winning_amount': 100,
                  'eligible_date': datetime.date(2025, 3, 1),
                  'deposit_date': datetime.date(2025, 1, 3)
                  }              }

# TODO: This
total_eligible_bonds =  {datetime.date(2025,12,1): 134_625_121_463,
                         datetime.date(2025,11,1): 133_692_841_902,
                         datetime.date(2025,10,1): 133_096_700_225,
                         datetime.date(2025,9,1): 132_593_956_732,
                         datetime.date(2025,8,1): 132_118_833_579,
                         datetime.date(2025,7,1): 131_905_631_906,
                         datetime.date(2025,6,1): 131_438_233_006,
                         }