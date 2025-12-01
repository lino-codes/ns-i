from bond_record import bonds_history
from premiun_bond import PremiumBonds


def run_analysis():
    """TODO: 1. Check whether we have the entire winner history"""
    bond_obj = PremiumBonds(bonds_history)
    bond_obj.get_complete_historical_prize_winners()
    # bond_obj.bond_analysis()
    # bond_obj.historical_prize_winner()
    # bond_obj.read_historical_winners()
    # bond_obj.odds_analysis()


if __name__ == '__main__':
    run_analysis()