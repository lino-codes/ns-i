from bond_record import bonds_history
from premiun_bond import PremiumBonds


def run_analysis():
    bond_obj = PremiumBonds(bonds_history)
    # bond_obj.bond_analysis()
    bond_obj.read_historical_winners()


if __name__ == '__main__':
    run_analysis()