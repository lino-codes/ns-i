from bond_record import bonds_history
from premiun_bond import PremiumBonds


def run_analysis():
    """TODO: 1. Check whether we have the entire winner history"""
    bond_obj = PremiumBonds(bonds_history)
    bond_obj.run_premium_bond_analysis()

if __name__ == '__main__':
    run_analysis()