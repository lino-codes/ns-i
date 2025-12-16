from bond_record import bonds_history
from premiun_bond import PremiumBonds


def check_prize():
    # NOTE: Run
    bond_obj = PremiumBonds(bonds_history)
    bond_obj.check_if_high_value_winner()


if __name__ == '__main__':
    check_prize()
