from bond_record import eligible_bonds
from premiun_bond import PremiumBonds


def check_prize():
    bond_obj = PremiumBonds(eligible_bonds)
    bond_obj.download_prize_winner()
    bond_obj.check_prize()


if __name__ == '__main__':
    check_prize()
