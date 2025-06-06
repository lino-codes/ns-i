from premiun_bond import PremiumBonds
bond_record = {'0': {'start_id': '615AF256808',
                     'end_id': '615AF259807'}}

def check_prize():
    bond_obj = PremiumBonds(bond_record)
    bond_obj.download_prize_winner()
    bond_obj.check_prize()

if __name__ == '__main__':
    check_prize()
