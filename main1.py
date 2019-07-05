from data import *
from clfs import CLFS
from regs import REGS
from reward_fns import RFNS
from experiment import Experiment

def main():

    print("\nusps")
    X, y = load_usps()
    e = Experiment("usps_02", X, y, CLFS, REGS, RFNS)
    e.start()
    e.export("experiments/")

    print("\ncredit card")
    X, y = load_credit_card()
    e = Experiment("credit_card_02", X, y, CLFS, REGS, RFNS)
    e.start()
    e.export("experiments/")

    print("\nbank")
    X, y = load_bank()
    e = Experiment("bank_02", X, y, CLFS, REGS, RFNS)
    e.start()
    e.export("experiments/")

if __name__ == "__main__":
    main()
