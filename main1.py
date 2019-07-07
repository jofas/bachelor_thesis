from data import *
from clfs import CLFS
from regs import REGS
from reward_fns import RFNS, RFNS2
from experiment import Experiment

def main():

    print("\nusps")
    X, y = load_usps()
    e = Experiment("usps_03", X, y, CLFS, REGS, RFNS2)
    e.start()
    e.export("experiments/")

    print("\ncredit card")
    X, y = load_credit_card()
    e = Experiment("credit_card_03", X, y, CLFS, REGS, RFNS2)
    e.start()
    e.export("experiments/")

    print("\nbank")
    X, y = load_bank()
    e = Experiment("bank_03", X, y, CLFS, REGS, RFNS2)
    e.start()
    e.export("experiments/")

if __name__ == "__main__":
    main()
