from data import *
from clfs import CLFS
from regs import REGS
from reward_fns import RFNS
from experiment import Experiment

def main():

    print("\nwine quality")
    X, y = load_wine_quality()
    e = Experiment("wine_quality_02", X, y, CLFS, REGS, RFNS)
    e.start()
    e.export("experiments/")

    print("\ncar")
    X, y = load_car()
    e = Experiment("car_02", X, y, CLFS, REGS, RFNS)
    e.start()
    e.export("experiments/")

    print("\nbank additional")
    X, y = load_bank_additional()
    e = Experiment("bank_additional_02", X, y, CLFS, REGS, RFNS)
    e.start()
    e.export("experiments/")

if __name__ == "__main__":
    main()

