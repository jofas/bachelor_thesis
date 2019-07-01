from data import load_usps
from clfs import CLFS
from regs import REGS
from reward_fns import RFNS
from experiment import Experiment

def main():
    X, y = load_usps()
    X, y = X[:600], y[:600]

    e = Experiment("01", X, y, CLFS, REGS, RFNS)
    e.start()
    e.export("experiments/")
    e.plot()

if __name__ == "__main__":
    main()
