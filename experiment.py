import numpy as np
import os
import math

from abstain import AbstainClassifier

import matplotlib.pyplot as plt
plt.style.use("ggplot")

class SingleExperiment:
    def __init__(self, label, label_rf):
        self.label     = label
        self.label_rf  = label_rf
        self.result    = ([], [], [])
        self.points    = []

    def add_points(self, pts, label):
        self.points.append((pts, label))

class Experiment:
    def __init__(
        self, label, X, y, clfs, regs, rw_fns, ratio=0.1
    ):
        self.label = label

        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        self.X, self.y = X[idxs], y[idxs]

        self.ratio  = ratio
        self.clfs   = clfs
        self.regs   = regs
        self.rw_fns = rw_fns

        self.exps  = []

    def start(self):
        train = int(len(self.X) * (1 - self.ratio))

        for clf in self.clfs:
            for rw_fn in self.rw_fns:
                abstain = AbstainClassifier(
                    clf, rw_fn, self.regs
                )

                self.exps.append(abstain.experiment)

                abstain.train_kfold(
                    self.X[:train], self.y[:train]
                )

                print(clf.label, rw_fn.label)
                abstain.score(
                    self.X[train:], self.y[train:]
                ).stdout()

    def export(self, path):
        path += self.label
        self._mkdir(path)

        for exp in self.exps:
            path_exp = path + "/" + exp.label
            self._mkdir(path_exp)

            path_rf = path_exp + "/" + exp.label_rf
            self._mkdir(path_rf)

            for pts, label in exp.points:
                path_ds = path_rf + "/" + label + ".csv"
                with open(path_ds, "w") as f:
                    for row in pts:
                        f.write("{};{}\n".format(*row))

            with open(path_rf + "/result.csv", "w") as f:
                f.write("Regressor;Reward;Rejected\n")
                for rew, rej, label in zip(*exp.result):
                    f.write("{};{};{}\n".format(
                        label, rew, rej
                    ))

    def _mkdir(self, path):
        try: os.mkdir(path)
        except FileExistsError: pass

    def plot(self):
        lw = math.floor(math.sqrt(len(self.exps)))

        up = lw + 1 if lw ** 2 < len(self.exps) \
            else lw

        fig, axs = plt.subplots(lw, up)

        if lw == 1 and up == 1:
            self._plot(axs, self.exps[0])
        else:
            for i, exp in enumerate(self.exps):
                lw_i = int(i / lw)
                if lw > 1:
                    up_i = i - lw_i
                    self._plot(axs[lw_i, up_i], exp)
                else:
                    self._plot(axs[lw_i], exp)
        plt.show()

    def _plot(self, ax, exp):
        ax.set_title(exp.label)

        for pts, label in exp.points:
            ax.plot(pts[:,0], pts[:,1], label=label)

        ax.legend()
