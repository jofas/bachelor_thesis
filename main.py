import h5py
import numpy as np
import random
import math

import matplotlib.pyplot as plt
plt.style.use("ggplot")

from libconform import CP
from libconform.ncs import NCSKNearestNeighbors

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as KNN,\
    KNeighborsRegressor as KNNR
from sklearn.ensemble import RandomForestRegressor as RFR,\
    RandomForestClassifier as RF
from sklearn.gaussian_process import \
    GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, \
    ConstantKernel as C
from sklearn.svm import SVR

from infinity import inf
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr

def load_usps():
    with h5py.File('conform/data/usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]

        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]


    X = np.vstack((X_tr, X_te))
    y = np.concatenate((y_tr, y_te))

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    return X, y

def load_usps_random():
    X, y = load_usps()

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y

class AbstainPredictor:
    def __init__(
        self, clf_train, clf_predict, clf_score, reward,
        regressors, ax = None, e=1e-4, scaled = False
    ):
        self.clf_train   = clf_train
        self.clf_predict = clf_predict
        self.clf_score   = clf_score

        self.reward = reward
        self.T      = 1.0

        self.ax     = ax
        self.e      = e
        self.scaled = scaled

        self.regressors = regressors

    def train_cal(self, X, y, ratio=0.1):
        train = int(len(X) * (1 - ratio))
        self.clf_train(X[:train], y[:train])
        self._train(self._A(X[train:], y[train:]))

    def train_kfold(self, X, y, kfolds=5):
        A = []

        kf = KFold(n_splits = kfolds)
        for idx_train, idx_test in kf.split(X):
            X_train, y_train = X[idx_train], y[idx_train]
            X_test,  y_test  = X[idx_test],  y[idx_test]

            self.clf_train(X_train, y_train)

            A += self._A(X_test, y_test)

        self.clf_train(X, y)
        self._train(A)

    def _train(self, A):
        self.T, rew_pts = self._reward_calc(A)
        self._cal_regressors(rew_pts)
        self._plot(rew_pts, "Training set")

    # score {{{
    def score(self, X, y):
        A = self._A(X, y)

        T, rew_pts = self._reward_calc(A)
        rew, rej = self._score_T(self.T, rew_pts)

        print("{:20}  {:5}  {:5}".format(
            "Regressor", "Rew", "Rej"
        ))

        print("{:20}  {:.3f}  {:.3}".format(
            "raw T", rew, rej
        ))

        for reg in self.regressors:
            rew, rej = self._score_T(reg["T"], rew_pts)

            print("{:20}  {:.3f}  {:.3f}".format(
                reg["label"], rew, rej
            ))
        print()

        self._plot(rew_pts, "Test set")
    # }}}

    # _reward_calc {{{
    def _reward_calc(self, A):
        A = np.array(sorted(A, key=lambda x: x[-1]))

        rews = self.reward(A[:,0], A[:,1])

        T                = 0.0
        min_rew, max_rew = 0.0, 0.0
        rew_pts          = [[0.0, 0.0]]
        sig_c, rew_c     = A[0][-1], rews[0]

        for (y, p, sig), rew in zip(A, rews):
            if sig_c != sig:
                rew_pts.append([sig_c, rew_c])
                if rew_c >= max_rew: max_rew=rew_c; T=sig_c
                if rew_c <  min_rew: min_rew=rew_c

                sig_c, rew_c = sig, rew

        rew_c = rews[-1]

        rew_pts.append([sig_c, rew_c])
        if rew_c >= max_rew: max_rew = rew_c; T = sig_c
        if rew_c <  min_rew: min_rew = rew_c

        scale_r = lambda x: (x - min_rew) \
                          / (max_rew - min_rew)

        rew_pts = np.array([[s, scale_r(r)]
            for s, r in rew_pts])

        return T, rew_pts
    # }}}

    # _cal_regressors {{{
    def _cal_regressors(self, rew_pts):
        X, y = rew_pts[:,0].reshape(-1,1), rew_pts[:,1]

        X_pred = [0.0]
        while X_pred[-1] < X[-1]:
            X_pred.append(X_pred[-1] + self.e)
        X_pred = np.array(X_pred).reshape(-1,1)

        for reg in self.regressors:
            reg["reg"].fit(X,y)

            P = reg["reg"].predict(X_pred)

            max_reward_ = -inf
            for x, p in zip(X_pred, P):
                if p > max_reward_:
                    max_reward_ = p
                    reg["T"]    = x[0]

            pts = np.array([[x, y] for x, y in
                zip(X_pred[:,0], P)])
            self._plot(pts, reg["label"])
    # }}}

    def _score_T(self, T, rew_pts):
        res = 0.0
        for i, (sig, rew) in enumerate(rew_pts):
            if sig <= T: res = rew
            else: break
        return res, 1 - (i / len(rew_pts))

    def _plot(self, pts, label):
        if self.ax is not None:
            self.ax.plot(pts[:,0], pts[:,1], label=label)
            self.ax.legend()

    def _A(self, X, y):
        return zip(y,self.clf_predict(X),self.clf_score(X))

def main():
    X, y = load_usps_random()

    #X_train, y_train = X[:-2400],      y[:-2400]
    #X_cal,   y_cal   = X[-2400:-1200], y[-2400:-1200]
    #X_test,  y_test  = X[-1200:],      y[-1200:]

    X_train, y_train = X[:-1200], y[:-1200]
    X_test,  y_test  = X[-1200:], y[-1200:]

    X_train, y_train = X[:400],    y[:400]
    X_test,  y_test  = X[400:600], y[400:600]

    regressors = [
        { "label": "GP [1e-3,1]"
        , "reg": GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-3,1.0)))
        , "T": 0.0 },
        { "label": "GP [1e-1,1]"
        , "reg": GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-1,1.0)))
        , "T": 0.0 },
        { "label": "GP [1,2]"
        , "reg": GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1.0,2.0)))
        , "T": 0.0 },
        { "label": "SVR RBF C100"
        , "reg": SVR(gamma="scale", C=100.0)
        , "T": 0.0 },
        { "label": "SVR RBF C1"
        , "reg": SVR(gamma="scale", C=1.0)
        , "T": 0.0 }
        #{ "label": "Random Forest"
        #, "reg": RFR(n_estimators=10)
        #, "T": 1.0 }
    ]

    def reward_fn(T, P):
        res = []
        rew = 0.0
        for t, p in zip(T, P):
            rew += 1.0 if t == p else -1000.0
            res.append(rew)
        return res

    def reward_fn(T, P):
        res = []
        rew = 0.0
        for t, p in zip(T, P):
            rew = rew + 1.0 if t == p else 0.0
            res.append(rew)
        return res

    def reward_fn(T, P):
        res = []
        rew = 0.0
        for t, p in zip(T, P):
            rew += random.uniform(0.0, 1.0) if t == p \
                else - 10 * random.uniform(0.0, 1.0)
            res.append(rew)
        return res

    def reward_fn(T, P):
        res = []
        rew = 0.0
        for t, p in zip(T, P):
            rew += 0.0 if t == p \
                else -1.0
            res.append(rew)
        return res

    def reward_fn(T, P):
        res = []
        rew = 0.0
        err = 0
        for t, p in zip(T, P):
            if t == p: rew += 1000.0
            else:
                err += 1
                rew -= math.exp(err/3)
            res.append(rew)
        return res

    fig, ax = plt.subplots(1,2)

    ax[0].set_title("Conformal Prediction")
    ax[1].set_title("Random Forest")

    # cp {{{
    ncs = NCSKNearestNeighbors(n_neighbors=1)
    cp = CP(ncs, [], smoothed=False)

    cp_train   = lambda X, y: cp.train(X, y, override=True)
    cp_predict = lambda X: cp.predict_best(X, p_vals=False)
    cp_score   = lambda X: cp.predict_best(X)[1]

    clf = AbstainPredictor(
        cp_train, cp_predict, cp_score, reward_fn,
        regressors, ax=ax[0]
    )

    clf.train_kfold(X_train, y_train, kfolds=5)
    #clf.train_cal(X_train, y_train, ratio=0.5)
    clf.score(X_test, y_test)
    # }}}

    # rf {{{
    rf = RF(n_estimators=100)

    rf_train   = lambda X, y: rf.fit(X, y)
    rf_predict = lambda X: rf.predict(X)
    rf_score   = lambda X: \
        [1 - max(row) for row in rf.predict_proba(X)]

    clf = AbstainPredictor(
        rf_train, rf_predict, rf_score, reward_fn,
        regressors, ax=ax[1]
    )

    clf.train_kfold(X_train, y_train, kfolds=5)
    #clf.train_cal(X_train, y_train, ratio=0.5)
    clf.score(X_test, y_test)
    # }}}

    plt.show()

if __name__ == "__main__":
    main()
