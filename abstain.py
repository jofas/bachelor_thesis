import numpy as np
from sklearn.model_selection import KFold

class AbstainClassifier:
    def __init__(
        self, base, reward, regressors, e=1e-4,
        scaled = False
    ):
        from experiment import SingleExperiment
        self.experiment = SingleExperiment(
            base.label, reward.label
        )

        self.base = base

        self.reward = reward
        self.T      = 1.0

        self.e      = e
        self.scaled = scaled

        self.regressors = regressors

    def train_cal(self, X, y, ratio=0.1):
        train = int(len(X) * (1 - ratio))
        self.base.train(X[:train], y[:train])
        self._train(self._A(X[train:], y[train:]))

    def train_kfold(self, X, y, kfolds=5):
        A = []

        kf = KFold(n_splits = kfolds)
        for idx_train, idx_test in kf.split(X):
            X_train, y_train = X[idx_train], y[idx_train]
            X_test,  y_test  = X[idx_test],  y[idx_test]

            self.base.train(X_train, y_train)

            A += self._A(X_test, y_test)

        self.base.train(X, y)
        self._train(A)

    def _train(self, A):
        self.T, rew_pts = self._reward_calc(A)
        self._cal_regressors(rew_pts)
        self.experiment.add_points(rew_pts, "Training set")

    # score {{{
    def score(self, X, y):
        class Res:
            def __init__(self):
                self.rewards  = []
                self.rejected = []
                self.labels   = []

            def append(self, rew, rej, label):
                self.rewards.append(rew)
                self.rejected.append(rej)
                self.labels.append(label)

            def stdout(self):
                print("{:20}  {:5}  {:5}".format(
                    "Regressor", "Rew", "Rej"
                ))

                for rew, rej, label in zip( self.rewards
                                          , self.rejected
                                          , self.labels ):
                    print("{:20}  {:.3f}  {:.3f}".format(
                        label, rew, rej
                    ))
                print()

            def export(self):
                return self.rewards, self.rejected, \
                    self.labels

        res = Res()
        A = self._A(X, y)
        T, rew_pts = self._reward_calc(A)

        res.append(*self._score_T(self.T, rew_pts), "bare")
        for reg in self.regressors:
            res.append( *self._score_T(reg.T, rew_pts)
                      , reg.label )

        self.experiment.result = res.export()
        self.experiment.add_points(rew_pts, "Test set")

        return res
    # }}}

    # _reward_calc {{{
    def _reward_calc(self, A):
        def scale_r(x, mn, mx): return (x - mn) / (mx - mn)

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

        rew_pts = np.array([[s, scale_r(r,min_rew,max_rew)]
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
            reg.model.fit(X,y)

            P = reg.model.predict(X_pred)

            max_reward_ = P[0]
            for x, p in zip(X_pred, P):
                if p > max_reward_:
                    max_reward_ = p
                    reg.T       = x[0]

            pts = np.array([[x, y] for x, y in
                zip(X_pred[:,0], P)])
            self.experiment.add_points(pts, reg.label)
    # }}}

    def _score_T(self, T, rew_pts):
        res = 0.0
        for i, (sig, rew) in enumerate(rew_pts):
            if sig <= T: res = rew
            else: break
        return res, 1 - (i / len(rew_pts))

    def _A(self, X, y):
        return zip(
            y, self.base.predict(X), self.base.score(X)
        )

