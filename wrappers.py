import numpy as np

class ScoringClassifier:
    def __init__(self, label, train, predict, score):
        self.label   = label
        self.train   = train
        self.predict = predict
        self.score   = score

class Regressor:
    def __init__(self, label, model):
        self.label = label
        self.model = model
        self.T     = 0.0

class RewardFn:
    def __init__(self, label, gain, loss):
        self.label = label
        self.gain  = gain
        self.loss  = loss

    def __call__(self, T, P):
        res = []
        rew = 0.0
        for t, p in zip(T, P):
            rew += self.gain(t, p) \
                 - self.loss(t, p)
            res.append(rew)
        return np.array(res)

class RewardFnAbstain:
    def __init__(self, label, gain, loss, abstain):
        self.label   = label
        self.gain    = gain
        self.loss    = loss
        self.abstain = abstain

    def __call__(self, T, P):
        res = []
        rew = 0.0

        for i, (t, p) in enumerate(zip(T, P)):
            rew += self.gain(t, p) \
                 - self.loss(t, p) \
                 - self.abstain(len(T) - (i + 1))
            res.append(rew)
        return np.array(res)
