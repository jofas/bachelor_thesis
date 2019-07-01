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
    def __init__(self, label, reward_fn):
        self.label     = label
        self.reward_fn = reward_fn

    def __call__(self, T, P):
        return self.reward_fn(T, P)

