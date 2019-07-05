from libconform import CP
from libconform.ncs import NCSKNearestNeighbors

from sklearn.ensemble import RandomForestClassifier as RF

from wrappers import ScoringClassifier

def __cp_scorer():
    ncs = NCSKNearestNeighbors(n_neighbors=1)
    cp = CP(ncs, [], smoothed=False)
    train   = lambda X, y: cp.train(X, y, override=True)
    predict = lambda X: cp.predict_best(X, p_vals=False)
    score   = lambda X: cp.predict_best(X)[1]
    return ScoringClassifier("cp", train, predict, score)

def __rf_scorer():
    rf = RF(n_estimators=100)
    train   = lambda X, y: rf.fit(X, y)
    predict = lambda X: rf.predict(X)
    score   = lambda X: \
        [1 - max(row) for row in rf.predict_proba(X)]
    return ScoringClassifier("rf", train, predict, score)

CLFS = [__cp_scorer, __rf_scorer]
