import math
import random

from wrappers import RewardFn

def __reward_fn(T, P):
    res = []
    rew = 0.0
    for t, p in zip(T, P):
        rew += 1.0 if t == p else -1000.0
        res.append(rew)
    return res

def __reward_fn(T, P):
    res = []
    rew = 0.0
    for t, p in zip(T, P):
        rew = rew + 1.0 if t == p else 0.0
        res.append(rew)
    return res

def __reward_fn(T, P):
    res = []
    rew = 0.0
    for t, p in zip(T, P):
        rew += random.uniform(0.0, 1.0) if t == p \
            else - 10 * random.uniform(0.0, 1.0)
        res.append(rew)
    return res

def __reward_fn_1(T, P):
    res = []
    rew = 0.0
    for t, p in zip(T, P):
        rew += 0.0 if t == p \
            else -1.0
        res.append(rew)
    return res

def __reward_fn(T, P):
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

RFNS = [ RewardFn("test_fn", __reward_fn)
       , RewardFn("test_fn2", __reward_fn_1) ]
