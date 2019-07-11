import numpy as np
import math
import random

from wrappers import RewardFn, RewardFnAbstain

__ASYM_MAT50 = \
    np.array([
        [37, 13, 18, 13, 16, 46, 43, 48,  4, 39, 32],
        [38, 17, 40, 38, 47,  0, 31, 30, 44,  5, 23],
        [29, 50, 20, 31, 48, 50, 18, 11, 28, 37, 10],
        [20, 37, 40, 31, 23, 20, 13, 27, 34, 34, 11],
        [41, 42, 37, 37, 38, 40, 45, 39, 24,  1, 20],
        [ 1, 36,  9, 40,  8,  4, 12,  6, 10, 49, 50],
        [10, 38, 21, 34, 36,  6, 43,  7, 36, 22, 44],
        [50, 38,  6, 18, 15, 39, 13, 23, 43,  7,  5],
        [45, 24, 28, 49, 11,  5, 22, 31, 11, 34,  9],
        [50, 22, 11, 25,  3, 27, 25, 18, 30,  1, 12],
        [14, 44, 37,  1, 34, 17, 46, 30, 50,  7, 43]
    ])

__ASYM_MAT200 = \
    np.array([
        [ 59, 199,  95, 128, 126, 122,  29, 104,  12, 164, 200],
        [ 81,  28, 182,  40,  60, 156,  49,  64, 199, 180, 127],
        [195, 177, 200,  10, 188, 198, 105,  34,  48, 175,   3],
        [142,  96,  33,  92,  80, 138,  27, 199, 122, 150,  56],
        [186,  69, 176, 143, 159,  83, 132,  79,  37,  26,  72],
        [156, 110,  59, 194, 195,  94,  66, 171, 180,   3, 185],
        [198, 151, 182,   3, 156, 155, 184, 187,  94,  74,  44],
        [156,  21, 200,  47, 112,  10, 112,  32,  28, 125, 122],
        [179,  44, 147, 106,  61,  80, 194,  81,  78, 124,  69],
        [ 66,  23, 150, 102,  11,  95,  38, 150,  75, 121,   7],
        [137,  36,  84, 119, 120,  85, 117,  47,  60,  96,   9]
    ])

__ASYM_MAT1000 = \
    np.array([
        [306, 977, 476, 230, 744, 481, 708, 701, 598,  30, 954],
        [637, 369,   9, 685, 736, 431, 288, 262,  95, 394, 213],
        [503, 678, 878, 279, 196, 100, 178, 791, 636, 398, 634],
        [346, 667,   2, 294, 172, 260, 412, 211, 832, 675, 756],
        [278, 396, 516, 674, 769,  81, 422, 764, 574, 341, 195],
        [266, 368, 983, 513, 653, 537, 621, 141, 512, 778,  95],
        [363, 716, 409, 366,  96, 951, 782, 573, 125, 633, 623],
        [922, 184, 708, 153, 511, 649, 621,  30, 391, 795, 124],
        [444,  59, 123, 684, 794, 223, 296, 875, 506, 948, 104],
        [432, 123, 693, 742, 541, 294,  12,   1, 242, 233, 104],
        [774, 656, 312, 996, 956, 346, 603, 438, 232, 504, 745]
    ])

def simple_abstain(amount):
    return amount

def simple_gain(t, p):
    return 1.0 if t == p else 0.0

def simple_loss(t, p):
    return 0.0 if t == p else 1.0

def random_gain(t, p):
    return 0.0 if t != p else random.uniform(0,1)

def random_loss(t, p):
    return 0.0 if t == p else random.uniform(0,1)

def asym_gain50(t, p):
    return 0.0 if t != p else __ASYM_MAT50[int(p),int(t)]

def asym_gain200(t, p):
    return 0.0 if t != p else __ASYM_MAT200[int(p),int(t)]

def asym_gain1000(t, p):
    return 0.0 if t != p else __ASYM_MAT1000[int(p),int(t)]

def asym_loss50(t, p):
    return 0.0 if t == p else __ASYM_MAT50[int(p),int(t)]

def asym_loss200(t, p):
    return 0.0 if t == p else __ASYM_MAT200[int(p),int(t)]

def asym_loss1000(t, p):
    return 0.0 if t == p else __ASYM_MAT1000[int(p),int(t)]

RFNS2 = [
    RewardFn(
        "random_gain",
        random_gain,
        random_loss,
    ),
    RewardFn(
        "random_gain_scaled_1_5",
        lambda t, p: 5.0 * random_gain(t, p),
        lambda t, p: 5.0 * random_loss(t, p),
    ),
    RewardFn(
        "random_gain_scaled_1_20",
        lambda t, p: 20.0 * random_gain(t, p),
        lambda t, p: 20.0 * random_loss(t, p),
    ),
    RewardFn(
        "random_gain_scaled_1_100",
        lambda t, p: 100.0 * random_gain(t, p),
        lambda t, p: 100.0 * random_loss(t, p),
    ),
    RewardFnAbstain(
        "asymetric_50_abstain",
        asym_gain50,
        asym_loss50,
        simple_abstain,
    ),
    RewardFnAbstain(
        "asymetric_200_abstain",
        asym_gain200,
        asym_loss200,
        simple_abstain,
    ),
    RewardFnAbstain(
        "asymetric_1000_abstain",
        asym_gain1000,
        asym_loss1000,
        simple_abstain,
    ),
]

RFNS = [
    RewardFn("simple", simple_gain, simple_loss),
    RewardFn(
        "simple_scaled_1_5",
        simple_gain,
        lambda t, p: 5.0 * simple_loss(t, p),
    ),
    RewardFn(
        "simple_scaled_1_20",
        simple_gain,
        lambda t, p: 20.0 * simple_loss(t, p),
    ),
    RewardFn(
        "simple_scaled_1_100",
        simple_gain,
        lambda t, p: 100.0 * simple_loss(t, p),
    ),
    RewardFn(
        "asymetric_50",
        asym_gain50,
        asym_loss50,
    ),
    RewardFn(
        "asymetric_200",
        asym_gain200,
        asym_loss200,
    ),
    RewardFn(
        "asymetric_1000",
        asym_gain1000,
        asym_loss1000,
    ),
    RewardFn(
        "random",
        simple_gain,
        random_loss,
    ),
    RewardFn(
        "random_scaled_1_5",
        simple_gain,
        lambda t, p: 5.0 * random_loss(t, p),
    ),
    RewardFn(
        "random_scaled_1_20",
        simple_gain,
        lambda t, p: 20.0 * random_loss(t, p),
    ),
    RewardFn(
        "random_scaled_1_100",
        simple_gain,
        lambda t, p: 100.0 * random_loss(t, p),
    ),
]
