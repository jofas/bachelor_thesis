import numpy as np
import csv
import h5py

def load_usps():
    with h5py.File('data/usps.h5', 'r') as hf:
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

def load_credit_card():
    Z = __import_csv("data/credit_card.csv")
    return Z[:,:-1], Z[:,-1]

def load_wine_quality():
    Z = __import_csv(
        "data/wine_quality/winequality-red.csv"
    )
    Z1 = __import_csv(
        "data/wine_quality/winequality-white.csv"
    )

    Z = np.vstack((Z, Z1))
    return Z[:,:-1], Z[:,-1]

def load_car():
    with open("data/car/car.csv") as f:
        Z = np.array(
            [_ for _ in csv.reader(f, delimiter=",")]
        )

        for i in range(Z.shape[1]):
            uni = {k : i for i, k in
                enumerate(np.unique(Z[:,i]))}

            for j in range(Z.shape[0]):
                Z[j,i] = uni[Z[j,i]]

        Z = np.array(Z, dtype=np.float64)
        return Z[:,:-1], Z[:,-1]

def load_bank():
    with open("data/bank/bank-full.csv") as f:
        Z = np.array(
            [_ for _ in csv.reader(f, delimiter=";")][1:]
        )

        for i in [1,2,3,4,6,7,8,10,15,16]:
            uni = {k : i for i, k in
                enumerate(np.unique(Z[:,i]))}

            for j in range(len(Z)):
                Z[j,i] = uni[Z[j,i]]

        Z = np.array(Z, dtype=np.float64)
        return Z[:,:-1], Z[:,-1]

def load_bank_additional():
    with open(
        "data/bank-additional/bank-additional-full.csv"
    ) as f:
        Z = np.array(
            [_ for _ in csv.reader(f, delimiter=";")][1:]
        )

        for i in [1,2,3,4,5,6,7,8,9,14,20]:
            uni = {k : i for i, k in
                enumerate(np.unique(Z[:,i]))}

            for j in range(len(Z)):
                Z[j,i] = uni[Z[j,i]]

        Z = np.array(Z, dtype=np.float64)
        return Z[:,:-1], Z[:,-1]

def __import_csv(path, delimiter=";", header=True):
    idx = 1 if header else 0
    with open(path) as f:
        return np.array([_ for _ in
                csv.reader(f, delimiter=delimiter)][idx:],
            dtype=np.float64)
