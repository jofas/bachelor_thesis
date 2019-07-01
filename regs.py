from sklearn.gaussian_process import \
    GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, \
    ConstantKernel as C
from sklearn.svm import SVR

from wrappers import Regressor

REGS = [
    Regressor(
        "GP [1e-3, 1]",
        GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-3,1.0)))
    ),
    Regressor(
        "GP [1e-1,1]",
        GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-1,1.0)))
    ),
    Regressor(
        "GP [1,2]",
        GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1.0,2.0)))
    ),
    Regressor("SVR RBF C100", SVR(gamma="scale", C=100.0)),
    Regressor("SVR RBF C1", SVR(gamma="scale", C=1.0))
]
