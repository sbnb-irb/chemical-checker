from sklearn.preprocessing import Imputer, RobustScaler
from fancyimpute import MICE as fancyImputer
import numpy as np

perc = 99.9

def deextremize(X, z_extreme):
    # Be careful! This only works once normalized.
    up = np.min([np.percentile(X, perc), z_extreme])
    dw = np.max([np.percentile(X, 100. - perc), -z_extreme])
    up_lim = up + 1
    dw_lim = dw - 1
    X[X > up] = up
    X[X < dw] = dw
    return X

def scaleimpute(X, z_extreme = 10):
    X = np.array(X)
    imputer = Imputer(strategy = "median")
    M = imputer.fit_transform(X)
    scaler = RobustScaler()
    M = scaler.fit_transform(M)
    M = deextremize(M, z_extreme)
    M[np.isnan(X)] = np.nan
    if np.any(np.isnan(M)):
        M = fancyImputer().complete(M)
        return deextremize(M, z_extreme)
    else:
        return M
