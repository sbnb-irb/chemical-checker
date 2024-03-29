"""Scale and impute."""
import os
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

perc = 99.9


def deextremize(X, z_extreme, up=None, dw=None):
    # Be careful! This only works once normalized.
    if up is None:
        up = np.min([np.percentile(X, perc), z_extreme])
    if dw is None:
        dw = np.max([np.percentile(X, 100. - perc), -z_extreme])
    X[X > up] = up
    X[X < dw] = dw
    return X, up, dw


def gaussian_scale_impute(X, z_extreme=10, models_path=None, up=None, dw=None):
    imputer_file = "imput.pcl"
    scaler_file = "scale.pcl"
    fancy_file = "fancy.pcl"
    X = np.array(X)
    if models_path is None:
        imputer_file = None
        scaler_file = None
        fancy_file = None
    else:
        imputer_file = os.path.join(models_path, imputer_file)
        scaler_file = os.path.join(models_path, scaler_file)
        fancy_file = os.path.join(models_path, fancy_file)

    if imputer_file is None or not os.path.exists(imputer_file):
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X)
        with open(imputer_file, 'wb') as fh:
            pickle.dump(imputer, fh)
    else:
        imputer = pickle.load(open(imputer_file, 'rb'))

    M = imputer.transform(X)

    if scaler_file is None or not os.path.exists(scaler_file):
        scaler = RobustScaler()
        scaler.fit(M)
        with open(scaler_file, 'wb') as fh:
            pickle.dump(scaler, fh)
    else:
        scaler = pickle.load(open(scaler_file, 'rb'))

    M = scaler.transform(M)
    M, up, dw = deextremize(M, z_extreme, up, dw)
    M[np.isnan(X)] = np.nan
    if fancy_file is None or not os.path.exists(fancy_file):
        fancy = IterativeImputer()
        fancy.fit(M)
        with open(fancy_file, 'wb') as fh:
            pickle.dump(fancy, fh)
    else:
        fancy = pickle.load(open(fancy_file, 'rb'))
    if np.any(np.isnan(M)):
        M = fancy.transform(M)
        return deextremize(M, z_extreme)
    else:
        return M, up, dw
