"""Conformal prediction functionalities"""

from ..nonconformist.acp import CrossConformalClassifier
from ..nonconformist.base import ClassifierAdapter
from ..nonconformist.icp import IcpClassifier
from ..nonconformist.nc import ClassifierNc


def condition(x):
    return x[1]


def get_cross_conformal_classifier(base_mod, n_models):
    mod = ClassifierAdapter(base_mod)
    nc = ClassifierNc(mod)
    icp = IcpClassifier(nc, condition=condition)  # Mondrian
    ccp = CrossConformalClassifier(icp, n_models=n_models)
    return ccp


def get_cross_conformal_regressor(base_mod):
    pass
