"""Conformal prediction functionalities"""

from ..nonconformist.base import ClassifierAdapter
from ..nonconformist.icp import IcpClassifier
from ..nonconformist.nc import ClassifierNc
from ..nonconformist.acp import CrossConformalClassifier

def condition(x):
	return x[1]

def get_cross_conformal_classifier(base_mod):    
    mod = ClassifierAdapter(base_mod)
    nc  = ClassifierNc(mod)
    icp = IcpClassifier(nc, condition = condition) # Mondrian
    ccp = CrossConformalClassifier(icp)
    return ccp

def get_cross_conformal_regressor(base_mod):

    pass