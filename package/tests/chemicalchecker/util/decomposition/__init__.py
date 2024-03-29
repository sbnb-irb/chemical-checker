"""Cross decomposition.

Canonical correlation analysis (CCA) and partial least squares (PLS) are
well-known approach for feature extraction from two sets of multi-dimensional
arrays and cross decomposition.
We tailor the impletation for the comparison of CC signatures, for the moment
this is just a collection of functions.
"""
from .correlation import dataset_correlation
