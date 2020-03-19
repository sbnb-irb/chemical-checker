"""Simple sanitization of matrices, i.e. imputation and removal of columns and rows with too many NANs"""
import numpy as np

from chemicalchecker.util import logged

@logged
class Sanitizer(object):

    def __init__(self, trim, max_keys=1000, max_features=5000):
        """Initialize

            Args:
                max_keys(int): Maximum number of keys to be used when using a sign as a reference (default=1000).
                max_features(int): Maximum number of features to keep (default=5000).
        """
        self.trim = trim
        self.max_keys = max_keys
        self.max_features = max_features

    def transform(self, V, keys, keys_raw, features, sign=None):
        """Sanitize data
        
            Args:
                V(matrix): Input matrix.
                sign(CC signature): Auxiliary CC signature used to impute (default=None).
        """
        if self.trim:
            if V.shape[1] >= self.max_features:
                self.__log.debug("More than %d features, trimming the least informative ones" % self.max_features)
                sums = np.nansum(V, axis=0)
                idxs = np.argsort(-sums)[:self.max_features]
                V    = V[:,idxs]
                features = features[idxs]
                idxs = np.argsort(features)
                V    = V[:,idxs]
                features = features[idxs]
            self.__log.debug("Removing keys with no data")
            mask = np.nansum(V, axis=1) > 0
            V = V[mask]
            keys = keys[mask]
            keys_raw = keys_raw[mask]
        if not np.any(np.isnan(V)) and not np.any(np.isinf(V)):
            self.__log.debug("Matrix does not need sanitization")
            return V, keys, keys_raw, features
        if sign is not None:
            self.__log.debug("Sanitizing using the signature of reference.")
            W = sign.subsample(self.max_keys)[0]
            if V.shape[1] != W.shape[1]:
                raise Exception("V and signature do not have the same dimensions")
            self.__log.debug("Capping inf values if necessary")
            maxs = np.max(W, axis=0)
            mins = np.min(W, axis=0)
            for j in range(0, V.shape[1]):
                mask = np.isposinf(V[:,j])
                V[mask,j] = maxs[j]
                mask = np.isneginf(V[:,j])
                V[mask,j] = mins[j]
            self.__log.debug("Removing NaN values if necessary")
            meds = np.nanmedian(W, axis=0)
            for j in range(0, V.shape[1]):
                mask = np.isnan(V[:,j])
                V[mask,j] = meds[j]
        else:
            self.__log.debug("Sanitizing using the signature itself.")
            self.__log.debug("Capping inf values if necessary")
            for j in range(0, V.shape[1]):
                mask = np.isposinf(V[:,j])
                V[mask,j] = np.max(V[~mask,j])
                mask = np.isneginf(V[:,j])
                V[mask,j] = np.min(V[~mask,j])
            self.__log.debug("Removing NaN values if necessary")
            for j in range(0, V.shape[1]):
                mask = np.isnan(V[:,j])
                V[mask,j] = np.median(V[~mask,j])
        return V, keys, keys_raw, features