"""Simple sanitization of matrices, i.e. imputation and removal of columns and rows with too many NANs"""
import numpy as np

from chemicalchecker.util import logged

@logged
class Sanitizer(object):

    def __init__(self, trim, max_keys=1000, max_features=5000, min_feature_freq=5, max_feature_freq=0.8, min_key_freq=1, max_key_freq=0.8):
        """Initialize

            Args:
                trim(bool): Trim dataset to have a maximum number of features.
                max_keys(int): Maximum number of keys to be used when using a sign as a reference (default=1000).
                max_features(int): Maximum number of features to keep (default=5000).
                min_feature_freq(int): Minimum number (counts) of occurrences of feature, row-wise. Only applies to categorical data (default=5).
                max_feature_freq(float): Maximum proportion of occurrences of the feature, row-wise. Only applies to categorical dta (default=0.8).
                min_key_freq(int): Minimum number (counts) of non-zero features per row (column-wise). Only applies to categorical data (default=1).
                max_key_freq(float): Maximum proportion of non-zero occurrences per row (column-wise). Only applies to categorical data (default=0.8).
        """
        self.trim = trim
        self.max_keys = max_keys
        self.max_features = max_features
        self.min_feature_freq = min_feature_freq
        self.max_feature_freq = max_feature_freq
        self.min_key_freq = min_key_freq
        self.max_key_freq = max_key_freq

    @staticmethod
    def boolean_matrix(V):
        B = np.zeros((V.shape[0], V.shape[1]), dtype=np.int8)
        idxs = np.where(V != 0)
        B[idxs] = 1
        idxs = np.where(np.isnan(V))
        B[idxs] = 0
        idxs = np.where(np.isinf(V))
        B[idxs] = 0
        return B

    def transform(self, V, keys, keys_raw, features, sign=None):
        """Sanitize data
        
            Args:
                V(matrix): Input matrix.
                sign(CC signature): Auxiliary CC signature used to impute (default=None).
        """
        self.__log.debug("Determining type of data (categorical/continuous)")
        if sign is not None:
            vals = sign[:1000].ravel()
        else:
            vals = V[:1000].ravel()
        if np.sum(vals==0)/len(vals) > 0.5:
            self.__log.debug("...sparse")
            is_sparse = True
            B = self.boolean_matrix(V)
        else:
            self.__log.debug("...dense")
            is_sparse = False
            B = None
        if self.trim:
            if V.shape[1] >= self.max_features:
                self.__log.debug("More than %d features, trimming the least informative ones" % self.max_features)
                if B is not None:
                    sums = np.sum(B, axis=0)
                else:
                    sums = np.nansum(V, axis=0)
                idxs = np.argsort(-sums)[:self.max_features]
                V = V[:,idxs]
                if B is not None:
                    B = B[:,idxs]
                features = features[idxs]
                idxs = np.argsort(features)
                V = V[:,idxs]
                if B is not None:
                    B = B[:,idxs]
                features = features[idxs]
            if is_sparse:
                sums = np.sum(B, axis=0)
                mask = sums >= self.min_feature_freq
                V = V[:,mask]
                B = B[:,mask]
                features = features[mask]
                sums = sums[mask]
                self.__log.debug("Removing poorly populated features %d -> %d" % (len(mask), np.sum(mask)))
                mask = sums <= V.shape[0]*self.max_feature_freq
                V = V[:,mask]
                B = B[:,mask]
                features = features[mask]
                self.__log.debug("Removing highly populated features %d -> %d" % (len(mask), np.sum(mask)))
        if is_sparse:
            sums = np.sum(B, axis=1)
            mask = sums >= self.min_key_freq
            V = V[mask]
            B = B[mask]
            keys = keys[mask]
            keys_raw = keys_raw[mask]
            sums = sums[mask]
            self.__log.debug("Removing poorly populated keys %d -> %d" % (len(mask), np.sum(mask)))
            mask = sums <= V.shape[1]*self.max_key_freq
            V = V[mask]
            B = B[mask]
            keys = keys[mask]
            keys_raws = keys_raw[mask]
            self.__log.debug("Removing highly populated keys %d -> %d" % (len(mask), np.sum(mask)))
        if not np.any(np.isnan(V)) and not np.any(np.isinf(V)):
            self.__log.debug("Matrix does not need further sanitization")
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