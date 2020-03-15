"""Simple sanitization of matrices, i.e. imputation and removal of columns and rows with too many NANs"""
import numpy as np

from chemicalchecker.util import logged

@logged
class Sanitizer(object):

    def __init__(self, max_keys=1000):
        """Initialize

            Args:
                max_keys(int): Maximum number of keys to be used when using a sign as a reference (default=1000).
        """
        self.max_keys = max_keys

    def transform(self, V, sign=None):
        """Sanitize data
        
            Args:
                V(matrix): Input matrix.
                sign(CC signature): Auxiliary CC signature used to impute (default=None).
        """
        if not np.any(np.isnan(V)) and not np.any(np.isinf(V)):
            self.__log.debug("Matrix does not need sanitization")
            return V
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
        return V