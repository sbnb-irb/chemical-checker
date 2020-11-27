"""Simple sanitization of matrices.

Imputation and removal of columns and rows with too many NANs.
"""
import os
import h5py
import uuid
import numpy as np

from chemicalchecker.core.signature_data import DataSignature

from chemicalchecker.util import Config, logged

N_DET = 1000


@logged
class Sanitizer(object):
    """Sanitizer class."""

    def __init__(self, trim, max_keys=1000, max_features=5000,
                 min_feature_freq=5, max_feature_freq=0.8, min_key_freq=1,
                 max_key_freq=0.8, chunk_size=10000):
        """Initialize a Sanitizer instance.

        Args:
            trim (bool): Trim dataset to have a maximum number of features.
            max_keys (int): Maximum number of keys to be used when using a
                sign as a reference (default=1000).
            max_features (int): Maximum number of features to keep
                (default=5000).
            min_feature_freq (int): Minimum number (counts) of occurrences
                of feature, row-wise. Only applies to categorical data
                (default=5).
            max_feature_freq (float): Maximum proportion of occurrences of
                the feature, row-wise. Only applies to categorical dta
                (default=0.8).
            min_key_freq (int): Minimum number (counts) of non-zero
                features per row (column-wise). Only applies to categorical
                data (default=1).
            max_key_freq (float): Maximum proportion of non-zero
                occurrences per row (column-wise). Only applies to
                categorical data (default=0.8).
        """
        self.trim = trim
        self.max_keys = max_keys
        self.max_features = max_features
        self.min_feature_freq = min_feature_freq
        self.max_feature_freq = max_feature_freq
        self.min_key_freq = min_key_freq
        self.max_key_freq = max_key_freq
        self.chunk_size = chunk_size

    @staticmethod
    def data_n(data, name="V"):
        with h5py.File(data, "r") as hf:
            n = hf[name].shape[0]
        return n

    @staticmethod
    def data_m(data, name="V"):
        with h5py.File(data, "r") as hf:
            m = hf[name].shape[1]
        return m

    @staticmethod
    def data_shape(data, name="V"):
        with h5py.File(data, "r") as hf:
            s = hf[name].shape
        return s

    def chunker(self, n):
        size = self.chunk_size
        for i in range(0, n, size):
            yield slice(i, i + size)

    def boolean_matrix(self, data):
        n = self.data_n(data)
        with h5py.File(data, "a") as hf:
            if "B" in hf.keys():
                self.__log.warn(
                    "There was a B (boolean) dataset. It has been removed.")
                del hf["B"]
            for chunk in self.chunker(n):
                V = hf["V"][chunk]
                B = np.zeros((V.shape[0], V.shape[1]), dtype=np.int8)
                idxs = np.where(V != 0)
                B[idxs] = 1
                idxs = np.where(np.isnan(V))
                B[idxs] = 0
                idxs = np.where(np.isinf(V))
                B[idxs] = 0
                if "B" not in hf.keys():
                    hf.create_dataset("B", data=B, maxshape=(None, B.shape[1]))
                else:
                    hf["B"].resize((hf["B"].shape[0] + B.shape[0]), axis=0)
                    hf["B"][-B.shape[0]:] = B
            if hf["B"].shape != hf["V"].shape:
                raise Exception("Shape of B and V are not the same")

    def returner(self, data, was_data):
        if was_data:
            return None
        else:
            with h5py.File(data, "r") as hf:
                V = hf["V"][:]
                keys = hf["keys"].asstr()[:]
                keys_raw = hf["keys_raw"].asstr()[:]
                features = hf["features"].asstr()[:]
            os.remove(data)
            return V, keys, keys_raw, features

    def transform(self, data=None, V=None, keys=None, keys_raw=None,
                  features=None, sign=None):
        """Sanitize data

        Args:
            data (str): Path to data (default=None).
            V (matrix): Input matrix (default=None).
            keys (array): Keys (default=None).
            keys_raw (array): Keys raw (default=None).
            features (array): Features (default=None).
            sign (signature): Auxiliary CC signature used to impute
                (default=None).
        """
        if data is not None and V is not None:
            raise Exception("Too many inputs! Decide between data and V.")
        if data is None:
            was_data = False
            if V is None or keys is None or keys_raw is None or features is None:
                raise Exception(
                    "There is no data or V, keys, keys_raw, features provided...")
            tag = str(uuid.uuid4())
            data = os.path.join(Config().PATH.CC_TMP, "%s.h5" % tag)
            self.__log.debug("Saving temporary data to %s" % data)
            with h5py.File(data, "w") as hf:
                hf.create_dataset("V", data=V)
                hf.create_dataset("keys", data=np.array(
                    keys, DataSignature.string_dtype()))
                hf.create_dataset("keys_raw", data=np.array(
                    keys_raw, DataSignature.string_dtype()))
                hf.create_dataset("features", data=np.array(
                    features, DataSignature.string_dtype()))
            V = None
            keys = None
            keys_raw = None
            features = None
        else:
            was_data = True
            with h5py.File(data, "r") as hf:
                ds = hf.keys()
                if "V" not in ds or "keys" not in ds or "keys_raw" not in ds or "features" not in ds:
                    raise Exception(
                        "data should have V, keys, keys_raw and features...")
        self.__log.debug("Determining type of data (categorical/continuous)")
        if sign is not None:
            vals = sign[:N_DET].ravel()
        else:
            with h5py.File(data, "r") as hf:
                vals = hf["V"][:N_DET].ravel()
        if np.sum(vals == 0) / len(vals) > 0.5:
            self.__log.debug("...sparse")
            is_sparse = True
            self.boolean_matrix(data)
        else:
            self.__log.debug("...dense")
            is_sparse = False

        def do_sums(data, name, axis):
            n = self.data_n(data)
            if axis == 0:
                sums = []
                for chunk in self.chunker(n):
                    with h5py.File(data, "r") as hf:
                        M = hf[name][chunk]
                    sums += [np.nansum(M, axis=0)]
                sums = np.sum(np.array(sums), axis=0)
                return sums
            else:
                sums = []
                for chunk in self.chunker(n):
                    with h5py.File(data, "r") as hf:
                        M = hf[name][chunk]
                        sums += list(np.sum(M, axis=1))
                sums = np.array(sums)
                return sums

        def rewrite_str_array_h5(data, mask, name):
            name_tmp = "%s_tmp" % name
            with h5py.File(data, "a") as hf:
                array_tmp = hf[name][:][mask]
                hf.create_dataset(name_tmp, data=np.array(
                    array_tmp, DataSignature.string_dtype()))
                del hf[name]
                hf[name] = hf[name_tmp]
                del hf[name_tmp]

        def rewrite_matrix_h5(data, mask, axis, name):
            name_tmp = "%s_tmp" % name
            with h5py.File(data, "a") as hf:
                n = hf[name].shape[0]
                create = True
                for chunk in self.chunker(n):
                    if axis == 1:
                        M_tmp = hf[name][chunk][:, mask]
                    else:
                        mask_ = mask[chunk]
                        M_tmp = hf[name][chunk][mask_]
                    if create:
                        hf.create_dataset(name_tmp, data=M_tmp,
                                          maxshape=(None, M_tmp.shape[1]))
                        create = False
                    else:
                        hf[name_tmp].resize(
                            (hf[name_tmp].shape[0] + M_tmp.shape[0]), axis=0)
                        hf[name_tmp][-M_tmp.shape[0]:] = M_tmp
                del hf[name]
                hf[name] = hf[name_tmp]
                del hf[name_tmp]

        def rewrite_features_h5(data, mask):
            rewrite_str_array_h5(data, mask, "features")

        def rewrite_keys_h5(data, mask):
            rewrite_str_array_h5(data, mask, "keys")
            rewrite_str_array_h5(data, mask, "keys_raw")

        if self.trim:
            m = self.data_m(data)
            if m >= self.max_features:
                self.__log.debug(
                    "More than %d features, trimming the least informative ones" % self.max_features)
                if is_sparse:
                    sums = do_sums(data, "B", axis=0)
                else:
                    sums = do_sums(data, "V", axis=0)
                idxs = np.argsort(-sums)[:self.max_features]
                mask = np.array([False] * m)
                mask[idxs] = True
                rewrite_features_h5(data, mask)
                rewrite_matrix_h5(data, mask, axis=1, name="V")
                if is_sparse:
                    rewrite_matrix_h5(data, mask, axis=1, name="B")
            if is_sparse:
                sums = do_sums(data, "B", axis=0)
                mask = sums >= self.min_feature_freq
                sums = sums[mask]
                rewrite_features_h5(data, mask)
                rewrite_matrix_h5(data, mask, axis=1, name="V")
                rewrite_matrix_h5(data, mask, axis=1, name="B")
                self.__log.debug(
                    "Removing poorly populated features %d -> %d" % (len(mask), np.sum(mask)))
                n = self.data_n(data)
                mask = sums <= n * self.max_feature_freq
                rewrite_features_h5(data, mask)
                rewrite_matrix_h5(data, mask, axis=1, name="V")
                rewrite_matrix_h5(data, mask, axis=1, name="B")
                self.__log.debug(
                    "Removing highly populated features %d -> %d" % (len(mask), np.sum(mask)))

        if is_sparse:
            sums = do_sums(data, "B", axis=1)
            mask = sums >= self.min_key_freq
            sums = sums[mask]
            rewrite_keys_h5(data, mask)
            rewrite_matrix_h5(data, mask, axis=0, name="V")
            rewrite_matrix_h5(data, mask, axis=0, name="B")
            self.__log.debug(
                "Removing poorly populated keys %d -> %d" % (len(mask), np.sum(mask)))
            m = self.data_m(data)
            mask = sums <= m * self.max_key_freq
            rewrite_keys_h5(data, mask)
            rewrite_matrix_h5(data, mask, axis=0, name="V")
            rewrite_matrix_h5(data, mask, axis=0, name="B")
            self.__log.debug(
                "Removing highly populated keys %d -> %d" % (len(mask), np.sum(mask)))

        if is_sparse:
            self.__log.debug("Deleting B (boolean) dataset")
            with h5py.File(data, "a") as hf:
                del hf["B"]

        def has_nan_or_inf(data):
            n = self.data_n(data)
            for chunk in self.chunker(n):
                with h5py.File(data, "r") as hf:
                    V = hf["V"][chunk]
                    if np.any(np.isnan(V)):
                        return True
                    if np.any(np.isinf(V)):
                        return True
            return False

        if not has_nan_or_inf(data):
            self.__log.debug("Matrix does not need further sanitizing")
            return self.returner(data, was_data)

        def subsample(data, max_n):
            n = self.data_n(data)
            if n < max_n:
                with h5py.File(data, "r") as hf:
                    W = hf["V"][:]
            else:
                idxs = np.array(sorted(np.random.choice(
                    n, max_n, replace=False)))
                with h5py.File(data, "r") as hf:
                    W = hf["V"][idxs]
            return W

        if sign is None:
            self.__log.debug("Sanitizing using the signature itself.")
            W = subsample(data, self.max_keys)
        else:
            self.__log.debug("Sanitizing using the signature of reference.")
            W = sign.subsample(self.max_keys)[0]

        maxs = []
        mins = []
        meds = []
        for j in range(0, W.shape[1]):
            mask0 = ~np.isnan(W[:, j])
            mask1 = ~np.isposinf(W[:, j])
            mask1 = np.logical_and(mask0, mask1)
            if np.any(mask1):
                maxs += [np.max(W[mask1, j])]
            else:
                maxs += [0]
            mask2 = ~np.isneginf(W[:, j])
            mask2 = np.logical_and(mask0, mask2)
            if np.any(mask2):
                mins += [np.min(W[mask2, j])]
            else:
                mins += [0]
            mask3 = np.logical_and(mask1, mask2)
            if np.any(mask3):
                meds += [np.median(W[mask3, j])]
            else:
                meds += [0]
        maxs = np.array(maxs)
        mins = np.array(mins)
        meds = np.array(meds)
        n = self.data_n(data)
        with h5py.File(data, "a") as hf:
            create = True
            for chunk in self.chunker(n):
                V = hf["V"][chunk]
                for j in range(0, V.shape[1]):
                    mask = np.isposinf(V[:, j])
                    V[mask, j] = maxs[j]
                    mask = np.isneginf(V[:, j])
                    V[mask, j] = mins[j]
                meds = np.nanmedian(W, axis=0)
                for j in range(0, V.shape[1]):
                    mask = np.isnan(V[:, j])
                    V[mask, j] = meds[j]
                name_tmp = "V_tmp"
                if create:
                    hf.create_dataset(name_tmp, data=V,
                                      maxshape=(None, V.shape[1]))
                    create = False
                else:
                    hf[name_tmp].resize(
                        (hf[name_tmp].shape[0] + V.shape[0]), axis=0)
                    hf[name_tmp][-V.shape[0]:] = V
            del hf["V"]
            hf["V"] = hf[name_tmp]
            del hf[name_tmp]

        return self.returner(data, was_data)
