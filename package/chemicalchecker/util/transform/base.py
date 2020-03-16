"""General class for transforming data.

Transformations are always done on the reference dataset. However, this is an internal issue and what needs to be specified is the full dataset.
"""
import os
import h5py
import numpy as np
import pickle

from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import logged

@logged
class BaseTransform(object):
    """Base transform class"""
    def __init__(self, sign1, name, max_keys):
        """Initialize with a full sign1"""
        self.sign = sign1
        if self.sign.cctype != "sign1":
            raise Exception("Transformations are only allowed for signature 1")
        if self.sign.molset != "full":
            raise Exception("This is a high level functionality of the CC. Only 'full' molset is allowed.")
        self.sign_ref = self.sign.get_molset("reference")
        self.name = name
        self.model_path = self.sign_ref.model_path
        self.max_keys = max_keys
        self.categorical = self.is_categorical()

    def reindex_triplets(self, sign1, keys):
        fn = os.path.join(sign1.model_path, "triplets.h5")
        if not os.path.exists(fn):
            self.__log.debug("No triplets file found in %s" % fn)
            return
        keys_old = sign1.keys
        self.__log.debug("Reindexing triplets")
        with h5py.File(fn, "r") as hf:
            triplets_old = hf["triplets"][:]
        keys_dict = dict((k,i) for i,k in enumerate(keys))
        maps_dict = {}
        for i,k in enumerate(keys_old):
            if k not in keys_dict: continue
            maps_dict[i] = keys_dict[k]
        triplets = []
        for t in triplets_old:
            if t[0] not in maps_dict: continue
            if t[1] not in maps_dict: continue
            if t[2] not in maps_dict: continue
            triplets += [(maps_dict[t[0]], maps_dict[t[1]], maps_dict[t[2]])]
        triplets = np.array(triplets, dtype=np.int)
        with h5py.File(fn, "r+") as hf:
            del hf["triplets"]
            hf["triplets"] = triplets

    def overwrite(self, sign1, V, keys):
        self.reindex_triplets(sign1, keys)
        data_path = sign1.data_path
        with h5py.File(data_path, "r+") as hf:
            del hf["V"]
            hf["V"] = V
            del hf["keys"]
            hf["keys"] = np.array(keys, DataSignature.string_dtype())
        sign1.keys = keys

    def save(self):
        self.__log.debug("Saving transformer object")
        fn = os.path.join(self.model_path, self.name+".pkl")
        with open(fn, "wb") as f:
            pickle.dump(self, f)

    def predict_check(self, sign1):
        if sign1.cctype != "sign1":
            raise Exception("Only predictions for sign1 are accepted")

    def subsample(self):
        max_keys = self.max_keys
        if max_keys is not None:
            if max_keys >= self.sign_ref.shape[0]:
                max_keys = None
        if max_keys is None:
            self.__log.debug("Considering all data")
            V, keys = self.sign_ref[:], self.sign_ref.keys
        else:
            self.__log.debug("Subsampling data")
            V, keys = self.sign_ref.subsample(max_keys)
        features = np.array(["f%d" % i for i in range(self.sign_ref.shape[1])])
        if self.categorical:
            V = V.astype(np.int)
        return V, keys, features

    @staticmethod
    def chunker(n, size=2000):
        for i in range(0, n, size):
            yield slice(i, i+size)

    def is_categorical(self, n=1000):
        self.__log.debug("Checking continuous or categorical")
        V = self.sign_ref[:n]
        is_cat = np.all(V == V.astype(np.int))
        if not is_cat:
            return False
        V = self.sign[:n]
        is_cat = np.all(V == V.astype(np.int))
        if is_cat:
            return True
        else:
            return False