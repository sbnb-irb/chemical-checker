"""General class for transforming data.

Transformations are always done on the reference dataset. However, this is an internal issue and what needs to be specified is the full dataset.
"""
import os
import h5py
import numpy as np
import pickle
import random
from tqdm import tqdm

from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import logged


@logged
class BaseTransform(object):
    """BaseTransform class"""

    def __init__(self, sign1, name, max_keys, tmp):
        """Initialize a BaseTransform instance.

        Initialize with a full sign1.
        """
        self.sign = sign1
        if self.sign.cctype != "sign1":
            raise Exception("Transformations are only allowed for signature 1")
        if self.sign.molset != "full":
            raise Exception(
                "This is a high level functionality of the CC. "
                "Only 'full' molset is allowed.")
        self.sign_ref = self.sign.get_molset("reference")
        self.name = name
        self.model_path = self.sign_ref.model_path
        self.max_keys = max_keys
        self.categorical = self.is_categorical()
        self.tmp = tmp


    def reindex_triplets(self, sign1, keys):
        fn = os.path.join(sign1.model_path, "triplets.h5")
        if not os.path.exists(fn):
            self.__log.debug("No triplets file found in %s" % fn)
            return
        keys_old = sign1.keys
        if not np.any(keys != keys_old):
            self.__log.debug("...reindexing is not necessary!")
            return
        self.__log.debug("Reindexing triplets")
        with h5py.File(fn, "r") as hf:
            triplets_old = hf["triplets"][:]
        keys_dict = dict((k, i) for i, k in enumerate(keys))
        maps_dict = {}
        self.__log.debug("...enumerating old keys")
        for i, k in enumerate(keys_old):
            if k not in keys_dict:
                continue
            maps_dict[i] = keys_dict[k]
        self.__log.debug("...redoing triplets")
        triplets = []
        for t in triplets_old:
            if t[0] not in maps_dict:
                continue
            if t[1] not in maps_dict:
                continue
            if t[2] not in maps_dict:
                continue
            triplets += [(maps_dict[t[0]], maps_dict[t[1]], maps_dict[t[2]])]
        self.__log.debug("...Saving triplets to %s" % fn)
        triplets = np.array(triplets, dtype=int)
        with h5py.File(fn, "r+") as hf:
            del hf["triplets"]
            hf["triplets"] = triplets

    def remap(self, sign1):
        sign1.refresh()
        self.__log.debug("Re-doing the mappings (if necessary)")
        s1_full = sign1.get_molset("full")
        s1_ref = sign1.get_molset("reference")
        if not os.path.exists(s1_ref.data_path):
            self.__log.debug("Reference not available")
            return
        mappings = s1_ref.get_h5_dataset("mappings")
        keys = s1_full.keys

        temp_keys = keys
        temp_map = mappings[:, 0]
        if mappings[:, 0].shape[0] != keys.shape[0]:
            if mappings[:, 0].shape[0] < keys.shape[0]:
                temp_keys = keys[0:mappings[:, 0].shape[0]]
            else:
                temp_map = mappings[0:keys.shape[0], 0]
        if not np.any(temp_map != temp_keys):
            self.__log.debug("...mappings not necessary!")
        mask = np.isin(list(mappings[:, 0]), list(keys))
        mappings = mappings[mask]
        with h5py.File(s1_ref.data_path, "r+") as hf:
            del hf["mappings"]
            hf.create_dataset("mappings", data=np.array(
                mappings, DataSignature.string_dtype()))
        sign1.refresh()

    def overwrite(self, sign1, V, keys):
        sign1.refresh()
        self.reindex_triplets(sign1, keys)
        data_path = sign1.data_path
        sign1.close_hdf5()
        with h5py.File(data_path, "r+") as hf:
            if self.tmp:
                keys_ = hf["keys"][:]
                mask = np.isin(list(keys_), list(keys))
                del hf["V_tmp"]
                hf["V_tmp"] = V
                V = hf["V"][:][mask]
                del hf["V"]
                hf["V"] = V
            else:
                del hf["V"]
                hf["V"] = V
                if "V_tmp" in hf.keys():
                    self.__log.debug("Overwriting tmp with the actual dataset")
                    del hf["V_tmp"]
                    hf["V_tmp"] = V
            del hf["keys"]
            hf["keys"] = np.array(keys, DataSignature.string_dtype())
        self.remap(sign1)

    def save(self):
        self.__log.debug("Saving transformer object")
        fn = os.path.join(self.model_path, self.name + ".pkl")
        with open(fn, "wb") as f:
            pickle.dump(self, f)

    def predict_check(self, sign1):
        if sign1.cctype != "sign1":
            raise Exception("Only predictions for sign1 are accepted")

    def subsample(self):
        max_keys = self.max_keys
        if max_keys is None or max_keys >= self.sign_ref.shape[0]:
            self.__log.debug("Considering all reference data")
            keys = self.sign_ref.keys
            if self.tmp:
                V = self.sign_ref.get_h5_dataset("V_tmp")
            else:
                V = self.sign_ref.get_h5_dataset("V")
        else:
            self.__log.debug(
                "Subsampling data (ensuring coverage of at least one feature)")
            idxs = set()
            with h5py.File(self.sign_ref.data_path, "r") as hf:
                if self.tmp:
                    dkey = "V_tmp"
                else:
                    dkey = "V"
                for j in tqdm(range(0, self.sign_ref.shape[1])):
                    if len(idxs) >= self.max_keys:
                        break
                    v = hf[dkey][:, j]
                    non_zero_feat = np.argwhere(v != 0).ravel()
                    candidates = list(set(non_zero_feat) - idxs)
                    if len(candidates) == 0:
                        raise Exception(
                            'No feature specific candidates for subsampling. '
                            'This might be because data not being sanitized, '
                            'try using class `util.sanitize.Sanitizer`')
                    selected = np.random.choice(candidates,
                                                min(10, len(candidates)),
                                                replace=False)
                    idxs.update(selected)
            if len(idxs) < self.sign_ref.shape[1]:
                raise Exception('Could not subsample sufficiently, '
                                'please implement a strategy to sample more.')
            idxs = np.array(sorted(idxs))
            self.__log.debug("...%d subsampled" % len(idxs))
            with h5py.File(self.sign_ref.data_path, "r") as hf:
                if self.tmp:
                    V = hf["V_tmp"][idxs]
                else:
                    V = hf["V"][idxs]
            keys = np.array(self.sign_ref.keys)[idxs]
        self.__log.debug("...subsampling done")
        features = np.array(["f%d" % i for i in range(self.sign_ref.shape[1])])
        if self.categorical:
            V = V.astype(int)
        return V, keys, features

    @staticmethod
    def chunker(n, size=2000):
        for i in range(0, n, size):
            yield slice(i, i + size)

    def is_categorical(self, n=1000):
        self.__log.debug("Checking continuous or categorical")
        V = self.sign_ref[:n]
        self.sign_ref.close_hdf5()
        is_cat = np.all(V == V.astype(int))
        if not is_cat:
            return False
        V = self.sign[:n]
        self.sign.close_hdf5()
        is_cat = np.all(V == V.astype(int))
        if is_cat:
            return True
        else:
            return False
