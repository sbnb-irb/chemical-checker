"""Signature type 0 are basically raw features. Each bioactive space has
a peculiar format which might be categorial, discrete or continuous.
"""
import os
import imp
import h5py
import numpy as np
from tqdm import tqdm
import collections
import datetime
import random
import pickle

from .signature_data import DataSignature
from .signature_base import BaseSignature

from chemicalchecker.util import logged
from chemicalchecker.util.remove_near_duplicates import RNDuplicates
from chemicalchecker.util.sampler.triplets import TripletSampler

@logged
class sign0(BaseSignature, DataSignature):

    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): the path to the signature directory.
        """
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s' % signature_path)
        self.data_path = os.path.join(self.signature_path, "sign0.h5")
        DataSignature.__init__(self, self.data_path, **params)
        self.__log.debug('data path: %s' % self.data_path)

    def process_keys(self, keys, key_type):
        """Given keys, process them so they are acceptable CC types"""
        keys_ = []
        maps_ = []
        if key_type == "inchikey":
            self.__log.debug("Processing inchikeys. Only valids are kept.")
            for i,k in enumerate(keys):
                if len(k) == 27:
                    if k[14] == "-" and k[25] == "-":
                        keys_ += [k]
                        maps_ += [i]
        elif key_type == "smiles":
            self.__log.debug("Processing smiles. Only standard smiles are kept")
            from chemicalchecker.util.parser import Converter
            conv = Converter()
            for i,k in enumerate(keys):
                try:
                    keys_ += [conv.smiles_to_inchi(k)[0]]
                    maps_ += [i]
                except:
                    continue
        else:
            raise "key_type must be 'inchikey' or 'smiles'"
        self.__log.info("Initial keys: %d / Final keys: %d" % (len(keys), len(keys_)))
        return np.array(keys_), np.array(maps_)

    def process_features(self, features, n):
        """Process features. Give an arbitrary name to features if None are provided."""
        if features is None:
            self.__log.debug("No features were provided, giving arbitrary names")
            l = int(np.log10(n))+1
            features = []
            for i in range(0, n):
                s = "%d" % i
                s = s.zfill(l)
                features += ["feature_%s" % s]
        return np.array(features)

    def fit(self, cc, pairs=None, X=None, keys=None, key_type="inchikey", features=None, preprocess_func=None, **params):
        """Process the input data. We produce a sign0 (full) and a sign0(reference). Data are sorted (keys and features).
        
        Args:
            cc(Chemical Checker): .
            pairs(array of tuples): Data.
            X(matrix): Data.
            keys(array): Row names.
            key_type(str): Type of key. May be inchikey or smiles (default='inchikey').
            features(array): Column names (default=None).
            preprocess_func: A preprocessing function may be given. In this case X, keys and features are ignored (default=None).
        """
        if preprocess_func is not None:
            if X is not None or pairs is not None:
                raise Exception("If you input a function, X or pairs should not be specified!")
            self.__log.info("Input data was a preprocessing function")
            pass
        else:
            if pairs is not None:
                if X is not None:
                    raise Exception("If you input pairs, X should not be specified!")
                self.__log.info("Input data were pairs")
                keys  = list(set([x[0] for x in pairs]))
                features = list(set([x[1] for x in pairs]))
                self.__log.debug("Processing keys and features")
                keys, key_maps = self.process_keys(keys, key_type)
                features = self.process_features(features, len(features))
                keys_dict  = dict((k, i) for i, k in enumerate(keys))
                features_dict = dict((k, i) for i, k in enumerate(features))
                self.__log.debug("Iterating over pairs and doing matrix")
                pairs_ = collections.defaultdict(list)
                if len(pairs[0]) == 2:
                    self.__log.debug("Binary pairs")
                    for p in pairs:
                        pairs_[(keys_dict[p[0]], features_dict[p[1]])] += [1]
                else:
                    self.__log.debug("Valued pairs")
                    for p in pairs:
                        pairs_[(keys_dict[p[1]], features_dict[p[1]])] += [p[2]]
                X = np.zeros((len(keys), len(features)))
                self.__log.debug("Aggregating duplicates")
                for k,v in pairs_.items():
                    X[k[0], k[1]] = np.mean(v)
            else:
                if X is None:
                    raise Exception("No data were provided!")
                self.__log.debug("Processing keys")
                keys, key_maps   = self.process_keys(keys, key_type)
                self.__log.debug("Processing features")
                features = self.process_features(features, X.shape[1])
            self.__log.debug("Sorting")
            key_idxs  = np.argsort(keys)
            feature_idxs = np.argsort(features)
            # sort all data
            X = X[key_idxs]
            X = X[:,feature_idxs]
            # sort keys
            keys = keys[key_idxs]
            key_maps = key_maps[key_idxs]
            # sort features
            features = features[feature_idxs]
            self.__log.debug("Saving dataset")
            with h5py.File(self.data_path, "w") as hf:
                hf.create_dataset(
                    "name", data=np.array([str(self.dataset) + "sig"], DataSignature.string_dtype()))
                hf.create_dataset(
                    "date", data=np.array([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
                hf.create_dataset("V", data=X)
                hf.create_dataset("keys", data=np.array(keys, DataSignature.string_dtype()))
                hf.create_dataset("features", data=np.array(features, DataSignature.string_dtype()))
                hf.create_dataset("key_maps", data=key_maps)
                hf.create_dataset("feature_idxs", data=feature_idxs)
        self.__log.info("Removing redundancy")
        sign0_ref = self.get_molset("reference")
        rnd = RNDuplicates(cpu=10)
        rnd.remove(self.data_path, save_dest=sign0_ref.data_path)
        with h5py.File(self.data_path, "r") as hf:
            features = hf["features"][:]
        with h5py.File(sign0_ref.data_path, 'a') as hf:
            hf.create_dataset('features', data=features)
        self.features = features
        # Making triplets
        sampler = TripletSampler(cc, self, save=True)
        sampler.sample(**params)
        self.__log.debug("Done. Marking as ready.")
        # Marking as ready
        sign0_ref.mark_ready()
        self.mark_ready()

    def predict(self, pairs=None, X=None, keys=None, features=None):
        pass