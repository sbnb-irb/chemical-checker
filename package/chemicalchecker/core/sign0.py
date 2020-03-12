"""Signature type 0 are basically raw features. Each bioactive space has
a peculiar format which might be categorical, discrete or continuous.
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
from .signature_data import cached_property

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
        """Given keys, process them so they are acceptable CC types. If None is specified, then all keys are kept."""
        if key_type is None:
            return np.array(keys), None
        keys_ = []
        keys_raw = []
        if key_type == "inchikey":
            self.__log.debug("Processing inchikeys. Only valids are kept.")
            for i, k in enumerate(keys):
                if len(k) == 27:
                    if k[14] == "-" and k[25] == "-":
                        keys_ += [k]
                        keys_raw += [k]
        elif key_type == "smiles":
            self.__log.debug(
                "Processing smiles. Only standard smiles are kept")
            from chemicalchecker.util.parser import Converter
            conv = Converter()
            for i, k in enumerate(keys):
                try:
                    keys_ += [conv.smiles_to_inchi(k)[0]]
                    keys_raw += [k]
                except:
                    continue
        else:
            raise "key_type must be 'inchikey' or 'smiles'"
        self.__log.info("Initial keys: %d / Final keys: %d" %
                        (len(keys), len(keys_)))
        return np.array(keys_), np.array(keys_raw)

    def process_features(self, features, n):
        """Process features. Give an arbitrary name to features if None are provided."""
        if features is None:
            self.__log.debug(
                "No features were provided, giving arbitrary names")
            l = int(np.log10(n)) + 1
            features = []
            for i in range(0, n):
                s = "%d" % i
                s = s.zfill(l)
                features += ["feature_%s" % s]
        return np.array(features)

    def get_data(self, pairs, X, keys, features, key_type):
        if pairs is not None:
            if X is not None:
                raise Exception(
                    "If you input pairs, X should not be specified!")
            if type(pairs) == str:
                self.__log.debug("Pairs input file is: " + pairs)
                if os.path.isfile(pairs) and pairs[-3:] == ".h5":
                    dh5 = h5py.File(pairs, 'r')
                    if "pairs" not in dh5.keys():
                        raise Exception(
                            "H5 file " + pairs + " does not contain datasets 'pairs'")
                    pairs = dh5["pairs"][:]
                    # TODO ORIOL
                    dh5.close()
                else:
                    raise Exception("This module only accepts .h5 files")
            else:
                if len(pairs[0]) == 2:
                    has_values = False
                else:
                    has_values = True
                self.__log.debug("Data provided in memory.")
            self.__log.info("Input data were pairs")
            keys = list(set([x[0] for x in pairs]))
            features = list(set([x[1] for x in pairs]))
            self.__log.debug("Processing keys and features")
            keys, keys_raw = self.process_keys(keys, key_type)
            features = self.process_features(features, len(features))
            keys_dict = dict((k, i) for i, k in enumerate(keys))
            features_dict = dict((k, i) for i, k in enumerate(features))
            self.__log.debug("Iterating over pairs and doing matrix")
            pairs_ = collections.defaultdict(list)
            if not has_values:
                self.__log.debug("Binary pairs")
                for p in pairs:
                    pairs_[(keys_dict[p[0]], features_dict[p[1]])] += [1]
            else:
                self.__log.debug("Valued pairs")
                for p in pairs:
                    pairs_[(keys_dict[p[1]], features_dict[p[1]])] += [p[2]]
            X = np.zeros((len(keys), len(features)))
            self.__log.debug("Aggregating duplicates")
            for k, v in pairs_.items():
                X[k[0], k[1]] = np.mean(v)
        else:
            if X is None:
                raise Exception("No data were provided! X cannot be None if pairs aren't provided")
            if type(X) == str:
                self.__log.debug("Data input file is: " + pairs)
                if os.path.isfile(X) and X[-3:] == ".h5":
                    dh5 = h5py.File(X, 'r')
                    if "X" not in dh5.keys() or "keys" not in dh5.keys():
                        raise Exception(
                            "H5 file " + X + " does not contain datasets 'X' or 'keys'")
                    X = dh5["X"][:]
                    keys = dh5["keys"][:]
                    if "features" in dh5.keys():
                        features = dh5["features"][:]
                    dh5.close()
                else:
                    raise Exception("This module only accepts .h5 files")
            else:
                self.__log.debug("Data provided in memory")
                if keys is None:
                    raise Exception("keys cannot be None")
                if features is None:
                    raise Exception("features cannot be None")
            self.__log.debug("Processing keys")
            keys, keys_raw = self.process_keys(keys, key_type)
            self.__log.debug("Processing features")
            features = self.process_features(features, X.shape[1])
        results = {
            "X": X,
            "keys": keys,
            "keys_raw": keys_raw,
            "features": features
        }
        return results

    @cached_property
    def agg_method(self):
        """Get the agg method of the signature."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if "agg_method" not in hf.keys():
                self.__log.warn("No agg_method available for this signature!")
                return None
            if hasattr(hf["agg_method"][0], 'decode'):
                return [k.decode() for k in hf["agg_method"][:]][0]
            else:
                return hf["agg_method"][0]

    def fit(self, cc=None, pairs=None, X=None, keys=None, features=None, key_type="inchikey", agg_method="average", **params):
        """Process the input data. We produce a sign0 (full) and a sign0(reference). Data are sorted (keys and features).

        Args:
            cc(Chemical Checker): A CC instance. This is important to produce the triplets. If None specified, the same CC where the signature is present will be used (default=None).
            pairs(array of tuples or file): Data. If file it needs to H5 file with dataset called 'pairs'.
            X(matrix or file): Data. If file it needs to H5 file with datasets called 'X', 'keys' and maybe 'features'.
            keys(array): Row names.
            key_type(str): Type of key. May be inchikey or smiles (default='inchikey').
            features(array): Column names (default=None).
        """
        if cc is None:
            cc = self.get_cc()
        self.__log.debug("Getting data")
        res = self.get_data(pairs=pairs, X=X, keys=keys, features=features, key_type=key_type)
        X = res["X"]
        keys = res["keys"]
        keys_raw = res["keys_raw"]
        features = res["features"]
        self.__log.debug("Sorting")
        key_idxs = np.argsort(keys)
        feature_idxs = np.argsort(features)
        # sort all data
        X = X[key_idxs]
        X = X[:, feature_idxs]
        # sort keys
        keys = keys[key_idxs]
        keys_raw = keys_raw[key_idxs]
        # sort features
        features = features[feature_idxs]
        self.__log.debug("Aggregating if necessary")
        agg = Aggregate(method=agg_method)
        X, keys, keys_raw = agg.transform(V=X, keys=keys, keys_raw=keys_raw)
        self.__log.debug("Saving dataset")
        with h5py.File(self.data_path, "w") as hf:
            hf.create_dataset(
                "name", data=np.array([str(self.dataset) + "sig"], DataSignature.string_dtype()))
            hf.create_dataset(
                "date", data=np.array([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
            hf.create_dataset("V", data=X)
            hf.create_dataset("keys", data=np.array(
                keys, DataSignature.string_dtype()))
            hf.create_dataset("features", data=np.array(
                features, DataSignature.string_dtype()))
            hf.create_dataset("keys_raw", data=np.array(
                keys_raw, DataSignature.string_dtype()))
            hf.create_dataset("agg_method", data=np.array([str(agg_method)], DataSignature.string_dtype()))
        self.__log.info("Removing redundancy")
        sign0_ref = self.get_molset("reference")
        rnd = RNDuplicates(cpu=10)
        rnd.remove(self.data_path, save_dest=sign0_ref.data_path)
        with h5py.File(self.data_path, "r") as hf:
            features = hf["features"][:]
        with h5py.File(sign0_ref.data_path, 'a') as hf:
            hf.create_dataset('features', data=features)
        # Making triplets
        sampler = TripletSampler(cc, self, save=True)
        sampler.sample(**params)
        self.__log.debug("Done. Marking as ready.")
        # Marking as ready
        sign0_ref.mark_ready()
        self.mark_ready()

    def predict(self, pairs=None, X=None, keys=None, features=None, key_type=None, merge=False, merge_method="new", destination=None):
        """Given data, produce a sign0.

        Args:
            pairs(array of tuples or file): Data. If file it needs to H5 file with dataset called 'pairs'.
            X(matrix or file): Data. If file it needs to H5 file with datasets called 'X', 'keys' and maybe 'features'.
            keys(array): Row names.
            key_type(str): Type of key. May be inchikey or smiles. If None specified, no filtering is applied (default=None).
            features(array): Column names (default=None).
            merge(bool): Merge queried data with the currently existing one.
            merge_method(str): Merging method to be applied when a repeated key is found. Can be 'average', 'old' or 'new' (default=new).
            destination(str): Path to the H5 file. If none specified, a (V, keys, features) tuple is returned. 
        """
        assert self.is_fit(), "Signature is not fitted yet"
        self.__log.debug("Setting up the signature data based on fit")
        if merge:
            self.__log.info("Merging. Loading existing signature.")
            V_ = self[:]
            keys_ = self.keys
            keys_raw_ = self.keys_raw
        else:
            self.__log.info("Not merging. Just producing signature for the inputted data.")
            V_ = None
            keys_ = None
            keys_raw_ = None
        features_ = self.features
        features_idx = dict((k,i) for i,k in enumerate(features_))
        self.__log.debug("Preparing input data")
        res = self.get_data(pairs=pairs, X=X, keys=keys, features=features, key_type=key_type)
        X = res["X"]
        keys = res["keys"]
        keys_raw = res["keys_raw"]
        self.__log.debug("Aggregating as it was done at fit time")
        agg = Aggregate(method=self.agg_method)
        X, keys, keys_raw = agg.transform(V=X, keys=keys, keys_raw=keys_raw)
        features = res["features"]
        self.__log.debug("Putting input in the same features arrangement than the fitted signature.")
        W = np.zeros(len(keys), len(features_))
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                feat = features[j]
                if feat not in features_idx:
                    continue
                W[i, features_idx[feat]] = X[i,j]
        self.__log.debug("Refactoring")
        features = features_
        if V_ is None:
            V = W
        else:
            self.__log.debug("Stacking")
            V = np.vstack((V_, W))
            keys = np.append(keys_, keys)
            keys_raw = np.append(keys_raw_, keys_raw)
            self.__log.debug("Aggregating (merging) again")
            agg = Aggregate(method=agg_method)
            V, keys, keys_raw = agg.transform(V=V, keys=keys, keys_raw=keys_raw)
        
        self.__log.debug("Done")
        if destination is None:
            self.__log.debug("Returning a dictionary of V, keys, features and keys_raw")
            results = {
                "V": V,
                "keys": keys,
                "features": features,
                "keys_raw": keys_raw
            }
            return results
        else:
            self.__log.debug("Saving H5 file in %s" % destination)
            with h5py.File(self.data_path, "w") as hf:
                hf.create_dataset(
                    "name", data=np.array([str(self.dataset) + "sig"], DataSignature.string_dtype()))
                hf.create_dataset(
                    "date", data=np.array([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
                hf.create_dataset("V", data=X)
                hf.create_dataset("keys", data=np.array(
                    keys, DataSignature.string_dtype()))
                hf.create_dataset("features", data=np.array(
                    features, DataSignature.string_dtype()))
                hf.create_dataset("keys_raw", data=np.array(
                    keys_raw, DataSignature.string_dtype()))