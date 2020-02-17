"""Signature type 0.

Signature type 0 are basically raw features. Each bioactive space has
a peculiar format which might be categorial, discrete or continous.
Given the diversity of formats and datasources the signaturization process
is started here but performed in tailored pre-process scripts (available
in the pipeline folder).
The `fit` method takes as input a matrix of keys and features, or it can invoke the pre-process script where
we essentially `learn` the feature to consider.
The `predict` argument (called by the `predict` method) allow deriving
signatures without altering the feature set. This can also be used when mapping
to a bioactive space different entities (i.e. not only compounds)
E.g.
categorical: "C0015230,C0016436..." which translates in n array of 0s or 1s.
discrete: "GO:0006897(8),GO:0006796(3),..." which translates in an array of
integers
continous: "0.515,1.690,0.996" which is an array of floats
"""
import os
import imp
import h5py
import numpy as np
from tqdm import tqdm

from .signature_data import DataSignature
from .signature_base import BaseSignature

from chemicalchecker.util import logged

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

    def process_feats(self, X, feats):
        if feats is None:
            self.__log.debug("No features were provided, giving arbitrary names")
            l = int(np.log10(X.shape[1]))+1
            feats = []
            for i in range(0, X.shape[1]):
                s = "%d" % i
                s = s.zfill(l)
                feats += ["feat_%s" % s]
        return np.array(feats)

    def process_keys_feats(self, X, keys, key_type, feats):
        self.__log.debug("Processing keys")
        keys, key_map = self.process_keys(keys, key_type)
        self.__log.debug("Sorting by inchikey")
        idxs = np.argsort(keys)
        keys = keys[idxs]
        key_map = key_map[idxs]
        X = X[idxs]
        self.__log.debug("Processing feats")
        feats = self.process_feats(X, feats)
        self.__log.debug("Sorting by feats")
        idxs  = np.argsort(feats)
        feats = feats[idxs]
        X = X[:,idxs]

    def fit(self, X=None, keys=None, key_type="inchikey", feats=None, preprocess_func=None):
        """Process the input data.
        
        Args:
            X(matrix): Data.
            keys(array): Row names.
            feats(array): Column names (default=None).
            preprocess_func: A preprocessing function may be given. In this case X, keys and feats are ignored (default=None).
        """
        if preprocess_func is not None:
            pass
        else:
            X, keys, feats, maps = self.process_keys_feats(X, keys, key_type, feats)


    def predict(self, X):
        pass

    def transform(self):
        self.predict(self)