"""Signature type 1.

Signatures type 1 are basically processed signatures. The typical preprocessing is a PCA (continuous data) or TF-IDF LSI (continuous).

Processing steps can be added as a pipeline previous to fitting the data.

A reference (non-redundant) dataset is always used for all of the fits.
"""
import os
import h5py
import numpy as np
import datetime
import shutil
from tqdm import tqdm
from scipy.spatial.distance import cosine
from .signature_data import DataSignature
from .signature_base import BaseSignature
from chemicalchecker.util import logged

@logged
class sign1(BaseSignature, DataSignature):
    """Signature type 1 class."""
    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): the path to the signature directory.
            model_path(str): Where the persistent model is.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(self.signature_path, "sign1.h5")
        DataSignature.__init__(self, self.data_path)

    def copy_sign0_to_sign1(self, s0, s1):
        """Copy from sign0 to sign1"""
        if s0.molset != s1.molset:
            raise Exception("Copying from signature 0 to 1 is only allowed for same molsets (reference or full)")
        self.__log.debug("Copying HDF5 dataset")
        with h5py.File(s1.data_path, "w") as hf:
            hf.create_dataset(
                "name", data=np.array([str(self.dataset) + "sig"], DataSignature.string_dtype()))
            hf.create_dataset(
                "date", data=np.array([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
            hf.create_dataset("V", data=s0[:])
            hf.create_dataset("keys", data=np.array(s0.keys, DataSignature.string_dtype()))
            if s0.molset == "reference":
                mappings = s0.get_h5_dataset("mappings")
                hf.create_dataset("mappings", data=np.array(mappings, DataSignature.string_dtype()))
        self.__log.debug("Copying triplets")
        fn0 = os.path.join(s0.model_path, "triplets.h5")
        fn1 = os.path.join(s1.model_path, "triplets.h5")
        shutil.copyfile(fn0, fn1)

    def fit(self, sign0):
        """Fit a signature 1, given a signature 0

            Args:
                sign0: A signature 0.
        """
        self.__log.debug("Fitting")
        if sign0.cctype != "sign0":
            raise Exception("A signature type 0 is expected..!")
        if sign0.molset != "full":
            raise Exception("Fit should be done with the full signature 0 (even if inside reference is used)")
        sign0_ref = sign0.get_molset("reference")
        sign1_ref = self.get_molset("reference")
        self.__log.debug("Placing sign0 to sign1 (done for reference)")
        self.copy_sign0_to_sign1(sign0_ref, sign1_ref)
        self.__log.debug("Placing sign0 to sign1 (done for full)")
        self.copy_sign0_to_sign1(sign0, self)

    
        #self.predict(self.sign0)
        
    def predict(self):
        pass


    def score(self, max_triplets=100000):
        """Score based on triplets"""
        self.__log.debug("Score the transformation based on triplets accuracy")
        with h5py.File(os.path.join(self.model_path, "triplets.h5"), "r") as hf:
            triplets = hf["triplets"][:]
        V = self[:]
        idxs = np.random.choice(triplets.shape[0], max_triplets, replace=False)
        triplets = triplets[idxs]
        acc = 0
        for t in triplets:
            if cosine(V[t[0]], V[t[1]]) < cosine(V[t[0]], V[t[2]]):
                acc += 1
        acc /= len(triplets)
        return acc

    def get_triplets(self):
        with h5py.File(os.path.join(self.model_path, "triplets.h5"), "r") as hf:
            triplets = hf["triplets"][:]
        return triplets