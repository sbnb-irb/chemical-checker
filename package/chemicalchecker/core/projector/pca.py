import os
import h5py
import random
import pickle
import datetime
from tqdm import tqdm
import numpy as np
from time import time
from datetime import datetime
from datetime import timedelta
from sklearn.decomposition import IncrementalPCA as sklearnPCA

from chemicalchecker.core.signature_base import BaseSignature
from chemicalchecker.core.signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.util.plot import Plot


@logged
class PCA(BaseSignature, DataSignature):
    """A Signature bla bla."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the projection class.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)

        self.proj_name = self.__class__.__name__
        self.data_path = os.path.join(
            signature_path, "proj_%s.h5" % self.proj_name)
        self.model_path = os.path.join(self.model_path, self.proj_name)
        if not os.path.isdir(self.model_path):
            original_umask = os.umask(0)
            os.makedirs(self.model_path, 0o775)
            os.umask(original_umask)
        self.stats_path = os.path.join(self.stats_path, self.proj_name)
        if not os.path.isdir(self.stats_path):
            original_umask = os.umask(0)
            os.makedirs(self.stats_path, 0o775)
            os.umask(original_umask)
        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)
        self.algo_path = os.path.join(self.model_path, 'algo.pkl')
        if os.path.isfile(self.algo_path):
            self.algo = pickle.load(open(self.algo_path))
        else:
            self.algo = sklearnPCA(n_components=2, **params)
        self.name = "_".join([str(self.dataset), "proj", self.proj_name])
        self.proj_data = None

    def fit(self, signature, validations=True, chunk_size=100):
        # perform fit
        self.__log.info("Projecting with PCA...")
        t_start = time()
        with h5py.File(signature.data_path, "r") as src:
            src_len = src["V"].shape[0]
            for i in tqdm(range(0, src_len, chunk_size), 'fit'):
                chunk = slice(i, i + chunk_size)
                self.algo.partial_fit(src["V"][chunk])
        self.proj_data = list()
        with h5py.File(signature.data_path, "r") as src:
            src_len = src["V"].shape[0]
            for i in tqdm(range(0, src_len, chunk_size), 'transform'):
                chunk = slice(i, i + chunk_size)
                self.proj_data.append(self.algo.transform(src["V"][chunk]))
        self.proj_data = np.vstack(self.proj_data)
        t_end = time()
        t_delta = timedelta(seconds=t_end - t_start)
        self.__log.info("Projecting took %s" % t_delta)
        # save model
        pickle.dump(self.algo, open(self.algo_path, 'w'))
        # save h5
        sdtype = DataSignature.string_dtype()
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(self.data_path, "w") as dst:
            dst.create_dataset("keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=[self.name], dtype=sdtype)
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=[date_str], dtype=sdtype)
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, 2), dtype=np.float32)
            for i in tqdm(range(0, src_len, chunk_size), 'write'):
                chunk = slice(i, i + chunk_size)
                dst['V'][chunk] = self.algo.transform(src['V'][chunk])
        # run validation
        if validations:
            self.validate()
        self.mark_ready()

    def predict(self, signature, destination, chunk_size=100):
        """Predict new projections."""
        # create destination file
        sdtype = DataSignature.string_dtype()
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(destination, "w") as dst:
            dst.create_dataset("keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=[self.name], dtype=sdtype)
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=[date_str], dtype=sdtype)
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, 2), dtype=np.float32)
            for i in tqdm(range(0, src_len, chunk_size), 'transform'):
                chunk = slice(i, i + chunk_size)
                dst['V'][chunk] = self.algo.transform(src['V'][chunk])
