import os
import h5py
import numpy as np
try:
    import faiss
except ImportError:
    pass
import datetime
from time import time
from numpy import linalg as LA
from chemicalchecker.util import logged
from .signature_base import BaseSignature


@logged
class neig1(BaseSignature):
    """A Signature bla bla."""

    def __init__(self, signature_path, validation_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): the path to the signature directory.
            metric(str): The metric used in the KNN algorithm: euclidean or cosine (default: cosine)
            k_neig(int): The number of k neighbours to search for (default:1000)
            cpu(int): The number of cores to use (default:1)
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, validation_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(signature_path, "neig1.h5")
        self.__log.debug('data_path: %s', self.data_path)
        self.__log.debug('param file: %s', self.param_file)
        self.metric = "cosine"
        self.cpu = 1
        self.k_neig = 1000
        self.norms_file = os.path.join(self.model_path, "norms.h5")
        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)
            if "metric" in params:
                self.metric = params["metric"]
            if "cpu" in params:
                self.cpu = params["cpu"]
            if "k_neig" in params:
                self.k_neig = params["k_neig"]

    def fit(self, sign1):
        """Take an input and learns to produce an output."""
        BaseSignature.fit(self)

        faiss.omp_set_num_threads(self.cpu)

        mappings = None

        if os.path.isfile(sign1.data_path):
            dh5 = h5py.File(sign1.data_path)
            if "keys" not in dh5.keys() or "V" not in dh5.keys():
                raise Exception(
                    "H5 file " + sign1.data_path + " does not contain datasets 'keys' and 'V'")
            self.data = np.array(dh5["V"][:], dtype=np.float32)
            self.data_type = dh5["V"].dtype
            self.keys = dh5["keys"][:]
            if "mappings" in dh5.keys():
                mappings = dh5["mappings"][:]
            dh5.close()

        else:
            raise Exception("The file " + sign1.data_path + " does not exist")

        if self.metric == "euclidean":
            index = faiss.IndexFlatL2(self.data.shape[1])
        else:
            index = faiss.IndexFlatIP(self.data.shape[1])

        k = min(self.data.shape[0], self.k_neig)

        index.add(self.data)

        D, I = index.search(self.data, k)

        norms = LA.norm(self.data, axis=1)

        if self.metric == "cosine":

            t_start = time()
            mat = np.ones((self.data.shape[0], k))
            mat = mat / norms[:, None]
            for i in range(0, self.data.shape[0]):
                for j in range(0, k):
                    mat[i, j] = mat[i, j] / norms[I[i, j]]
            D = np.maximum(0.0, 1.0 - (D * mat))
            t_end = time()
            t_delta = str(datetime.timedelta(seconds=t_end - t_start))
            self.__log.info("Converting to cosine distance took %s", t_delta)

            # print D[0, 4], D1[0, 4]

        fout = h5py.File(self.data_path, 'w')

        fout.create_dataset("row_keys", data=self.keys)
        fout["col_keys"] = h5py.SoftLink('/row_keys')
        fout.create_dataset("indices", data=I)
        fout.create_dataset("distances", data=D)
        fout.create_dataset("shape", data=D.shape)
        fout.create_dataset(
            "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
        fout.create_dataset(
            "metric", data=[self.metric.encode(encoding='UTF-8', errors='strict')])
        if mappings is not None:
            fout.create_dataset("mappings", data=mappings)

        fout.close()

        faiss.write_index(index, os.path.join(
            self.model_path, "faiss_neig1.index"))

        with h5py.File(self.norms_file, "w") as hw:
            hw.create_dataset("norms", data=norms)

    def predict(self, sign1, destination=None):
        """Use the fitted models to go from input to output."""
        BaseSignature.predict(self)

        if destination is None:
            raise Exception("There is no destination file specified")

        faiss.omp_set_num_threads(self.cpu)

        if os.path.isfile(sign1.data_path):
            dh5 = h5py.File(sign1.data_path)
            if "keys" not in dh5.keys() or "V" not in dh5.keys():
                raise Exception(
                    "H5 file " + sign1.data_path + " does not contain datasets 'keys' and 'V'")
            self.data = np.array(dh5["V"][:], dtype=np.float32)
            self.data_type = dh5["V"].dtype
            self.keys = dh5["keys"][:]
            dh5.close()

        else:
            raise Exception("The file " + sign1.data_path + " does not exist")

        index = faiss.read_index(os.path.join(
            self.model_path, "faiss_neig1.index"))

        k = min(self.data.shape[0], self.k_neig)

        D, I = index.search(self.data, k)

        if self.metric == "cosine":
            norms = LA.norm(self.data, axis=1)

            with h5py.File(self.norms_file, "r") as hw:
                norms_fit = hw["norms"][:]

            t_start = time()
            mat = np.ones((self.data.shape[0], k))
            mat = mat / norms[:, None]
            for i in range(0, self.data.shape[0]):
                for j in range(0, k):
                    mat[i, j] = mat[i, j] / norms_fit[I[i, j]]
            D = np.maximum(0.0, 1.0 - (D * mat))
            t_end = time()
            t_delta = str(datetime.timedelta(seconds=t_end - t_start))
            self.__log.info("Converting to cosine distance took %s", t_delta)

        with h5py.File(self.data_path) as hr5:
            col_keys = hr5["row_keys"][:]

        fout = h5py.File(destination, 'w')

        fout.create_dataset("row_keys", data=self.keys)
        fout.create_dataset("col_keys", data=col_keys)
        fout.create_dataset("indices", data=I)
        fout.create_dataset("distances", data=D)
        fout.create_dataset("shape", data=D.shape)
        fout.create_dataset(
            "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
        fout.create_dataset(
            "metric", data=[self.metric.encode(encoding='UTF-8', errors='strict')])

        fout.close()

    def __iter__(self):
        """Iterate on neighbours indeces and distances."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            for i in range(self.shape[0]):
                yield hf['indices'][i], hf['distances'][i]
