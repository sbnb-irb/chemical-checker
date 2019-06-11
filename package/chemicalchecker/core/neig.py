import os
import h5py
import datetime
import numpy as np
from time import time
from numpy import linalg as LA

from .signature_base import BaseSignature

from chemicalchecker.util import logged


@logged
class neig(BaseSignature):
    """A Signature bla bla."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): the path to the signature directory.
            metric(str): The metric used in the KNN algorithm: euclidean or cosine (default: cosine)
            k_neig(int): The number of k neighbours to search for (default:1000)
            cpu(int): The number of cores to use (default:1)
            chunk(int): The size of the chunk to read the data (default:1000)
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(signature_path, "neig.h5")
        self.__log.debug('data_path: %s', self.data_path)
        self.metric = "cosine"
        self.cpu = 1
        self.chunk = 1000
        self.k_neig = 1000
        self.norms_file = os.path.join(self.model_path, "norms.h5")
        self.index_filename = os.path.join(self.model_path, 'faiss_neig.index')
        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)
            if "metric" in params:
                self.metric = params["metric"]
            if "cpu" in params:
                self.cpu = params["cpu"]
            if "k_neig" in params:
                self.k_neig = params["k_neig"]
            if "chunk" in params:
                self.chunk = params["chunk"]

    def fit(self, sign1):
        """Take an input and learns to produce an output."""
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")
        BaseSignature.fit(self)

        faiss.omp_set_num_threads(self.cpu)

        if os.path.isfile(sign1.data_path):
            with h5py.File(sign1.data_path) as dh5, h5py.File(self.data_path, 'w') as dh5out:
                if "keys" not in dh5.keys() or "V" not in dh5.keys():
                    raise Exception(
                        "H5 file " + sign1.data_path + " does not contain datasets 'keys' and 'V'")

                self.datasize = dh5["V"].shape
                self.data_type = dh5["V"].dtype

                k = min(self.datasize[0], self.k_neig)

                dh5out.create_dataset("row_keys", data=dh5["keys"][:])
                dh5out["col_keys"] = h5py.SoftLink('/row_keys')
                dh5out.create_dataset(
                    "indices", (self.datasize[0], k), dtype=np.int32)
                dh5out.create_dataset(
                    "distances", (self.datasize[0], k), dtype=np.float32)
                dh5out.create_dataset("shape", data=(self.datasize[0], k))
                dh5out.create_dataset(
                    "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
                dh5out.create_dataset(
                    "metric", data=[self.metric.encode(encoding='UTF-8', errors='strict')])

                if self.metric == "euclidean":
                    index = faiss.IndexFlatL2(self.datasize[1])
                else:
                    index = faiss.IndexFlatIP(self.datasize[1])

                norms = None

                for chunk in sign1.chunker():
                    data_temp = np.array(dh5["V"][chunk], dtype=np.float32)
                    index.add(data_temp)

                for chunk in sign1.chunker():
                    data_temp = np.array(dh5["V"][chunk], dtype=np.float32)
                    Dt, It = index.search(data_temp, k)

                    if self.metric == "cosine":
                        normst = LA.norm(data_temp, axis=1)

                        if norms is None:
                            norms = normst
                        else:
                            norms = np.concatenate((norms, normst))

                    dh5out["indices"][chunk] = It
                    dh5out["distances"][chunk] = Dt

            if self.metric == "cosine":

                with h5py.File(self.data_path, "r+") as hw:
                    t_start = time()
                    mat = np.ones((self.datasize[0], k))
                    for chunk in sign1.chunker():
                        mat[chunk] = mat[chunk] / norms[chunk, None]
                    # We load all to make it faster, if memory issue then we need
                    # to get each element one by one
                    I = hw["indices"][:]
                    for i in range(0, self.datasize[0]):
                        for j in range(0, k):
                            mat[i, j] = mat[i, j] / norms[I[i, j]]
                    del I
                    for chunk in sign1.chunker():
                        hw["distances"][chunk] = np.maximum(
                            0.0, 1.0 - (hw["distances"][chunk] * mat[chunk]))
                    t_end = time()
                    t_delta = str(datetime.timedelta(seconds=t_end - t_start))
                    self.__log.info(
                        "Converting to cosine distance took %s", t_delta)

                with h5py.File(self.norms_file, "w") as hw:
                    hw.create_dataset("norms", data=norms)
        else:
            raise Exception("The file " + sign1.data_path + " does not exist")

        faiss.write_index(index, self.index_filename)

        self.mark_ready()

    def predict(self, sign1, destination=None):
        """Use the fitted models to go from input to output."""
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")
        BaseSignature.predict(self)

        if destination is None:
            raise Exception("There is no destination file specified")

        faiss.omp_set_num_threads(self.cpu)

        if os.path.isfile(sign1.data_path):
            with h5py.File(sign1.data_path) as dh5, h5py.File(destination, 'w') as dh5out:
                if "keys" not in dh5.keys() or "V" not in dh5.keys():
                    raise Exception(
                        "H5 file " + sign1.data_path + " does not contain datasets 'keys' and 'V'")

                self.datasize = dh5["V"].shape
                self.data_type = dh5["V"].dtype

                k = min(self.datasize[0], self.k_neig)

                dh5out.create_dataset("row_keys", data=dh5["keys"][:])
                with h5py.File(self.data_path) as hr5:
                    dh5out.create_dataset("col_keys", data=hr5["row_keys"][:])
                dh5out.create_dataset(
                    "indices", (self.datasize[0], k), dtype=np.int32)
                dh5out.create_dataset(
                    "distances", (self.datasize[0], k), dtype=np.float32)
                dh5out.create_dataset("shape", data=(self.datasize[0], k))
                dh5out.create_dataset(
                    "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
                dh5out.create_dataset(
                    "metric", data=[self.metric.encode(encoding='UTF-8', errors='strict')])

                norms = None

                index = faiss.read_index(self.index_filename)

                for chunk in sign1.chunker():
                    data_temp = np.array(dh5["V"][chunk], dtype=np.float32)
                    Dt, It = index.search(data_temp, k)

                    if self.metric == "cosine":
                        normst = LA.norm(data_temp, axis=1)

                        if norms is None:
                            norms = normst
                        else:
                            norms = np.concatenate((norms, normst))

                    dh5out["indices"][chunk] = It
                    dh5out["distances"][chunk] = Dt

        else:
            raise Exception("The file " + sign1.data_path + " does not exist")

        if self.metric == "cosine":

            with h5py.File(self.norms_file, "r") as hw:
                norms_fit = hw["norms"][:]

            with h5py.File(destination, "r+") as hw:
                t_start = time()
                mat = np.ones((self.datasize[0], k))
                for chunk in sign1.chunker():
                    mat[chunk] = mat[chunk] / norms[chunk, None]
                # We load all to make it faster, if memory issue then we need
                # to get each element one by one
                I = hw["indices"][:]
                for i in range(0, self.datasize[0]):
                    for j in range(0, k):
                        mat[i, j] = mat[i, j] / norms_fit[I[i, j]]
                del I
                for chunk in sign1.chunker():
                    hw["distances"][chunk] = np.maximum(
                        0.0, 1.0 - (hw["distances"][chunk] * mat[chunk]))
                t_end = time()
                t_delta = str(datetime.timedelta(seconds=t_end - t_start))
                self.__log.info(
                    "Converting to cosine distance took %s", t_delta)

    def get_kth_nearest(self, signatures, k=None):
        """Return up to the k-th nearest neighbor.

        This function returns the k-th closest neighbor.
        A k>1 is useful when we expect and want to exclude a perfect match,
        i.e. when the signature we query for are the same that have been used
        to generate the neighbors.

        Args:
            signatures(array): Matrix or list of signatures for which we want
                to find neighbors.
            k(int): Amount of neigbors to find, if None return the maximum
                possible.
        Returns:
            dict with keys: 
                1. 'indeces' the indeces of neighbors 
                2. 'keys' the inchikey of neighbors
                3. 'distances' the cosine distances.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")
        # open faiss model
        faiss.omp_set_num_threads(self.cpu)
        index = faiss.read_index(self.index_filename)
        # decide K
        max_k = index.ntotal
        if k is None:
            k = max_k
        if k > max_k:
            self.__log.warning("Maximum k is %s.", max_k)
            k = max_k
        # convert signatures to float32 as faiss is very picky
        data = np.array(signatures, dtype=np.float32)
        # get neighbors idx and distances 
        dists, idx = index.search(data, k)
        predictions = dict()
        predictions["distances"] = dists
        predictions["indices"] = idx
        with h5py.File(self.data_path, 'r') as hf:
            keys = hf['col_keys'][:]
        predictions["keys"] = keys[idx]
        # convert distances to cosine
        norms = LA.norm(data, axis=1)
        with h5py.File(self.norms_file, "r") as hw:
            norms_fit = hw["norms"][:]
        t_start = time()
        mat = np.ones((len(signatures), k))
        mat = mat / norms[:, None]
        I = predictions["indices"]
        for i in range(0, len(signatures)):
            for j in range(0, k):
                mat[i, j] = mat[i, j] / norms_fit[I[i, j]]
            predictions["distances"] = np.maximum(
                0.0, 1.0 - (predictions["distances"] * mat))
        t_delta = str(datetime.timedelta(seconds=time() - t_start))
        self.__log.info(
            "Converting to cosine distance took %s", t_delta)
        return predictions

    @staticmethod
    def jaccard_similarity(n1, n2):
        """Compute Jaccard similarity.

        Args:
            n1(np.array): First set of neighbors, row are molecule each
                column the idx of a neighbor
            n1(np.array): Second set of neighbors, row are molecule each
                column the idx of a neighbor
        """
        res = list()
        for r1, r2 in zip(n1, n2):
            s1 = set(r1)
            s2 = set(r2)
            inter = len(set.intersection(s1, s2))
            uni = len(set.union(s1, s2))
            res.append(inter / float(uni))
        return np.array(res)

    def __iter__(self):
        """Iterate on neighbours indeces and distances."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            for i in range(self.shape[0]):
                yield hf['indices'][i], hf['distances'][i]
