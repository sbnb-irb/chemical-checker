"""Nearest Neighbor Signature.

Identify nearest neighbors and distances.
"""
import os
import h5py
import datetime
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from bisect import bisect_left
from scipy.spatial.distance import euclidean, cosine

from .signature_base import BaseSignature
from .signature_data import DataSignature


from chemicalchecker.util import logged
from chemicalchecker.util.decorator import cached_property


@logged
class neig(BaseSignature, DataSignature):
    """Neighbors Signature class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize a Signature.

        Args:
            signature_path(str): the path to the signature directory.
            metric(str): The metric used in the KNN algorithm: euclidean or
                cosine (default: cosine)
            k_neig(int): The number of k neighbours to search for
                (default:1000)
            cpu(int): The number of cores to use (default:1)
            chunk(int): The size of the chunk to read the data (default:1000)
        """
        
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(signature_path, "neig.h5")
        self.__log.debug('data_path: %s', self.data_path)
        DataSignature.__init__(self, self.data_path,
                               ds_data='distances', keys_name='row_keys')
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

    def fit(self, sign=None):
        """Fit neighbor model given a signature."""
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")

        # signature specific checks
        if self.molset != "reference":
            self.__log.debug("Fit will be done with the reference neig1")
            self = self.get_molset("reference")
        if sign is None:
            sign = self.get_sign(
                'sign' + self.cctype[-1]).get_molset("reference")
        if sign.molset != "reference":
            self.__log.debug("Fit will be done with the reference sign")
            sign = self.get_sign(
                'sign' + self.cctype[-1]).get_molset("reference")
        if not sign.is_fit():
            raise Exception("sign is not fitted.")

        faiss.omp_set_num_threads(self.cpu)

        if os.path.isfile(sign.data_path):
            with h5py.File(sign.data_path, 'r') as dh5, h5py.File(self.data_path, 'w') as dh5out:
                if "keys" not in dh5.keys() or "V" not in dh5.keys():
                    raise Exception(
                        "H5 file " + sign.data_path + " does not contain datasets 'keys' and 'V'")

                self.datasize = dh5["V"].shape
                self.data_type = dh5["V"].dtype

                k = min(self.datasize[0], self.k_neig)

                dh5out.create_dataset("row_keys", data=dh5["keys"][:])
                dh5out["col_keys"] = h5py.SoftLink('/row_keys')
                dh5out.create_dataset(
                    "indices", (self.datasize[0], k), dtype='int32' )
                dh5out.create_dataset(
                    "distances", (self.datasize[0], k), dtype='float32')
                dh5out.create_dataset("shape", data=(self.datasize[0], k))
                dh5out.create_dataset(
                    "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
                dh5out.create_dataset(
                    "metric", data=[self.metric.encode(encoding='UTF-8', errors='strict')])

                if self.metric == "euclidean":
                    index = faiss.IndexFlatL2(self.datasize[1])
                else:
                    index = faiss.IndexFlatIP(self.datasize[1])

                for chunk in sign.chunker():
                    data_temp = np.array(dh5["V"][chunk], dtype='float32')
                    if self.metric == "cosine":
                        normst = LA.norm(data_temp, axis=1)
                        index.add( np.array( data_temp / normst[:, None], dtype='float32') )
                    else:
                        index.add( np.array(data_temp, dtype='float32')  )

                for chunk in sign.chunker():
                    data_temp = np.array(dh5["V"][chunk], dtype=float)
                    if self.metric == "cosine":
                        normst = LA.norm(data_temp, axis=1)
                        Dt, It = index.search( np.array(data_temp / normst[:, None], dtype='float32'), k)
                    else:
                        Dt, It = index.search( np.array(data_temp, dtype='float32'), k)

                    dh5out["indices"][chunk] = It
                    if self.metric == "cosine":
                        dh5out["distances"][chunk] = np.maximum(0.0, 1.0 - Dt)
                    else:
                        dh5out["distances"][chunk] = Dt

        else:
            raise Exception("The file " + sign.data_path + " does not exist")

        faiss.write_index(index, self.index_filename)

        # also predict for full if available
        sign_full = self.get_sign('sign' + self.cctype[-1]).get_molset("full")
        if os.path.isfile(sign_full.data_path):
            self.predict(sign_full, self.get_molset("full").data_path)
        self.mark_ready()

    def predict(self, sign, destination=None, validations=False):
        """Use the fitted models to go from input to output."""
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")

        if destination is None:
            raise Exception("There is no destination file specified")

        faiss.omp_set_num_threads(self.cpu)

        if os.path.isfile(sign.data_path):
            with h5py.File(sign.data_path, 'r') as dh5, h5py.File(destination, 'w') as dh5out:
                if "keys" not in dh5.keys() or "V" not in dh5.keys():
                    raise Exception(
                        "H5 file " + sign.data_path + " does not contain datasets 'keys' and 'V'")

                self.datasize = dh5["V"].shape
                self.data_type = dh5["V"].dtype

                index = faiss.read_index(self.index_filename)

                k = min(self.k_neig, index.ntotal)

                dh5out.create_dataset("row_keys", data=dh5["keys"][:])
                with h5py.File(self.data_path, 'r') as hr5:
                    dh5out.create_dataset("col_keys", data=hr5["row_keys"][:])
                dh5out.create_dataset(
                    "indices", (self.datasize[0], k), dtype='int32' )
                dh5out.create_dataset(
                    "distances", (self.datasize[0], k), dtype='float32')
                dh5out.create_dataset("shape", data=(self.datasize[0], k))
                dh5out.create_dataset(
                    "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
                dh5out.create_dataset(
                    "metric", data=[self.metric.encode(encoding='UTF-8', errors='strict')])

                for chunk in sign.chunker():
                    data_temp = np.array(dh5["V"][chunk], dtype=float)

                    if self.metric == "cosine":
                        normst = LA.norm(data_temp, axis=1)
                        Dt, It = index.search( np.array(data_temp / normst[:, None], dtype='float32'), k)
                    else:
                        Dt, It = index.search( np.array(data_temp, dtype='float32'), k)

                    dh5out["indices"][chunk] = It
                    if self.metric == "cosine":
                        dh5out["distances"][chunk] = np.maximum(0.0, 1.0 - Dt)
                    else:
                        dh5out["distances"][chunk] = Dt

        else:
            raise Exception("The file " + sign.data_path + " does not exist")

    @cached_property
    def keys(self):
        """Get the list of keys (usually inchikeys) in the signature."""
        self._check_data()
        self._check_dataset('row_keys')
        return self._get_all('row_keys')

    @cached_property
    def unique_keys(self):
        """Get the keys of the signature as a set."""
        return set(self.keys)

    def __getitem__(self, key):
        """Return the neighbours corresponding to the key.

        The key can be a string (then it's mapped though self.keys) or and
        int.
        Works fast with bisect, but should return None if the key is not in
        keys (ideally, keep a set to do this).

        Returns:
            dict with keys:
                1. 'indices' the indices of neighbors
                2. 'keys' the inchikey of neighbors
                3. 'distances' the cosine distances.
        """
        predictions = dict()

        if not os.path.isfile(self.data_path):
            raise Exception("Data file not available.")
        if isinstance(key, slice):
            with h5py.File(self.data_path, 'r') as hf:
                predictions["indices"] = hf['indices'][key]
                predictions["distances"] = hf['distances'][key]
                keys = hf['col_keys'][:].astype(str)
                predictions["keys"] = keys[predictions["indices"]]
        elif isinstance(key, str):
            if key not in self.unique_keys:
                raise Exception("Key '%s' not found." % key)
            idx = bisect_left(self.keys, key)
            with h5py.File(self.data_path, 'r') as hf:
                predictions["indices"] = hf['indices'][idx]
                predictions["distances"] = hf['distances'][idx]
                keys = hf['col_keys'][:].astype(str)
                predictions["keys"] = keys[predictions["indices"]]
        elif isinstance(key, int):
            with h5py.File(self.data_path, 'r') as hf:
                predictions["indices"] = hf['indices'][key]
                predictions["distances"] = hf['distances'][key]
                keys = hf['col_keys'][:].astype(str)
                predictions["keys"] = keys[predictions["indices"]]
        else:
            raise Exception("Key type %s not recognized." % type(key))

        return predictions

    def get_vectors(self, keys, include_nan=False, dataset_name='indices'):
        """Get vectors for a list of keys, sorted by default.

        Args:
            keys(list): a List of string, only the overlapping subset to the
                signature keys is considered.
            include_nan(bool): whether to include requested but absent
                molecule signatures as NaNs.
            dataset_name(str): return any dataset in the h5 which is organized
                by sorted keys.
        """
        self.__log.debug("Fetching Neig %s rows from dataset %s" %
                         (len(keys), dataset_name))
        valid_keys = list(set(self.row_keys) & set(keys))
        idxs = np.argwhere(
            np.isin(list(self.row_keys), list(valid_keys), assume_unique=True))
        inks, signs = list(), list()
        with h5py.File(self.data_path, 'r') as hf:
            dset = hf[dataset_name]
            col_keys = hf['col_keys'][:].astype(str)
            dset_shape = dset.shape
            for idx in sorted(idxs.flatten()):
                inks.append(self.row_keys[idx].astype(str))
                if dataset_name == 'indices':
                    signs.append(col_keys[dset[idx]])
                else:
                    signs.append(dset[idx])
        missed_inks = set(keys) - set(inks)
        # if missing signatures are requested add NaNs
        if include_nan:
            inks.extend(list(missed_inks))
            dimensions = (len(missed_inks), dset_shape[1])
            nan_matrix = np.zeros(dimensions) * np.nan
            signs.append(nan_matrix)
            if missed_inks:
                self.__log.info("NaN for %s requested keys as are not available.",
                                len(missed_inks))
        elif missed_inks:
            self.__log.warn("Following %s requested keys are not available:",
                            len(missed_inks))
            self.__log.warn(" ".join(list(missed_inks)[:10]) + "...")
        if len(inks) == 0:
            self.__log.warn("No requested keys available!")
            return None, None
        inks, signs = np.stack(inks), np.vstack(signs)
        sort_idx = np.argsort(inks)
        return inks[sort_idx], signs[sort_idx]

    def get_vectors_multiple_datasets(self, keys, include_nan=False, dataset_names = [], max_neighbors=500 ):
        """Get vectors for a list of keys, sorted by default.

        Args:
            keys(list): a List of string, only the overlapping subset to the
                signature keys is considered.
            dataset_name(str): return any datasets in the h5 which is organized
                by sorted keys.
        """
        self.__log.debug("Fetching %s rows from datasets" % ( len(keys) ) )
        
        valid_keys = list( set( self.row_keys ) & set( keys ) )
        idxs = np.argwhere( np.isin( list(self.row_keys), list(valid_keys), assume_unique=True) )
        idxs = idxs.flatten()
        oidxs = sorted(idxs)
        
        ncols = max_neighbors
        data = {}
        for d in dataset_names:
            data[d] = { 'inks': np.array([]), 'signs': np.full( ( 0, ncols ), np.nan ) }
        
        if( len(oidxs) > 0 ):
            inks = np.array( self.row_keys[ oidxs ].astype(str) )
            
            with h5py.File(self.data_path, 'r') as hf:
                col_keys = hf['col_keys'][:].astype(str)
                
                for d in dataset_names:
                    ncols = hf[d].shape[1]
                    data[d]['inks'] = inks
                    
                    temp = hf[d][oidxs, :max_neighbors]
                    if( d == 'indices' ):
                        data[d]['signs'] = list( map( lambda x: col_keys[ x ], temp ) )
                    else:
                        data[d]['signs'] = temp
                    data[d]['signs'] = np.array( data[d]['signs'] )
        
        missed_inks = set( list(keys) ) - set(valid_keys)
        # if missing signatures are requested add NaNs
            
        for d in dataset_names:
            inks, signs = data[d]['inks'], data[d]['signs']
            
            if include_nan:
                if( len(missed_inks) > 0 ):
                    inks = inks.tolist()
                    inks.extend( list(missed_inks) )
                    dimensions = ( len(missed_inks), ncols )
                    nan_matrix = np.zeros(dimensions) * np.nan
                    inks, signs = np.array( inks ), np.vstack( (signs, nan_matrix) )
            
            data[d]['inks'], data[d]['signs'] = inks, signs
            sort_idx = np.argsort( data[d]['inks'] )
            data[d]['inks'], data[d]['signs'] = data[d]['inks'][sort_idx], data[d]['signs'][sort_idx]
            
        return data

    def get_kth_nearest(self, signatures, k=1000, distances=True, keys=True):
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
                1. 'indices' the indices of neighbors
                2. 'keys' the inchikey of neighbors
                3. 'distances' the cosine distances.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")

        self.__log.info("Reading index file")
        with h5py.File(self.data_path, "r") as hw:
            metric_orig = hw["metric"][0]
            if type(hw["metric"][0]) != str:
                metric_orig = metric_orig.decode()
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
        data = np.array(signatures, dtype='float32')
        self.__log.info("Searching %s neighbors" % k)
        # get neighbors idx and distances
        if "cosine" in metric_orig:
            normst = LA.norm(data, axis=1)
            dists, idx = index.search( np.array(data / normst[:, None], dtype='float32'), k)
        else:
            dists, idx = index.search( np.array(data, dtype='float32'), k)

        predictions = dict()
        predictions["indices"] = idx
        if keys:
            with h5py.File(self.data_path, 'r') as hf:
                keys = hf['col_keys'][:]
            predictions["keys"] = keys[idx].astype(str)
        if distances:

            predictions["distances"] = dists
            if metric_orig == "cosine":

                predictions["distances"] = np.maximum(
                    0.0, 1.0 - predictions["distances"])

        return predictions

    def check_distances(self, n_sign=5, n_neig=10):
        sign = self.get_sign('sign' + self.cctype[-1])
        dist_fn = eval(self.metric)
        for ink1 in tqdm(sign.keys[:n_sign], desc='Checking distances'):
            s1 = sign[ink1]
            nn = self[ink1]
            inks = nn['keys'][:n_neig]
            dists = nn['distances'][:n_neig]
            for ink2, dist in zip(inks, dists):
                ink2 = ink2.decode()
                s2 = sign[ink2]
                comp_d = dist_fn(s1, s2)
                if not np.allclose(dist, comp_d, atol=1e-05):
                    self.__log.error('%s %s %.6f %.6f', ink1, ink2,  dist,
                                     comp_d)
                    np.testing.assert_allclose(dist, comp_d, atol=1e-05)

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

    @property
    def shape(self):
        """Get the V matrix sizes."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if 'shape' not in hf.keys():
                self.__log.warn("HDF5 file has no 'shape' dataset.")
                return hf['distances'].shape
            return hf['shape'][:]
