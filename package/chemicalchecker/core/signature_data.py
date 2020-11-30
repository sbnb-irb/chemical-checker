"""Signature data.

This class handles the ``HDF5`` file internal organization. It offers shortcut
attributes to read from file (e.g. ``sign.keys`` or accessing signature with
brackets [:] operator). It expose utility methods to fetch, stack, copy
check for consistency, index, subsample or map signatures.
"""
import os
import sys
import h5py
import numpy as np
from bisect import bisect_left
from scipy.spatial.distance import euclidean, cosine

from chemicalchecker.util import logged
from chemicalchecker.util.decorator import cached_property


@logged
class DataSignature(object):
    """DataSignature class."""

    def __init__(self, data_path, ds_data='V', keys_name='keys'):
        """Initialize a DataSignature instance."""
        self.data_path = os.path.abspath(data_path)
        self.ds_data = ds_data
        self.keys_name = keys_name
        self.PVALRANGES = np.array(
            [0, 0.001, 0.01, 0.1] + list(np.arange(1, 100)) + [100]) / 100.

    def _check_data(self):
        """Test if data file is available"""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)

    def _check_dataset(self, key):
        """Test if dataset is available"""
        with h5py.File(self.data_path, 'r') as hf:
            if key not in hf.keys():
                raise Exception("No '%s' dataset in this signature!" % key)

    @staticmethod
    def _decode(arg):
        """Apply decode function to input"""
        decoder = np.vectorize(lambda x: x.decode())
        return decoder(arg)

    def _get_shape(self, key):
        """Get shape of dataset"""
        with h5py.File(self.data_path, 'r') as hf:
            data = hf[key].shape
            return data

    def _get_dtype(self, key):
        """Get shape of dataset"""
        with h5py.File(self.data_path, 'r') as hf:
            data = hf[key][0].flat[0].dtype
            return data

    def _get_all(self, key):
        """Get complete dataset"""
        with h5py.File(self.data_path, 'r') as hf:
            data = hf[key][:]
            if hasattr(data.flat[0], 'decode'):
                return self._decode(data)
            return data

    def _get_chunk(self, key, chunk):
        """Get chunk of dataset"""
        with h5py.File(self.data_path, 'r') as hf:
            data = hf[key][chunk]
            if hasattr(data.flat[0], 'decode'):
                return self._decode(data)
            return data

    def chunk_iter(self, key, chunk_size):
        """Iterator on chunks of data"""
        self._check_data()
        self._check_dataset(key)
        tot_size = self._get_shape(key)[0]
        for i in range(0, tot_size, chunk_size):
            chunk = slice(i, i + chunk_size)
            yield self._get_chunk(key, chunk)

    def __iter__(self):
        """By default iterate on signatures V."""
        self._check_data()
        self._check_dataset(self.ds_data)
        with h5py.File(self.data_path, 'r') as hf:
            for i in range(self.shape[0]):
                yield hf[self.ds_data][i]

    def chunker(self, size=2000, n=None):
        """Iterate on signatures."""
        self._check_data()
        if n is None:
            n = self.shape[0]
        for i in range(0, n, size):
            yield slice(i, i + size)

    def _refresh(self, key):
        """Delete a cached property"""
        try:
            if hasattr(self, key):
                delattr(self, key)
        except:
            self.__log.warn("No %s in this signature" % key)

    def refresh(self):
        """Refresh all cached properties"""
        self._refresh("name")
        self._refresh("date")
        self._refresh("data")
        self._refresh("data_type")
        self._refresh("keys")
        self._refresh("row_keys")
        self._refresh("keys_raw")
        self._refresh("unique_keys")
        self._refresh("features")
        self._refresh("mappings")

    @cached_property
    def name(self):
        """Get the name of the signature."""
        self._check_data()
        self._check_dataset('name')
        return self._get_all('name')

    @cached_property
    def date(self):
        """Get the date of the signature."""
        self._check_data()
        self._check_dataset('date')
        return self._get_all('date')

    @cached_property
    def data(self):
        """Get the data of the signature."""
        self._check_data()
        self._check_dataset(self.ds_data)
        return self._get_all(self.ds_data)

    @cached_property
    def data_type(self):
        """Get the data of the signature."""
        self._check_data()
        self._check_dataset(self.ds_data)
        return self._get_dtype(self.ds_data)

    @cached_property
    def keys(self):
        """Get the list of keys (usually inchikeys) in the signature."""
        self._check_data()
        self._check_dataset(self.keys_name)
        return self._get_all(self.keys_name)

    @cached_property
    def row_keys(self):
        """Get the list of keys (usually inchikeys) in the signature."""
        self._check_data()
        self._check_dataset('row_keys')
        return self._get_all('row_keys')

    @cached_property
    def keys_raw(self):
        """Get the list of keys in the signature."""
        self._check_data()
        self._check_dataset('keys_raw')
        return self._get_all('keys_raw')

    @cached_property
    def unique_keys(self):
        """Get the keys of the signature as a set."""
        return set(self.keys)

    @cached_property
    def features(self):
        """Get the list of features in the signature."""
        self._check_data()
        try:
            self._check_dataset('features')
            return self._get_all('features')
        except Exception:
            self.__log.warning("Features are not available")
            return np.array([i for i in range(0, self.shape[1])])

    @cached_property
    def mappings(self):
        """Get the list of features in the signature."""
        self._check_data()
        try:
            self._check_dataset('mappings')
            return self._get_all('mappings')
        except Exception:
            self.__log.warning('Mappings are not available,' +
                               ' using implicit key-key mappings.')
            return np.vstack([self.keys, self.keys]).T

    @property
    def info_h5(self):
        """Get the dictionary of dataset and shapes."""
        self._check_data()
        infos = dict()
        with h5py.File(self.data_path, 'r') as hf:
            for key in hf.keys():
                infos[key] = hf[key].shape
        return infos

    @property
    def shape(self):
        """Get the V matrix shape."""
        self._check_data()
        self._check_dataset(self.ds_data)
        with h5py.File(self.data_path, 'r') as hf:
            return hf[self.ds_data].shape

    @staticmethod
    def string_dtype():
        if sys.version_info[0] == 2:
            # this works in py2 and fails in py3
            return h5py.special_dtype(vlen=unicode)
        else:
            # because str is the new unicode in py3
            return h5py.special_dtype(vlen=str)

    @staticmethod
    def h5_str(lst):
        return np.array(lst, dtype=DataSignature.string_dtype())

    def copy_from(self, sign, key):
        """Copy dataset 'key' to current signature.

        Args:
            sign(SignatureBase): The source signature.
            key(str): The dataset to copy from.
        """
        sign._check_data()
        sign._check_dataset(key)
        with h5py.File(sign.data_path, 'r') as hf:
            src = hf[key][:]
        with h5py.File(self.data_path, 'a') as hf:
            # delete if already there
            if key in hf:
                del hf[key]
            hf[key] = src

    def make_filtered_copy(self, destination, mask, include_all=False,
                           data_file=None):
        """
        Make a copy of applying a filtering mask.

        destination (str): The destination file path.
        mask (bool array): A numpy mask array (e.g. result of `np.isin`)
        include_all (bool): Whether to copy other dataset (e.g. features,
            date, name...)
        data_file (str): A specific file to copy (by default is the signature
            h5)
        """

        if data_file is None:
            data_file = self.data_path

        with h5py.File(data_file, 'r') as hf_in:
            with h5py.File(destination, 'w') as hf_out:
                for dset in hf_in.keys():
                    # skip all dataset that cannot be masked
                    if hf_in[dset].shape[0] != mask.shape[0]:
                        if not include_all:
                            continue
                        else:
                            masked = hf_in[dset][:][:]
                    else:
                        if dset == 'features':
                            masked = hf_in[dset][:][:]
                        else:
                            masked = hf_in[dset][:][mask]

                    self.__log.debug("Copy dataset %s of shape %s" %
                                     (dset, str(masked.shape)))
                    hf_out.create_dataset(dset, data=masked)

    @staticmethod
    def hstack_signatures(sign_list, destination, chunk_size=1000,
                          aggregate_keys=None):
        """Merge horizontally a list of signatures."""
        hsizes = [s.shape[1] for s in sign_list]
        vsizes = [s.shape[0] for s in sign_list]
        if not all([vsizes[0] == v for v in vsizes]):
            raise ValueError('All signatures must have same molecules.')
        for idx in range(len(sign_list) - 1):
            if not all(sign_list[idx].keys == sign_list[idx + 1].keys):
                raise ValueError('All signatures must have same molecules.')

        with h5py.File(destination, "w") as results:
            results.create_dataset('keys', data=np.array(
                sign_list[0].keys, DataSignature.string_dtype()))
            results.create_dataset('V', (vsizes[0], sum(hsizes)))

            for idx, sign in enumerate(sign_list):
                with h5py.File(sign.data_path, 'r') as hf_in:
                    for i in range(0, vsizes[0], chunk_size):
                        vchunk = slice(i, i + chunk_size)
                        hchunk = slice(sum(hsizes[:idx]), sum(
                            hsizes[:idx]) + hsizes[idx])
                        results['V'][vchunk, hchunk] = hf_in['V'][vchunk]
            # also copy other single column numerical vectors
            if aggregate_keys:
                for key in aggregate_keys:
                    tmp = list()
                    for idx, sign in enumerate(sign_list):
                        with h5py.File(sign.data_path, 'r') as hf_in:
                            tmp.append(hf_in[key][:])
                    results.create_dataset(key, data=np.vstack(tmp).T)

    @staticmethod
    def vstack_signatures(sign_list, destination, chunk_size=1000):
        """Merge horizontally a list of signatures."""
        hsizes = [s.shape[1] for s in sign_list]
        vsizes = [s.shape[0] for s in sign_list]
        if not all([hsizes[0] == h for h in hsizes]):
            raise ValueError('All signatures must have same features.')

        with h5py.File(destination, "w") as results:
            results.create_dataset('keys', data=np.array(
                np.hstack([s.keys for s in sign_list]),
                DataSignature.string_dtype()))
            results.create_dataset("V", (sum(vsizes), hsizes[0]))

            for idx, sign in enumerate(sign_list):
                with h5py.File(sign.data_path, 'r') as hf_in:
                    for i in range(0, vsizes[idx], chunk_size):
                        if i + chunk_size > vsizes[idx]:
                            end = vsizes[idx]
                        else:
                            end = i + chunk_size
                        vchunk_src = slice(i, end)
                        vchunk_dst = slice(sum(vsizes[:idx]) + i,
                                           sum(vsizes[:idx]) + end)
                        results['V'][vchunk_dst] = hf_in['V'][vchunk_src]

    def get_h5_dataset(self, h5_dataset_name, mask=None):
        """Get a specific dataset in the signature."""
        self.__log.debug("Fetching dataset %s" % h5_dataset_name)
        self._check_data()
        with h5py.File(self.data_path, 'r') as hf:
            if h5_dataset_name not in hf.keys():
                raise Exception("HDF5 file has no '%s'." % h5_dataset_name)
            if mask is None:
                ndim = hf[h5_dataset_name].ndim
                if hasattr(hf[h5_dataset_name][(0,) * ndim], 'decode'):
                    decoder = np.vectorize(lambda x: x.decode())
                    return decoder(hf[h5_dataset_name][:])
                else:
                    return hf[h5_dataset_name][:]
            else:
                ndim = hf[h5_dataset_name].ndim
                if hasattr(hf[h5_dataset_name][(0,) * ndim], 'decode'):
                    decoder = np.vectorize(lambda x: x.decode())
                    return decoder(hf[h5_dataset_name][mask])
                else:
                    return hf[h5_dataset_name][mask, :]

    def get_vectors(self, keys, include_nan=False, dataset_name='V', output_missing=False):
        """Get vectors for a list of keys, sorted by default.

        Args:
            keys(list): a List of string, only the overlapping subset to the
                signature keys is considered.
            include_nan(bool): whether to include requested but absent
                molecule signatures as NaNs.
            dataset_name(str): return any dataset in the h5 which is organized
                by sorted keys.
        """
        self.__log.debug("Fetching %s rows from dataset %s" %
                         (len(keys), dataset_name))
        valid_keys = list(self.unique_keys & set(keys))
        idxs = np.argwhere(
            np.isin(list(self.keys), list(valid_keys), assume_unique=True))
        inks, signs = list(), list()

        with h5py.File(self.data_path, 'r') as hf:
            dset = hf[dataset_name]
            dset_shape = dset.shape
            for idx in sorted(idxs.flatten()):
                inks.append(self.keys[idx])
                signs.append(dset[idx])
        missed_inks = set(keys) - set(inks)
        # if missing signatures are requested add NaNs
        if include_nan:
            inks.extend(list(missed_inks))
            dimensions = (len(missed_inks), dset_shape[1])
            nan_matrix = np.zeros(dimensions) * np.nan
            signs.append(nan_matrix)
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
        if output_missing:
            return inks[sort_idx], signs[sort_idx], missed_inks
        return inks[sort_idx], signs[sort_idx]

    def get_vectors_lite(self, keys, chunk_size=2000, chunk_above=10000):
        """Iterate on signatures."""
        from tqdm import tqdm
        keys = set(keys)
        idxs = []
        mask = []
        kept_keys = []
        for i, key in enumerate(self.keys):
            if key not in keys:
                mask += [False]
                continue
            else:
                idxs += [i]
                kept_keys += [key]
                mask += [True]
        idxs = np.array(idxs)
        kept_keys = np.array(kept_keys)
        if len(idxs) > chunk_above:
            with h5py.File(self.data_path, "r") as hf:
                dset = hf["V"]
                V = None
                for chunk in tqdm(self.chunker(size=chunk_size)):
                    mask_ = mask[chunk]
                    if not np.any(mask_):
                        continue
                    v = dset[chunk][mask_]
                    if V is None:
                        V = v
                    else:
                        V = np.vstack([V, v])
        else:
            with h5py.File(self.data_path, "r") as hf:
                dset = hf["V"]
                v = dset[0]
                V = np.zeros((len(idxs), self.shape[1]), dtype=v.dtype)
                for i, idx in tqdm(enumerate(idxs)):
                    V[i, :] = dset[idx]
        if len(kept_keys) != len(keys):
            self.__log.warn("There are %d missing keys" %
                            (len(keys) - len(kept_keys)))
        return kept_keys, V

    def index(self, key):
        """Give the index according to the key.

        Args:
            key(str): the key to search index in the matrix.
        Returns:
            index(int): Index in the matrix
        """
        if key not in self.unique_keys:
            raise Exception("Key '%s' not found." % key)
        idx = bisect_left(self.keys, key)
        return idx

    def __getitem__(self, key):
        """Return the vector corresponding to the key.

        The key can be a string (then it's mapped though self.keys) or and
        int.
        Works fast with bisect, but should return None if the key is not in
        keys (ideally, keep a set to do this)."""
        self._check_data()
        if isinstance(key, slice):
            with h5py.File(self.data_path, 'r') as hf:
                return hf[self.ds_data][key]
        elif isinstance(key, str):
            if key not in self.unique_keys:
                raise Exception("Key '%s' not found." % key)
            idx = bisect_left(self.keys, key)
            with h5py.File(self.data_path, 'r') as hf:
                return hf[self.ds_data][idx]
        elif isinstance(key, int):
            with h5py.File(self.data_path, 'r') as hf:
                return hf[self.ds_data][key]
        else:
            raise Exception("Key type %s not recognized." % type(key))

    def compute_distance_pvalues(self, bg_file, metric, sample_pairs=None,
                                 unflat=True, memory_safe=False):
        """Compute the distance pvalues according to the selected metric.

        Args:
            bg_file(Str): The file where to store the distances.
            metric(str): the metric name (cosine or euclidean).
            sample_pairs(int): Amount of pairs for distance calculation.
            unflat(bool): Remove flat regions whenever we observe them.
            memory_safe(bool): Computing distances is much faster if we can
                load the full matrix in memory.
        Returns:
            bg_distances(dict): Dictionary with distances and Pvalues
        """
        # lazily read already computed distance
        if os.path.isfile(bg_file):
            self.__log.info("Reading bg_distances file for metric: " + metric)
            bg_distances = dict()
            with h5py.File(bg_file, 'r') as f5:
                bg_distances["distance"] = f5["distance"][:]
                bg_distances["pvalue"] = f5["pvalue"][:]
            return bg_distances
        # otherwise compute and save them
        self.__log.info("Background distances not available, computing them.")
        # set metric function
        if metric not in ['cosine', 'euclidean']:
            raise Exception("Specified metric %s not available." % metric)
        metric_fn = eval(metric)
        # sample distances
        if memory_safe:
            matrix = self
        else:
            matrix = self[:]
        # how many molecules gives a proper sampling?
        if sample_pairs is None:
            # p and q are fixed parameters
            p, q = 0.5, 0.5
            # 5% confidence, 95% precision
            d, Z = 0.05, 1.96
            coef = Z**2 * p * q
            k = (coef * len(self.keys)) / (d**2 * (len(self.keys) - 1) + coef)
            sample_pairs = int(np.ceil(k**2))
        self.__log.info("Background distances sample_pairs: %s" % sample_pairs)
        if matrix.shape[0]**2 < sample_pairs:
            self.__log.warn("Requested more pairs then possible combinations")
            sample_pairs = matrix.shape[0]**2 - matrix.shape[0]
        bg = list()
        done = set()
        tries = 1e6
        tr = 0
        while len(bg) < sample_pairs and tr < tries:
            tr += 1
            i = np.random.randint(0, matrix.shape[0] - 1)
            j = np.random.randint(i + 1, matrix.shape[0])
            if (i, j) not in done:
                dist = metric_fn(matrix[i], matrix[j])
                if dist == 0.0:
                    self.__log.warn("Identical signatures for %s %s" % (i, j))
                if np.isnan(dist):
                    self.__log.warn("NaN distance for %s %s" % (i, j))
                    continue
                bg.append(dist)
                done.add((i, j))
        # pavalues as percentiles
        i = 0
        PVALS = [(0, 0., i)]  # DISTANCE, RANK, INTEGER
        i += 1
        percs = self.PVALRANGES[1:-1] * 100
        for perc in percs:
            PVALS += [(np.percentile(bg, perc), perc / 100., i)]
            i += 1
        PVALS += [(np.max(bg), 1., i)]
        # prepare returned dictionary
        bg_distances = dict()
        if not unflat:
            bg_distances["distance"] = np.array([p[0] for p in PVALS])
            bg_distances["pvalue"] = np.array([p[1] for p in PVALS])
        else:
            # Remove flat regions whenever we observe them
            dists = [p[0] for p in PVALS]
            pvals = np.array([p[1] for p in PVALS])
            top_pval = np.min(
                [1. / sample_pairs, np.min(pvals[pvals > 0]) / 10.])
            pvals[pvals == 0] = top_pval
            pvals = np.log10(pvals)
            dists_ = sorted(set(dists))
            pvals_ = [pvals[dists.index(d)] for d in dists_]
            dists = np.interp(pvals, pvals_, dists_)
            thrs = [(dists[t], PVALS[t][1], PVALS[t][2])
                    for t in range(len(PVALS))]
            bg_distances["distance"] = np.array([p[0] for p in thrs])
            bg_distances["pvalue"] = np.array([p[1] for p in thrs])
        # save to file
        with h5py.File(bg_file, "w") as hf:
            hf.create_dataset("distance", data=bg_distances["distance"])
            hf.create_dataset("pvalue", data=bg_distances["pvalue"])
        return bg_distances

    def subsample(self, n):
        """Subsample from a signature without replacement.

            Args:
               n(int): Maximum number of samples (default=10000).

            Returns:
               V(matrix): A (samples, features) matrix.
               keys(array): The list of keys.
        """
        self.__log.debug("Subsampling dataset (n=%d)" % n)
        if n > len(self.keys):
            V = self[:]
            keys = self.keys
        else:
            idxs = np.array(sorted(np.random.choice(
                len(self.keys), n, replace=False)))
            with h5py.File(self.data_path, "r") as hf:
                V = hf["V"][idxs]
            keys = np.array(self.keys)[idxs]
        return V, keys

    def consistency_check(self):
        """Check that signature is valid."""
        if os.path.isfile(self.data_path):
            # check that keys are unique
            if len(self.keys) != len(self.unique_keys):
                raise Exception("Inconsistent: keys are not unique.")
            # check that amout of keys is same as amount of signatures
            with h5py.File(self.data_path, 'r') as hf:
                nr_signatures = hf['V'].shape[0]
            if len(self.keys) > nr_signatures:
                raise Exception("Inconsistent: more Keys than signatures.")
            if len(self.keys) < nr_signatures:
                raise Exception("Inconsistent: more signatures than Keys.")
            # check that keys are sorted
            if not np.all(self.keys[:-1] <= self.keys[1:]):
                raise Exception("Inconsistent: Keys are not sorted.")

    def map(self, out_file):
        """Map signature throught mappings."""
        if "mappings" not in self.info_h5:
            raise Exception("Data file has no mappings.")
        with h5py.File(self.data_path, 'r') as hf:
            mappings_raw = hf['mappings'].asstr()[:]
        mappings = dict(mappings_raw)
        # avoid trivial mappings (where key==value)
        to_map = list(set(mappings.keys()) - set(mappings.values()))
        if len(to_map) == 0:
            # corner case where there's nothing to map
            with h5py.File(self.data_path, 'r') as hf:
                src_keys = hf['keys'].asstr()[:]
                src_vectors = hf['V'][:]
            with h5py.File(out_file, "w") as hf:
                hf.create_dataset('keys', data=np.array(
                    src_keys, DataSignature.string_dtype()))
                hf.create_dataset('V', data=src_vectors, dtype=np.float32)
                hf.create_dataset("shape", data=src_vectors.shape)
            return
        # prepare key-vector arrays
        dst_keys = list()
        dst_vectors = list()
        for dst_key in sorted(to_map):
            dst_keys.append(dst_key)
            dst_vectors.append(self[mappings[dst_key]])
        # to numpy arrays
        dst_keys = np.array(dst_keys)
        matrix = np.vstack(dst_vectors)
        # join with current key-signatures
        with h5py.File(self.data_path, 'r') as hf:
            src_vectors = hf['V'][:]
        dst_keys = np.concatenate((dst_keys, self.keys))
        matrix = np.concatenate((matrix, src_vectors))
        # get them sorted
        sorted_idx = np.argsort(dst_keys)
        with h5py.File(out_file, "w") as hf:
            hf.create_dataset('keys', data=np.array(
                dst_keys[sorted_idx], DataSignature.string_dtype()))
            hf.create_dataset('V', data=matrix[sorted_idx], dtype=np.float32)
            hf.create_dataset("shape", data=matrix.shape)

    def generator_fn(self, batch_size=None):
        """Return the generator function that we can query for batches."""
        hf = h5py.File(self.data_path, 'r')
        dset = hf['V']
        total = dset.shape[0]
        if not batch_size:
            batch_size = total

        def _generator_fn():
            beg_idx, end_idx = 0, batch_size
            while True:
                if beg_idx >= total:
                    self.__log.debug("EPOCH completed")
                    beg_idx = 0
                    return
                yield dset[beg_idx: end_idx]
                beg_idx, end_idx = beg_idx + batch_size, end_idx + batch_size

        return _generator_fn
