import os
import sys
import h5py
from bisect import bisect_left
import numpy as np
import random
from chemicalchecker.util import logged
from scipy.spatial.distance import euclidean, cosine


class cached_property(object):
    """Decorator for properties calculated/stored on-demand on first use."""

    def __init__(self, func):
        self._attr_name = func.__name__
        self._func = func

    def __get__(self, instance, owner):
        attr = self._func(instance)
        setattr(instance, self._attr_name, attr)
        return attr


@logged
class DataSignature(object):
    """A Signature data class.

    Implements methods and checks common to all signatures for accessing the
    data in HDF5 format.
    """

    def __init__(self, data_path, ds_data='V', keys_name='keys'):
        """Initialize or load the signature at the given path."""
        self.data_path = os.path.abspath(data_path)
        self.ds_data = ds_data
        self.keys_name = keys_name
        self.PVALRANGES = np.array([0, 0.001, 0.01, 0.1] +
                                   list(np.arange(1, 100)) + [100]) / 100.

    @cached_property
    def keys(self):
        """Get the list of keys (usually inchikeys) in the signature."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if self.keys_name not in hf.keys():
                raise Exception("HDF5 file has no 'keys' field.")
            # if keys have a decode attriute they have been generated in py2
            # for compatibility with new format we decode them
            if hasattr(hf[self.keys_name][0], 'decode'):
                return [k.decode() for k in hf[self.keys_name][:]]
            else:
                return hf[self.keys_name][:]

    @cached_property
    def unique_keys(self):
        """Get the keys of the signature as a set."""
        return set(self.keys)

    @property
    def info_h5(self):
        """Get the dictionary of dataset and shapes."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        infos = dict()
        with h5py.File(self.data_path, 'r') as hf:
            for key in hf.keys():
                infos[key] = hf[key].shape
        return infos

    @staticmethod
    def string_dtype():
        if sys.version_info[0] == 2:
            # this works in py2 and fails in py3
            return h5py.special_dtype(vlen=unicode)
        else:
            # because str is the new unicode in py3
            return h5py.special_dtype(vlen=str)

    def copy_from(self, sign, key):
        """Copy dataset 'key' to current signature.

        Args:
            sign(SignatureBase): The source signature.
            key(str): The dataset to copy from.
        """
        if key not in sign.info_h5:
            raise Exception("Data file %s has no dataset named '%s'." %
                            (sign.data_path, key))
        with h5py.File(sign.data_path, 'r') as hf:
            src = hf[key][:]
        with h5py.File(self.data_path, 'a') as hf:
            # delete if already there
            if key in hf:
                del hf[key]
            hf[key] = src

    def get_h5_dataset(self, h5_dataset_name, mask=None):
        """Get a specific dataset in the signature."""
        self.__log.debug("Fetching dataset %s" % h5_dataset_name)
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if h5_dataset_name not in hf.keys():
                raise Exception("HDF5 file has no '%s'." % h5_dataset_name)
            if mask is None:
                ndim = hf[h5_dataset_name].ndim
                if hasattr(hf[h5_dataset_name][(0,) * ndim], 'encode'):
                    encoder = np.vectorize(lambda x: x.encode())
                    return encoder(hf[h5_dataset_name][:])
                else:
                    return hf[h5_dataset_name][:]
            else:
                ndim = hf[h5_dataset_name].ndim
                if hasattr(hf[h5_dataset_name][(0,) * ndim], 'encode'):
                    encoder = np.vectorize(lambda x: x.encode())
                    return encoder(hf[h5_dataset_name][mask])
                else:
                    return hf[h5_dataset_name][mask, :]

    def get_vectors(self, keys, include_nan=False, dataset_name='V'):
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
            np.isin(self.keys, list(valid_keys), assume_unique=True))
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
        return inks[sort_idx], signs[sort_idx]

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
        if not os.path.isfile(self.data_path):
            raise Exception("Data file not available.")
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

    def __iter__(self):
        """Iterate on signatures."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            for i in range(self.shape[0]):
                yield hf['V'][i]

    def background_distances(self, metric, sample_pairs=100000, unflat=True,
                             memory_safe=False):
        """Give the background distances according to the selected metric.

        Args:
            metric(str): the metric name (cosine or euclidean).
            sample_pairs(int): Amount of pairs for distance calculation.
            unflat(bool): Remove flat regions whenever we observe them.
            memory_safe(bool): Computing distances is much faster if we can
                load the full matrix in memory.
        Returns:
            bg_distances(dict): Dictionary with distances and Pvalues
        """
        # lazily read already computed distance
        bg_file = os.path.join(self.model_path, "bg_%s_distances.h5" % metric)
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
        if matrix.shape[0]**2 < sample_pairs:
            self.__log.warn("Requested more pairs then possible combinations")
            sample_pairs = matrix.shape[0]**2 - matrix.shape[0]

        bg = list()
        done = set()
        while len(bg) < sample_pairs:
            i = np.random.randint(0, matrix.shape[0] - 1)
            j = np.random.randint(i + 1, matrix.shape[0])
            if (i, j) not in done:
                dist = metric_fn(matrix[i], matrix[j])
                if dist == 0.0:
                    self.__log.warn("Identical signatures for %s %s" % (i, j))
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
