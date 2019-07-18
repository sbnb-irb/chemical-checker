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

    def __init__(self, data_path):
        """Initialize or load the signature at the given path."""
        self.data_path = os.path.abspath(data_path)
        self.PVALRANGES = np.array([0, 0.001, 0.01, 0.1] +
                                   list(np.arange(1, 100)) + [100]) / 100.

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
                return hf['V'][key]
        elif isinstance(key, str):
            if key not in self.unique_keys:
                raise Exception("Key '%s' not found." % key)
            idx = bisect_left(self.keys, key)
            with h5py.File(self.data_path, 'r') as hf:
                return hf['V'][idx]
        elif isinstance(key, int):
            with h5py.File(self.data_path, 'r') as hf:
                return hf['V'][key]
        else:
            raise Exception("Key type %s not recognized." % type(key))

    def background_distances(self, metric, inchikey_vec=None, inchikeys=None, B=100000, unflat=True):
        """Give the background distances according to the selected metric.

        Args:
            metric(str): the metric name (cosine or euclidean).
            inchikey_vec(): the vectors to calculate the background distances.
        Returns:
            bg_distances(dict): Dictionary with distances and Pvalues
        """

        bg_distances = {}
        if inchikey_vec is None:

            self.__log.info("Reading bg_distances file for metric: " + metric)
            if metric == "cosine":
                bg_file = os.path.join(
                    self.model_path, "bg_cosine_distances.h5")
                if not os.path.isfile(bg_file):
                    raise Exception(
                        "The background distances for metric " + metric + " are not available.")
                with h5py.File(bg_file, 'r') as f5:
                    bg_distances["distance"] = f5["distance"][:]
                    bg_distances["pvalue"] = f5["pvalue"][:]

            if metric == "euclidean":
                bg_file = os.path.join(
                    self.model_path, "bg_euclidean_distances.h5")
                if not os.path.isfile(bg_file):
                    raise Exception(
                        "The background distances for metric " + metric + " are not available.")
                with h5py.File(bg_file, 'r') as f5:
                    bg_distances["distance"] = f5["distance"][:]
                    bg_distances["pvalue"] = f5["pvalue"][:]

            if len(bg_distances) == 0:
                raise Exception(
                    "The background distances for metric " + metric + " are not available.")

        else:

            if metric == "cosine":
                metric_fn = cosine

            if metric == "euclidean":
                metric_fn = euclidean

            # Check if it is a numpy array

            if type(inchikey_vec).__module__ == np.__name__:
                idxs = [i for i in xrange(inchikey_vec.shape[0])]
                bg = []
                for _ in xrange(B):
                    i, j = random.sample(idxs, 2)
                    bg += [metric_fn(inchikey_vec[i, :], inchikey_vec[j, :])]

            else:

                if inchikeys is None:
                    inchikeys = np.array(
                        [k for k, v in inchikey_vec.iteritems()])

                bg = []
                for _ in xrange(B):
                    ik1, ik2 = random.sample(inchikeys, 2)
                    bg += [metric_fn(inchikey_vec[ik1], inchikey_vec[ik2])]

            i = 0
            PVALS = [(0, 0., i)]  # DISTANCE, RANK, INTEGER
            i += 1
            percs = self.PVALRANGES[1:-1] * 100
            for perc in percs:
                PVALS += [(np.percentile(bg, perc), perc / 100., i)]
                i += 1
            PVALS += [(np.max(bg), 1., i)]

            if not unflat:
                bg_distances["distance"] = np.array([p[0] for p in PVALS])
                bg_distances["pvalue"] = np.array([p[1] for p in PVALS])
            else:
                # Remove flat regions whenever we observe them
                dists = [p[0] for p in PVALS]
                pvals = np.array([p[1] for p in PVALS])
                top_pval = np.min([1. / B, np.min(pvals[pvals > 0]) / 10.])
                pvals[pvals == 0] = top_pval
                pvals = np.log10(pvals)
                dists_ = sorted(set(dists))
                pvals_ = [pvals[dists.index(d)] for d in dists_]
                dists = np.interp(pvals, pvals_, dists_)
                thrs = [(dists[t], PVALS[t][1], PVALS[t][2])
                        for t in xrange(len(PVALS))]
                bg_distances["distance"] = np.array([p[0] for p in thrs])
                bg_distances["pvalue"] = np.array([p[1] for p in thrs])

        return bg_distances
