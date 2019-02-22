"""Implementation of the abstract signature class.

Each signature class derived from this base class will have to implement the
`fit`, `predict` and `validate` methods. As the underlying data format for
every signature is the same, this class implements the iterator and attribute
getter.
Also implements the signature status, and persistence of parameters.
"""
import os
import six
import json
import h5py
import numpy as np
from datetime import datetime
from bisect import bisect_left
from abc import ABCMeta, abstractmethod

from chemicalchecker.util import RNDuplicates
from chemicalchecker.util import logged


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
@six.add_metaclass(ABCMeta)
class BaseSignature(object):
    """A Signature base class.

    Implements methods and checks common to all signatures.
    """

    @abstractmethod
    def __init__(self, signature_path, validation_path, dataset, **params):
        """Initialize or load the signature at the given path."""
        self.dataset = dataset
        self.signature_path = signature_path
        self.param_file = os.path.join(signature_path, 'PARAMS.JSON')
        self.validation_path = validation_path

        if not os.path.isdir(signature_path):
            BaseSignature.__log.info(
                "Initializing new signature in: %s" % signature_path)
            original_umask = os.umask(0)
            os.makedirs(signature_path, 0o775)
            os.umask(original_umask)
            if not params:
                params = dict()
            with open(self.param_file, 'w') as fh:
                json.dump(params, fh)
        else:
            if not os.path.isfile(self.param_file):
                BaseSignature.__log.warning(
                    "Signature missing parameter file: %s" % self.param_file)
                BaseSignature.__log.warning(
                    "Updating with current: %s" % self.param_file)
                if not params:
                    params = dict()
                with open(self.param_file, 'w') as fh:
                    json.dump(params, fh)
        self.model_path = os.path.join(signature_path, "models")
        if not os.path.isdir(self.model_path):
            BaseSignature.__log.info(
                "Creating model_path in: %s" % self.model_path)
            original_umask = os.umask(0)
            os.makedirs(self.model_path, 0o775)
            os.umask(original_umask)
        self.stats_path = os.path.join(signature_path, "stats")
        if not os.path.isdir(self.stats_path):
            BaseSignature.__log.info(
                "Creating stats_path in: %s" % self.stats_path)
            original_umask = os.umask(0)
            os.makedirs(self.stats_path, 0o775)
            os.umask(original_umask)

    @abstractmethod
    def fit(self):
        """Take an input and learns to produce an output."""
        BaseSignature.__log.debug('fit')
        if os.path.isdir(self.model_path):
            BaseSignature.__log.warning("Model already available.")

    @abstractmethod
    def predict(self):
        """Use the fitted models to go from input to output."""
        BaseSignature.__log.debug('predict')
        if not os.path.isdir(self.model_path):
            raise Exception("Model file not available.")

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
            if not sorted(self.keys) == list(self.keys):
                raise Exception("Inconsistent: Keys are not sorted.")

    def map(self, out_file):
        """Map signature throught mappings."""
        if "mappings" not in self.info_h5:
            raise Exception("Data file has no mappings.")
        with h5py.File(self.data_path, 'r') as hf:
            mappings = dict(hf['mappings'][:])
        # avoid trivial mappings (where key==value)
        to_map = set(mappings.keys()) - set(mappings.values())
        if len(to_map) == 0:
            # corner case where there's nothing to map
            with h5py.File(self.data_path, 'r') as hf:
                src_keys = hf['keys'][:]
                src_vectors = hf['V'][:]
            with h5py.File(out_file, "w") as hf:
                hf.create_dataset('keys', data=src_keys)
                hf.create_dataset('V', data=src_vectors, dtype=np.float32)
                hf.create_dataset("shape", data=src_vectors.shape)
                hf.create_dataset(
                    "date", data=[datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
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
            hf.create_dataset('keys', data=dst_keys[sorted_idx])
            hf.create_dataset('V', data=matrix[sorted_idx], dtype=np.float32)
            hf.create_dataset("shape", data=matrix.shape)
            hf.create_dataset(
                "date", data=[datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    @property
    def shape(self):
        """Get the V matrix sizes."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if 'shape' not in hf.keys():
                raise Exception("HDF5 file has no 'shape' field.")
            return hf['shape'][:]

    @cached_property
    def keys(self):
        """Get the list of keys in the signature."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if 'keys' not in hf.keys():
                raise Exception("HDF5 file has no 'keys' field.")
            return hf['keys'][:]

    @cached_property
    def unique_keys(self):
        """Get the keys of the signature as a set."""
        return set(self.keys)

    def get_vectors(self, keys):
        """Get vectors for a list of keys.

        Args:
            keys(list): a List of string, only the overlapping subset to the
                signature keys is considered.
        """
        str_keys = set(k for k in keys if isinstance(k, str))
        valid_keys = list(self.unique_keys & str_keys)
        idxs = np.argwhere(
            np.isin(self.keys, list(valid_keys), assume_unique=True))
        inks, signs = list(), list()
        with h5py.File(self.data_path, 'r') as hf:
            dset = hf['V']
            for idx in sorted(idxs.flatten()):
                inks.append(self.keys[idx])
                signs.append(dset[idx])
        missed_inks = set(keys) - set(inks)
        if missed_inks:
            self.__log.warn("Following requested keys are not available:")
            for k in missed_inks:
                self.__log.warn("%s", k)
        return np.stack(inks), np.stack(signs)

    def __iter__(self):
        """Batch iteration, if necessary."""
        BaseSignature.__log.debug('__iter__')
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        BaseSignature.__log.debug('parsing data %s', self.data_path)
        yield

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

    def __repr__(self):
        """String representig the signature."""
        return self.data_path

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

    def get_non_redundant_intersection(self, sign):
        """Return the non redundant intersection between two signatures.

        (i.e. keys and vectors that are common to both signatures.)
        N.B: to maximize overlap it's better to use signatures of type 'full'.
        N.B: Near duplicates are found in the first signature.
        """
        shared_keys = self.unique_keys.intersection(sign.unique_keys)
        self.__log.debug("%s shared keys.", len(shared_keys))
        _, self_matrix = self.get_vectors(shared_keys)
        rnd = RNDuplicates()
        nr_keys, nr_matrix, mappings = rnd.remove(
            self_matrix, keys=list(shared_keys))
        a, self_matrix = self.get_vectors(nr_keys)
        b, sign_matrix = sign.get_vectors(nr_keys)
        assert(all(a == b))
        return a, self_matrix, sign_matrix
