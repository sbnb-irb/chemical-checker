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
from abc import ABCMeta, abstractmethod

from chemicalchecker.util import logged


@logged
@six.add_metaclass(ABCMeta)
class BaseSignature(object):
    """A Signature base class.

    Implements methods and checks common to all signatures.
    """

    @abstractmethod
    def __init__(self, signature_path, dataset_info, **params):
        """Initialize or load the signature at the given path."""
        self.dataset_info = dataset_info
        self.signature_path = signature_path
        self.param_file = os.path.join(signature_path, 'PARAMS.JSON')

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

    @abstractmethod
    def statistics(self):
        """Perform a validation across external data as MoA and ATC codes."""
        BaseSignature.__log.debug('statistics')
        if not os.path.isdir(self.model_path):
            raise Exception("Model file not available.")

    @property
    def shape(self):
        """Get the signature matrix shape (i.e. the sizes)."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if 'shape' not in hf.keys():
                raise Exception("HDF5 file has no 'shape' field.")
            return hf['shape'][:]

    def keys(self):
        """Get the signature matrix shape (i.e. the sizes)."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if 'keys' not in hf.keys():
                raise Exception("HDF5 file has no 'keys' field.")
            return hf['keys'][:]

    def __iter__(self):
        """Batch iteration, if necessary."""
        BaseSignature.__log.debug('__iter__')
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        BaseSignature.__log.debug('parsing data %s', self.data_path)
        yield

    def __getattr__(self):
        """Return the vector corresponding to the key.

        Works fast with bisect, but should return None if the key is not in
        keys (ideally, keep a set to do this).."""
        BaseSignature.__log.debug('__getattr__')
        if not os.path.isfile(self.data_path):
            raise Exception("Data file not available.")
        yield

    def __repr__(self):
        """String representig the signature."""
        return self.data_path
