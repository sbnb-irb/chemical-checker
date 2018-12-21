"""Implementation of the abstract signature class.

Each signature class derived from this base class will have to implement the
`fit`, `predict` and `validate` methods. As the underlying data format for
every signature is the same, this class implements the iterator and attribute
getter.
"""
import os
import six
from abc import ABCMeta, abstractmethod

from chemicalchecker.util import logged


@logged
@six.add_metaclass(ABCMeta)
class BaseSignature(object):
    """A Signature base class.

    Implements methods and checks common to all signatures.
    """

    @abstractmethod
    def __init__(self, data_path, model_path, stats_path, dataset_info):
        """From the recipe we derive all the cleaning logic."""
        BaseSignature.__log.debug('__init__')
        self.dataset_info = dataset_info
        self.data_path = data_path
        if not os.path.isfile(data_path):
            BaseSignature.__log.warning(
                "Data file not available: %s" % data_path)
        self.model_path = model_path
        if not os.path.isdir(model_path):
            BaseSignature.__log.warning(
                "Model directory not available: %s" % model_path)
        self.stats_path = stats_path
        if not os.path.isdir(stats_path):
            BaseSignature.__log.warning(
                "Plots directory not available: %s" % stats_path)
        if dataset_info is None:
            raise Exception("There is no information on the specied dataset.")

    @abstractmethod
    def fit(self):
        """Take an input and learns to produce an output."""
        BaseSignature.__log.debug('fit')
        if os.path.isfile(self.model_path):
            BaseSignature.__log.warning("Model already available.")

    @abstractmethod
    def predict(self):
        """Use the fitted models to go from input to output."""
        BaseSignature.__log.debug('predict')
        if not os.path.isfile(self.model_path):
            raise Exception("Model file not available.")

    @abstractmethod
    def statistics(self):
        """Perform a validation across external data as MoA and ATC codes."""
        BaseSignature.__log.debug('statistics')
        if not os.path.isfile(self.model_path):
            raise Exception("Model file not available.")

    def __iter__(self):
        """Batch iteration, if necessary."""
        BaseSignature.__log.debug('__iter__')
        if not os.path.isfile(self.data_path):
            raise Exception("Data file not available.")
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
