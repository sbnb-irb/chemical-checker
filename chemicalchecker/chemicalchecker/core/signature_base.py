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
    def __init__(self, path):
        """From the recipe we derive all the cleaning logic."""
        BaseSignature.__log.debug('__init__')
        if not os.path.isfile(path):
            BaseSignature.__log.warning("HDF5 file not available")

    @abstractmethod
    def fit(self):
        """Takes an input and learns to produce an output."""
        BaseSignature.__log.debug('fit')

    @abstractmethod
    def predict(self):
        """Uses the fitted models to go from input to output."""
        BaseSignature.__log.debug('predict')

    @abstractmethod
    def validate(self):
        """Performs a validation across external data as MoA and ATC codes."""
        BaseSignature.__log.debug('validate')

    def __iter__(self):
        """Batch iteration, if necessary."""
        BaseSignature.__log.debug('__iter__')
        yield

    def __getattr__(self):
        """Return the vector corresponding to the key. 

        Works fast with bisect, but should return None if the key is not in
        keys (ideally, keep a set to do this).."""
        BaseSignature.__log.debug('__getattr__')
        yield
