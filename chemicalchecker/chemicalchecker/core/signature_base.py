import six
from abc import ABCMeta, abstractmethod
from chemicalchecker.util import Config


@six.add_metaclass(ABCMeta)
class BaseSignature(object):
    """A Signature bla bla."""

    @abstractmethod
    def __init__(self, config_file=None):
        """From the recipe we derive all the cleaning logic."""
        self.config = Config(config_file)
        print 'BaseSignature'

    @abstractmethod
    def fit(self):
        """Takes an input and learns to produce an output."""

    @abstractmethod
    def predict(self):
        """Uses the fitted models to go from input to output."""

    @abstractmethod
    def validate(self):
        """Performs a validation across external data as MoA and ATC codes."""

    @abstractmethod
    def __iter__(self):
        """Batch iteration, if necessary."""

    @abstractmethod
    def __getattr__(self):
        """Return the vector corresponding to the key. 

        Works fast with bisect, but should return None if the key is not in
        keys (ideally, keep a set to do this).."""
