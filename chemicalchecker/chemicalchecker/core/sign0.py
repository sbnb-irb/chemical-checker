from chemicalchecker.util import logged
from .signature_base import BaseSignature


@logged
class sign0(BaseSignature):
    """A Signature bla bla."""

    def __init__(self, path):
        """From the recipe we derive all the cleaning logic."""
        BaseSignature.__init__(self, path)
        self.__log.debug('sign0 path: %s', path)
        self.path = path

    def fit(self):
        """Takes an input and learns to produce an output."""
        self.__log.debug('fit')

    def predict(self):
        """Uses the fitted models to go from input to output."""
        self.__log.debug('predict')

    def validate(self):
        """Performs a validation across external data as MoA and ATC codes."""
        self.__log.debug('validate')
