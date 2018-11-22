from chemicalchecker.util import logged
from .signature_base import BaseSignature


@logged
class SignatureZero(BaseSignature):
    """A Signature bla bla."""

    def __init__(self, config_file=None):
        """From the recipe we derive all the cleaning logic."""
        BaseSignature.__init__(self, config_file)
        self.__log.debug('SignatureZero')
        self.__log.debug(self.config.PATH.CC_ROOT)

    def fit(self):
        """Takes an input and learns to produce an output."""

    def predict(self):
        """Uses the fitted models to go from input to output."""

    def validate(self):
        """Performs a validation across external data as MoA and ATC codes."""

    def __iter__(self):
        """Batch iteration, if necessary."""

    def __getattr__(self):
        """Return the vector corresponding to the key.

        Works fast with bisect, but should return None if the key is not in
        keys (ideally, keep a set to do this).."""
