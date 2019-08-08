import os

from .signature_base import BaseSignature
from .signature_data import DataSignature

from .projector import Default
from .projector import PCA

from chemicalchecker.util import logged


@logged
class proj(BaseSignature, DataSignature):
    """A Signature bla bla."""

    def __init__(self, signature_path, dataset, proj_type='Default', **kwargs):
        """Initialize the projection class.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related
            kwargs(dict): the key is a projector name with the value being
                a dictionary of its parameters.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, signature_path, dataset, **kwargs)
        self.__log.debug('signature path is: %s', signature_path)

        # define which projector is needed
        self.data_path = os.path.join(signature_path, "proj_%s.h5" % proj_type)
        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)

        self.projector = eval(proj_type)(signature_path, dataset, **kwargs)

    def fit(self, signature, validations=True, *args, **kwargs):
        """Take an input and call corresponding projector."""
        self.projector.fit(signature, validations, *args, **kwargs)

    def predict(self, signature, destination, *args, **kwargs):
        """Load projector and predict projection for new data."""
        self.projector.predict(signature, destination, *args, **kwargs)
