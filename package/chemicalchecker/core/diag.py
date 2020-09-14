"""
Class to run diagnostic tests after a sign is fit in the CC_pipeline
"""
import os
import numpy as np



from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged


@logged
class diag(BaseSignature):
    """Projection Signature class."""

    def __init__(self, signature_path, dataset,**kwargs):
        """Initialize the proj class.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related
            kwargs(dict): the key is a projector name with the value being
                a dictionary of its parameters.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, signature_path, dataset, **kwargs)
        self.__log.debug('signature path is: %s', signature_path)

        #DataSignature.__init__(self, self.data_path)
        #self.__log.debug('data_path: %s', self.data_path)



    def fit(self):
        pass

    def predict(self):
        pass

    def is_fit(self):
        if len(os.listdir(self.diags_path)) == 0:
            return False
        else:
            print(self.diags_path, "contains some diagnostic plots, nothing to do here")
            return True
