"""
Class to run diagnostic tests after a sign is fit in the CC_pipeline
"""
import os
import numpy as np



from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged


@logged
class diag(BaseSignature, DataSignature):
    """Projection Signature class."""

    def __init__(self, signature_path, dataset, cc_instance,**kwargs):
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

        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)

        self.cc_instance=cc_instance


    def fit(self):
        signObj= self.cc_instance.get_signature(self.cc_type,self.molset,dataset)
        diag= self.cc_instance.diagnosis(signObj)
        diag.canvas()

    def predict(self)
        pass

    def is_fit(self):
        if len(os.listdir('/home/varun/temp') ) == 0: