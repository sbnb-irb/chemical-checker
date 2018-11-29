import os
from chemicalchecker.util import logged
from .signature_base import BaseSignature


@logged
class neig1(BaseSignature):
    """A Signature bla bla."""

    def __init__(self, data_path, model_path, dataset_info):
        """From the recipe we derive all the cleaning logic."""
        self.__log.debug('data_path: %s', data_path)
        self.data_path = data_path
        self.__log.debug('model_path: %s', model_path)
        self.model_path = model_path
        # edit model_path to the specific model
        BaseSignature.__init__(self, data_path, model_path, dataset_info)

    def fit(self, features):
        """Take an input and learns to produce an output."""
        BaseSignature.fit(self)
        self.__log.debug('Faiss kNN fit %s' % features)
        self.__log.debug('saving model to %s' % self.model_path)
        os.touch(self.model_path)

    def predict(self, features):
        """Use the fitted models to go from input to output."""
        BaseSignature.predict(self)
        self.__log.debug('loading model from %s' % self.model_path)
        self.__log.debug('Faiss kNN predict %s' % features)
        return 'neig1_pred'

    def validate(self, validation_set):
        """Perform a validation across external data as MoA and ATC codes."""
        BaseSignature.validate(self)
        self.__log.debug('Faiss kNN validate on %s' % validation_set)
