from chemicalchecker.util import logged
from .signature_base import BaseSignature


@logged
class sign0(BaseSignature):
    """Signature type 0 class.

    Signature type 0 is...
    """

    def __init__(self, data_path, model_path):
        """Initialize the signature.

        Args:
            data_path(str): Where the h5 file is.
            model_path(str): Where the persistent model is.
        """
        self.__log.debug('data_path: %s', data_path)
        self.data_path = data_path
        self.__log.debug('model_path: %s', model_path)
        self.model_path = model_path
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, data_path, model_path)

    def fit(self):
        """Signature type 0 has no models to fit."""
        self.__log.debug('nothing to fit.')

    def predict(self, preprocess_script):
        """Call the external preprocess script to generate h5 data."""
        self.__log.debug('calling pre-process script', preprocess_script)
        self.__log.debug('faking data in %s', self.data_path)
        open(self.data_path, 'a').close()

    def validate(self, validation_set):
        """Perform a validation across external data as MoA and ATC codes."""
        self.__log.debug('pre-process validated on %s' % validation_set)
