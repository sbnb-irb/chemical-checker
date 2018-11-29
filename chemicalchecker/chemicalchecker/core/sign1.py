from chemicalchecker.util import logged
from .signature_base import BaseSignature


@logged
class sign1(BaseSignature):
    """Signature type 1 class.

    Signature type 1 is...
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
        # Calling base class to trigger file existence checks
        BaseSignature.__init__(self, data_path, model_path)

    def fit(self, sign0):
        """Take `sign0` and learn an unsupervised `sign1` predictor.

        Args:
            sign0(sign0): a `sign0` instance.
        """
        # Calling base class to trigger file existence checks
        BaseSignature.fit(self)
        self.__log.debug('LSI/PCA fit %s' % sign0)
        for data in sign0:
            pass
        self.__log.debug('saving model to %s' % self.model_path)
        open(self.model_path, 'a').close()

    def predict(self, sign0, destination=None):
        """Take `sign0` and predict `sign1`.

        Args:
            sign0(sign0): a `sign0` instance.
            destination(str): where to save the prediction by default the
                current signature data path.
        """
        # Calling base class to trigger file existence checks
        BaseSignature.predict(self)
        self.__log.debug('loading model from %s' % self.model_path)
        self.__log.debug('LSI/PCA predict %s' % sign0)
        for data in sign0:
            pass
        if not destination:
            destination = self.data_path
        self.__log.debug('generating %s' % destination)
        open(destination, 'a').close()

    def validate(self, validation_set):
        """Perform a validation across external data as MoA and ATC codes."""
        # Calling base class to trigger file existence checks
        BaseSignature.validate(self)
        self.__log.debug('LSI/PCA validated on %s' % validation_set)
