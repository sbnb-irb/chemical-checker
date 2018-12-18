from chemicalchecker.util import logged
from chemicalchecker.util import Config
from .signature_base import BaseSignature
from subprocess import call
import os
import sys


@logged
class sign0(BaseSignature):
    """Signature type 0 class.

    Signature type 0 is...
    """

    def __init__(self, data_path, model_path, plots_path, dataset_info, **params):
        """Initialize the signature.

        Args:
            data_path(str): Where the h5 file is.
            model_path(str): Where the persistent model is.
        """
        self.__log.debug('data_path: %s', data_path)
        self.data_path = data_path
        self.__log.debug('model_path: %s', model_path)
        self.model_path = model_path
        self.dataset_info = dataset_info
        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, data_path, model_path, dataset_info)

    def fit(self):
        """Signature type 0 has no models to fit."""
        self.__log.debug('nothing to fit.')

    def predict(self, preprocess_script=None):
        """Call the external preprocess script to generate h5 data."""
        if preprocess_script is None:
            config = Config()
            preprocess_script = os.path.join(
                config.PATH.CC_REPO, "exemplary/preprocess", self.dataset_info.code, "run.py")

        self.__log.debug('calling pre-process script ' + preprocess_script)

        if not os.path.isfile(preprocess_script):
            raise Exception("Preprocess script " +
                            preprocess_script + " does not exist")

        try:
            cmdStr = "python " + preprocess_script + " -o " + self.data_path
            retcode = call(cmdStr, shell=True)
            self.__log.debug("FINISHED! " + cmdStr +
                             (" returned code %d" % retcode))
            if retcode != 0:
                if retcode > 0:
                    self.__log.error(
                        "ERROR return code %d, please check!" % retcode)
                elif retcode < 0:
                    self.__log.error(
                        "Command terminated by signal %d" % -retcode)
                sys.exit(1)
        except OSError as e:
            self.__log.critical("Execution failed: %s" % e)
            sys.exit(1)

        self.__log.debug('faking data in %s', self.data_path)
        # open(self.data_path, 'a').close()

    def validate(self, validation_set):
        """Perform a validation across external data as MoA and ATC codes."""
        self.__log.debug('pre-process validated on %s' % validation_set)
