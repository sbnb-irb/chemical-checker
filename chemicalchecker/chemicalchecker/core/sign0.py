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

    def __init__(self, signature_path, validation_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): the path to the signature directory.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, validation_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(signature_path, "sign0.h5")
        self.__log.debug('data_path: %s', self.data_path)
        self.__log.debug('param file: %s', self.param_file)
        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)

    def fit(self):
        """Signature type 0 has no models to fit."""
        """Call the external preprocess script to generate h5 data."""

        config = Config()
        preprocess_script = os.path.join(
            config.PATH.CC_REPO, "chemicalchecker/scripts/preprocess", self.dataset.code, "run.py")

        self.__log.debug('calling pre-process script ' + preprocess_script)

        if not os.path.isfile(preprocess_script):
            raise Exception("Preprocess script " +
                            preprocess_script + " does not exist")

        try:
            cmdStr = "python " + preprocess_script + " -o " + self.data_path + \
                " " + " -mp " + self.model_path + " " + " -m fit "
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

    def predict(self, input_data_file, destination, entry_point=None):
        """Call the external preprocess script to generate h5 data."""
        """
        Args:
            input_data_file(str): Path to the file with the raw to generate the signature0.
            destination(str): Path to a .h5 file where the predicted signature will be saved.
            entry_point(str): Entry point of the input data into the signaturization process. It
                                depends on the type of data passed at the input_data_file
        """

        config = Config()
        preprocess_script = os.path.join(
            config.PATH.CC_REPO, "chemicalchecker/scripts/preprocess", self.dataset.code, "run.py")

        self.__log.debug('calling pre-process script ' + preprocess_script)

        if not os.path.isfile(preprocess_script):
            raise Exception("Preprocess script " +
                            preprocess_script + " does not exist")

        self.data_path = destination

        try:
            cmdStr = "python " + preprocess_script + " -i " + input_data_file + " -o " + self.data_path + \
                " " + " -mp " + self.model_path + " " + " -m predict"
            if entry_point is not None:
                cmdStr += " -ep " + entry_point
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

    def statistics(self):
        """Perform a statistics."""
