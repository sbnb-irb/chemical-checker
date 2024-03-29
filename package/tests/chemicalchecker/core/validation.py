"""Generate validation sets.

Fetch the validation set generation scripts and run them.
"""
import os
import sys
from subprocess import call

from chemicalchecker.util import logged, Config


@logged
class Validation():
    """Validation class.

    Creates a validation set.
    """

    def __init__(self, validation_path, name):
        """Initialize a Validation instance.

        Args:
            name(str): The name of the validatoion set.
        """
        config = Config()
        self.name = name
        self.validation_sets_path = validation_path
        if not os.path.isdir(self.validation_sets_path):
            self.__log.warning("Creating validation sets dir")
            original_umask = os.umask(0)
            os.makedirs(self.validation_sets_path, 0o775)
            os.umask(original_umask)

        self.validation_script = os.path.join(
            config.PATH.CC_REPO, "package/scripts/validation_sets", name,
            "run.py")

        if not os.path.isfile(self.validation_script):
            raise Exception(
                "The validation script for validation set " + name +
                " is not available.")

        self.destination_file = os.path.join(
            self.validation_sets_path, self.name + "_validation.tsv")

    def run(self, destination=None):
        """Run the validation script."""
        if destination is None:
            destination = self.destination_file
        try:
            cmdStr = "python " + self.validation_script + " -o " + destination
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
