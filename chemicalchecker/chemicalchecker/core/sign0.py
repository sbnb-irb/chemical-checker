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


    def to_features(self, signatures):
        """Convert signature to explicit feature names.

        Args:
            signatures(array): a signature 0 for 1 or more molecules
        Returns:
            list of dict: 1 dictionary per signature where keys are
                feature_name and value as values.
        """
        # early stop if features file is not there
        feature_file = os.path.join(self.model_path, "features.h5")
        if not os.path.isfile(feature_file):
            self.__log.warn("No feature file found.")
            return None
        # read features names from file
        with h5py.File(feature_file) as hf:
            features = hf["features"][:]
        # handle single signature
        if len(signatures.shape) == 1:
            signatures = [signatures]
        # return list of dicts with feature_name as key and value as value
        result = list()
        for sign in signatures:
            keys = features[sign != 0]
            values = sign[sign != 0]
            result.append(dict(zip(keys, values)))
        return result

    def _compare_to_old(self, old_db_dump, to_sample=1000):
        """Compare current signature 0 to previous format.

        The db dump can be obtained running the following:
        psql -h aloy-dbsrv -U <user_name> -d mosaic -c "COPY (SELECT * FROM <table_name>) TO STDOUT WITH CSV" > file.csv

        Args:
            old_db_dump(str): file name with old raw features dumped from db.
            to_sample(int): Number of signatures to compare in the set of
                shared moleules.

        """
        # get old keys
        with open(old_db_dump, 'r') as fh:
            lines = fh.readlines()
        old_keys = set(l[:27] for l in lines)
        # compare to new
        old_only_keys = old_keys - self.unique_keys
        new_only_keys = self.unique_keys - old_keys
        self.__log.info("Old keys: %s", len(old_keys))
        self.__log.info("New keys: %s", len(self.unique_keys))
        self.__log.info("Shared keys: %s", len(self.unique_keys & old_keys))
        self.__log.info("Old only keys: %s", len(old_only_keys))
        self.__log.info("New only keys: %s", len(new_only_keys))

        old_has_more = 0
        new_has_more = 0
        # randomly sample 1000 entries
        to_sample = min(len(lines), to_sample)
        lines = np.random.choice(lines, to_sample, replace=False)
        for line in tqdm(lines):
            ink = line[:27]
            if ink not in self.unique_keys:
                continue
            feat_old = set(line[29:-2].split(','))
            feat_new = set(self.to_features(self[ink])[0].keys())
            if len(feat_old - feat_new) != 0:
                old_has_more += 1
            if len(feat_new - feat_old) != 0:
                new_has_more += 1
        self.__log.info("Among %s shared sampled signatures:", to_sample)
        self.__log.info("Old features set larger: %s", old_has_more)
        self.__log.info("New features set larger: %s", new_has_more)
