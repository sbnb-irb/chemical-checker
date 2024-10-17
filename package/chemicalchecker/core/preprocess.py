"""Data preprocessing.

Given the diversity of formats and datasources, the signaturization process
starts in tailored pre-process scripts (available in the package ``scripts``
folder).
The `fit` method invoke the pre-process script with a `fit` argument where
we essentially `learn` the feature to consider.
The `predict` method allow deriving signatures without altering the feature
set. This can also be used when mapping
to a bioactive space different entities (i.e. not only compounds)

E.g.
categorical: "C0015230,C0016436..." which translates in n array of 0s or 1s.
discrete: "GO:0006897(8),GO:0006796(3),..." which translates in an array of
integers
continous: "0.515,1.690,0.996" which is an array of floats
"""
import os
import h5py
import argparse
import importlib
import numpy as np

from .signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset


features_file = "features.h5"


@logged
class Preprocess():
    """Preprocess class."""

    def __init__(self, signature_path, dataset, *args, **kwargs):
        """Initialize a Preprocess instance.

        This class handles calling the external run.py for each dataset and
        provide shared methods.

        Args:
            signature_path(str): the path to the signature directory.
        """
        # TODO check is kwargs are needed (D1)
        self.__log.info('Preprocess signature: %s', signature_path)

        self.raw_path = os.path.join(signature_path, "raw")
        self.raw_model_path = os.path.join(signature_path, "raw", "models")
        if not os.path.isdir(self.raw_path):
            Preprocess.__log.debug(
                "Initializing raw path: %s" % self.raw_path)
            original_umask = os.umask(0)
            os.makedirs(self.raw_path, 0o775)
            os.umask(original_umask)

        if not os.path.isdir(self.raw_model_path):
            original_umask = os.umask(0)
            os.makedirs(self.raw_model_path, 0o775)
            os.umask(original_umask)

        # where preprocess data will be saved if fit is called
        self.data_path = os.path.join(self.raw_path, "preprocess.h5")

        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.preprocess_script = os.path.join(
            dir_path,
            '../..',
            "scripts/preprocess",
            dataset,
            "run.py")
        if not os.path.isfile(self.preprocess_script):
            self.__log.warning(
                "Preprocess script not found! %s", self.preprocess_script)

    def is_fit(self):
        if os.path.exists(self.data_path):
            return True
        else:
            return False

    def call_preprocess(self, output, method, infile=None, entry=None, cores=None): #params = {}
        """Call the external pre-process script."""
        # create argument list
        arglist = ["-o", output, "-mp", self.raw_model_path, "-m", method]
        if infile:
            arglist.extend(['-i', infile])
        if entry:
            arglist.extend(['-ep', entry])
        if cores:
            arglist.extend(['-c', cores])
        # import and run the run.py
        loader = importlib.machinery.SourceFileLoader('preprocess_script',
                                                      self.preprocess_script)
        preprocess = loader.load_module()
        # self.__log.debug('ARGS: %s' % str(arglist))
        preprocess.main(arglist)

    def fit(self):
        """Call the external preprocess script to generate H5 data.

        The preprocess script is invoked with the `fit` argument, which means
        features are extracted from datasoruce and saved.
        """
        # check that preprocess script is available and call it
        self.__log.info('Calling preprocess script: %s',
                        self.preprocess_script)

        if not os.path.isfile(self.preprocess_script):
            raise Exception("Preprocess script not found! %s",
                            self.preprocess_script)

        self.call_preprocess(self.data_path, "fit", None, None, None) #self.params

    def predict(self, input_data_file, destination, entry_point, cores):
        """Call the external preprocess script to generate H5 data."""
        """
        Args:
            input_data_file(str): Path to the file with the raw to generate
                the signature 0.
            destination(str): Path to a H5 file where the predicted signature
                will be saved.
            entry_point(str): Entry point of the input data for the
                signaturization process. It depends on the type of data passed
                at the input_data_file.
        """
        # check that preprocess script is available and call it
        self.__log.info('Calling preprocess script: %s',
                        self.preprocess_script)

        if not os.path.isfile(self.preprocess_script):
            raise Exception("Pre-process script not found! %s",
                            self.preprocess_script)

        self.call_preprocess(destination, "predict", infile=input_data_file,
                             entry=entry_point, cores=cores )

    def to_features(self, signatures):
        """Convert signature to explicit feature names.

        Args:
            signatures(array): a signature 0 for 1 or more molecules
        Returns:
            list of dict: 1 dictionary per signature where keys are
                feature_name and value as values.
        """
        # handle single signature
        if len(signatures.shape) == 1:
            signatures = [signatures]
        # if no features file is available then the signature is just an array
        feature_file = os.path.join(self.model_path, "features.h5")
        if not os.path.isfile(feature_file):
            features = np.arange(len(signatures[0]))
        else:
            # read features names from file
            with h5py.File(feature_file, 'r') as hf:
                features = hf["features"][:]
        # return list of dicts with feature_name as key and value as value
        result = list()
        for sign in signatures:
            keys = features[sign != 0]
            values = sign[sign != 0]
            result.append(dict(zip(keys, values)))
        return result

    @classmethod
    def preprocess(cls, sign, **params):
        """Return the file with the raw data preprocessed.
        Args:
            sign: signature object (e.g. obtained from cc.get_signature)
            params: specific parameters for a given preprocess script
        Returns:
            datafile(str): The name of the file where the data is saved.

        ex:
            os.path.join(self.raw_path, "preprocess.h5")
        """

        prepro = cls(sign.signature_path, sign.dataset, **params)
        if not prepro.is_fit():
            cls.__log.info(
                "No raw data file found, calling the preprocessing script")
            prepro.fit()
        else:
            cls.__log.info("Found {}".format(prepro.data_path))
        return prepro.data_path

    @classmethod
    def preprocess_predict(cls, sign, input_file, destination, entry_point, cores=None):
        """Runs the preprocessing script 'predict'.

        Run on an input file of raw data formatted correctly for the space of
        interest

        Args:
            sign: signature object ( e.g. obtained from cc.get_signature)
            input_file(str): path to the H5 file containing the data on which
                to apply 'predict'
            destination(str): Path to a H5 file where the predicted signature
                will be saved.
            entry_point(str): Entry point of the input data for the
                signaturization process. It depends on the type of data passed
                at the input_data_file.
        Returns:
            datafile(str): The H5 file containing the predicted data after
                preprocess
        """

        input_file = os.path.abspath(input_file)
        destination = os.path.abspath(destination)

        # Checking the provided paths

        if not os.path.exists(input_file):
            raise Exception("Error, {} does not exist!".format(input_file))

        prepro = cls(sign.signature_path, sign.dataset)
        prepro.predict(input_file, destination, entry_point, cores)

        return destination

    @staticmethod
    def get_parser():
        description = 'Run preprocess script.'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('-i', '--input_file', type=str,
                            required=False, default='.',
                            help='Input file only for predict method')
        parser.add_argument('-o', '--output_file', type=str,
                            required=False, default='.', help='Output file')
        parser.add_argument('-m', '--method', type=str,
                            required=False, default='fit',
                            help='Method: fit or predict')
        parser.add_argument('-mp', '--models_path', type=str,
                            required=False, default='',
                            help='The models path')
        parser.add_argument('-ep', '--entry_point', type=str,
                            required=False, default=None,
                            help='The predict entry point')
        parser.add_argument('-c', '--cores', type=int,
                            required=False, default=None,
                            help='Number of cores to use for chemical properties calculation')
        return parser

    @staticmethod
    def get_datasources(dataset_code):

        dataset = Dataset.get(dataset_code)

        map_files = {}

        for ds in dataset.datasources:
            map_files[ds.name] = ds.data_path

        return map_files

    @staticmethod
    def save_output(output_file, inchikey_raw, method, models_path, discrete,
                    features, features_int=False, chunk=2000):
        """Save raw data produced by the preprocess script as matrix.

        The result of preprocess scripts are usually in compact format (e.g.
        binary data only list features with value of 1) since data might be
        sparse and memory intensive to handle. This method convert it to a
        signature like (explicit, extended) format. The produced H5 will
        contain 3 dataset:
            * 'keys': identifier (usually inchikey),
            * 'features': features names,
            * 'X': the data matrix

        Args:
            output_file(str): Path to output H5 file.
            inchikey_raw(dict): inchikey -> list of values (dense format).
            method(str): Same as used in the preprocess script.
            models_path(str): Path to signature models directory.
            discrete(bool): True if data is binary/discrete, False for
                continuous data.
            features(list): List of feature names from original sign0,
                None when method is 'fit'.
            features_int(str): Features have no name, we can use integers as
                feature names.
            chunk(int): Chunk size for loading data.
        """
        keys = []

        if discrete:
            # check if categorical
            categ = False
            for k, v in inchikey_raw.items():
                if len(v) > 0:
                    if isinstance(v[0], tuple):
                        categ = True
                    break

            # words are all possible features
            words = set()
            for k in sorted(inchikey_raw.keys()):
                keys.append(str(k))
                if categ:
                    for word in inchikey_raw[k]:
                        words.add(word[0])
                else:
                    words.update(inchikey_raw[k])

            # if we have features available ('predict' method) check overlap
            if features is not None:
                orderwords = features
                Preprocess.__log.debug(
                    "Predict entries have a total of %s features,"
                    " %s overlap with trainset and will be considered.",
                    len(words), len(set(features) & words))
            # otherwise deduce features from data provided and sort them
            else:
                orderwords = list(words)
                del words
                if features_int:
                    orderwords.sort(key=int)
                else:
                    orderwords.sort()

            # prepare output file
            Preprocess.__log.info("Output file will be of shape: "
                                  "%s" % [len(keys), len(orderwords)])
            with h5py.File(output_file, "w") as hf:
                hf.create_dataset("keys", data=np.array(
                    keys, DataSignature.string_dtype()))
                hf.create_dataset(
                    "X", (len(keys), len(orderwords)), dtype='int8' )
                hf.create_dataset("features", data=np.array(
                    orderwords, DataSignature.string_dtype()))

            # write data in H5
            raws = np.zeros((chunk, len(orderwords)), dtype='int8' )
            wordspos = {k: v for v, k in enumerate(orderwords)}
            index = 0
            for i, k in enumerate(keys):
                # prepare chunk
                shared_features = set(inchikey_raw[k]) & set(orderwords)
                if len(shared_features) == 0:
                    Preprocess.__log.warn(
                        "%s has no shared features with trainset.", k)
                for word in inchikey_raw[k]:
                    if categ:
                        raws[index][wordspos[word[0]]] = word[1]
                    else:
                        raws[index][wordspos[word]] = 1
                index += 1

                # when chunk is complete or molecules are over, write to file
                if index == chunk or i == len(keys) - 1:
                    end = i + 1
                    if index != chunk:
                        chunk = index
                    with h5py.File(output_file, "r+") as hf:
                        dataset = hf["X"]
                        dataset[end - chunk:end] = raws[:chunk]

                    raws = np.zeros((chunk, len(orderwords)), dtype=np.int8)
                    index = 0
            saving_features = orderwords
        # continuous
        else:
            # get molecules inchikeys
            for k in inchikey_raw.keys():
                keys.append(str(k))
            keys = np.array(keys)
            inds = keys.argsort()
            data = []

            # sorted data
            for i in inds:
                data.append(inchikey_raw[keys[i]])

            # define features if not available
            if features is None:
                features = [str(i) for i in range(1, len(data[0]) + 1)]

            # save data
            with h5py.File(output_file, "w") as hf:
                hf.create_dataset("keys", data=np.array(
                    keys[inds], DataSignature.string_dtype()))
                hf.create_dataset("X", data=np.array(data))
                hf.create_dataset("features", data=np.array(
                    features, DataSignature.string_dtype()))

            saving_features = features 

        # if fitting, we also save the features
        if method == "fit":
            fn = os.path.join(models_path, features_file)
            with h5py.File(fn, "w") as hf:
                hf.create_dataset("features", data=np.array(saving_features, DataSignature.string_dtype()))
            

    def to_feature_string(self, signatures, string_func):
        """Covert signature to a string with feature names.

        Args:
            signatures(array): Signature array(s).
            string_func(func): A function taking a dictionary as input and
                returning a single string.
        """
        result_dicts = self.to_features(signatures)
        result_strings = list()
        for res_dict in result_dicts:
            result_strings.append(string_func(res_dict))
        return result_strings

    @staticmethod
    def _feat_key_only(res_dict):
        """Suited for discrete spaces."""
        strings = list()
        for k in sorted(res_dict.keys()):
            strings.append("%s" % k)
        return ','.join(strings)

    @staticmethod
    def _feat_value_only(res_dict):
        """Suited for continuous spaces."""
        strings = list()
        for k in sorted(res_dict.keys()):
            strings.append("%.3f" % res_dict[k])
        return ','.join(strings)

    @staticmethod
    def _feat_key_values(res_dict):
        """Suited for discrete spaces with values."""
        strings = list()
        for k in sorted(res_dict.keys()):
            strings.append("%s(%s)" % (k, res_dict[k]))
        return ','.join(strings)

    # def _compare_to_old(self, old_dbname, to_sample=1000):
    #     """Compare current signature 0 to previous format.

    #     Args:
    #         old_dbname(str): the name of the old db (e.g. 'mosaic').
    #         to_sample(int): Number of signatures to compare in the set of
    #             shared moleules.

    #     """
    #     try:
    #         from chemicalchecker.util import psql
    #     except ImportError as err:
    #         raise err

    #     old_table_names = {
    #         'A1': 'fp2d',
    #         'A2': 'fp3d',
    #         'A3': 'scaffolds',
    #         'A4': 'subskeys',
    #         'A5': 'physchem',
    #         'B1': 'moa',
    #         'B2': 'metabgenes',
    #         'B3': 'crystals',
    #         'B4': 'binding',
    #         'B5': 'htsbioass',
    #         'C1': 'molroles',
    #         'C2': 'molpathways',
    #         'C3': 'pathways',
    #         'C4': 'bps',
    #         'C5': 'networks',
    #         'D1': 'transcript',
    #         'D2': 'cellpanel',
    #         'D3': 'chemgenet',
    #         'D4': 'morphology',
    #         'D5': 'cellbioass',
    #         'E1': 'therapareas',
    #         'E2': 'indications',
    #         'E3': 'sideeffects',
    #         'E4': 'phenotypes',
    #         'E5': 'ddis'
    #     }
    #     table_name = old_table_names[self.dataset[:2]]
    #     string_funcs = {
    #         'A1': sign0._feat_key_only,
    #         'A2': sign0._feat_key_only,
    #         'A3': sign0._feat_key_only,
    #         'A4': sign0._feat_key_only,
    #         'A5': sign0._feat_value_only,
    #         'B1': sign0._feat_key_only,
    #         'B2': sign0._feat_key_only,
    #         'B3': sign0._feat_key_only,
    #         'B4': sign0._feat_key_values,
    #         'B5': sign0._feat_key_only,
    #         'C1': sign0._feat_key_only,
    #         'C2': sign0._feat_key_values,
    #         'C3': sign0._feat_key_values,
    #         'C4': sign0._feat_key_values,
    #         'C5': sign0._feat_key_values,
    #         'D1': sign0._feat_key_values,
    #         'D2': sign0._feat_value_only,
    #         'D3': sign0._feat_key_values,
    #         'D4': sign0._feat_value_only,
    #         'D5': sign0._feat_key_only,
    #         'E1': sign0._feat_key_only,
    #         'E2': sign0._feat_key_only,
    #         'E3': sign0._feat_key_only,
    #         'E4': sign0._feat_key_only,
    #         'E5': sign0._feat_key_only
    #     }
    #     continuous = ["A5", "D2", "D4"]
    #     string_func = string_funcs[self.dataset[:2]]
    #     if not self.dataset.startswith("A"):
    #         # get old keys
    #         res = psql.qstring('SELECT inchikey FROM %s;' %
    #                            table_name, old_dbname)
    #         old_keys = set(r[0] for r in res)
    #         # compare to new
    #         old_only_keys = old_keys - self.unique_keys
    #         new_only_keys = self.unique_keys - old_keys
    #         shared_keys = self.unique_keys & old_keys
    #         frac_present = len(shared_keys) / float(len(old_keys))
    #         self.__log.info(
    #    "Among %s OLD molecules %.2f%% are still present:",
    #                         len(old_keys),
    #                         100 * frac_present)
    #         self.__log.info("Old keys: %s", len(old_keys))
    #         self.__log.info("New keys: %s", len(self.unique_keys))
    #         self.__log.info("Shared keys: %s", len(shared_keys))
    #         self.__log.info("Old only keys: %s", len(old_only_keys))
    #         self.__log.info("New only keys: %s", len(new_only_keys))
    #     else:
    #         shared_keys = self.keys
    #         frac_present = 1.0
    #     # randomly check sample entries
    #     total = 0.0
    #     shared = 0.0
    #     changed = 0
    #     not_changed = 0
    #     most_diff = {
    #         'shared': 99999,
    #         'key': None,
    #         'old_sign': None,
    #         'new_sign': None
    #     }
    #     to_sample = min(len(shared_keys), to_sample)
    #     sample = np.random.choice(list(shared_keys), to_sample,
    #                               replace=False)
    #     res = psql.qstring(
    #         "SELECT inchikey,raw FROM %s WHERE inchikey =  ANY('{%s}');" %
    #         (table_name, ','.join(sample)), old_dbname)
    #     res = dict(res)
    #     for ink in tqdm(sample):
    #         feat_old = set(res[ink].split(','))
    #         if self.dataset[:2] in continuous:
    #             feat_old = set(["%.3f" % float(x)
    #                             for x in res[ink].split(',')])
    #         feat_new = set(self.to_feature_string(
    #             self[ink.encode()], string_func)[0].split(','))
    #         if feat_new == feat_old:
    #             not_changed += 1
    #         else:
    #             changed += 1
    #             curr_shared = len(feat_new & feat_old)
    #             shared += curr_shared
    #             if curr_shared < most_diff['shared']:
    #                 most_diff['shared'] = curr_shared
    #                 most_diff['key'] = ink
    #                 most_diff['old_sign'] = feat_old
    #                 most_diff['new_sign'] = feat_new
    #             total += len(feat_old)
    #     frac_equal = not_changed / float(to_sample)
    #     self.__log.info(
    #    "Among %s shared sampled signatures %.2f%% are equal:",
    #                     to_sample, 100 * frac_equal)
    #     self.__log.info("Equal: %s Changed: %s", not_changed, changed)
    #     if changed == 0:
    #         return frac_present, frac_equal, 1.0
    #     if total == 0.:
    #         frac_equal_feat = 0.0
    #     else:
    #         frac_equal_feat = shared / float(total)
    #     self.__log.info("Among changed %.2f%% of features are equal to old",
    #                     100 * frac_equal_feat)
    #     self.__log.info("Most different signature %s" % most_diff['key'])
    #     self.__log.info("OLD: %s" % sorted(list(most_diff['old_sign'])))
    #     self.__log.info("NEW: %s" % sorted(list(most_diff['new_sign'])))
    #     return frac_present, frac_equal, frac_equal_feat
