"""Signature type 0.

A sufficiently-processed version of the raw data. Each bioactive space has
a peculiar format which might be categorical, discrete or continuous.
They usually show explicit knowledge, which enables connectivity and
interpretation.
"""
import os, sys, shutil
import h5py
import datetime
import collections
import numpy as np

from .signature_data import DataSignature
from .signature_base import BaseSignature

from chemicalchecker.util import logged
from chemicalchecker.util.sanitize import Sanitizer
from chemicalchecker.util.aggregate import Aggregate
from chemicalchecker.util.decorator import cached_property
from chemicalchecker.util.sampler.triplets import TripletSampler
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class sign0(BaseSignature, DataSignature):
    """Signature type 0 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize a Signature.

        Args:
            signature_path (str): the path to the signature directory.
            dataset (str): NS ex A1.001, here only serves as the 'name' record
               of the h5 file.
        """
        BaseSignature.__init__(self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s' % signature_path)

        self.data_path = os.path.join(self.signature_path, "sign0.h5")
        DataSignature.__init__(self, self.data_path, **params)
        self.__log.debug('data path: %s' % self.data_path)

    def process_keys(self, keys, key_type):
        """
        Given keys, process them so they are acceptable CC types. If None is specified, then all keys are kept.
        NS: Returns the processed Inchikeys, the ray_keys and the indices of the selected processed keys from the raw keys iterable.
        """
        if key_type is None:
            return np.array(keys), None, np.array([i for i in range(0, len(keys))])
        keys_ = []
        keys_raw = []
        idxs = []
        if key_type == "inchikey":
            self.__log.debug("Processing inchikeys. Only valids are kept.")
            for i, k in enumerate(keys):
                if len(k) == 27:
                    if k[14] == "-" and k[25] == "-":
                        keys_ += [k]
                        keys_raw += [k]
                        idxs += [i]
        elif key_type == "smiles":
            self.__log.debug(
                "Processing smiles. Only standard smiles are kept")
            from chemicalchecker.util.parser import Converter
            conv = Converter()
            for i, k in enumerate(keys):
                try:
                    keys_ += [conv.smiles_to_inchi(k)[0]]
                    keys_raw += [k]
                    idxs += [i]
                except:
                    continue
        else:
            raise "key_type must be 'inchikey' or 'smiles'"
        self.__log.info("Initial keys: %d / Final keys: %d" %
                        (len(keys), len(keys_)))

        return np.array(keys_), np.array(keys_raw), np.array(idxs)

    def process_features(self, features, n):
        """
        Process features. Give an arbitrary name to features if None are provided.
        NS: returns the feature names as a np array of strings
        """
        if features is None:
            self.__log.debug(
                "No features were provided, giving arbitrary names")
            l = int(np.log10(n)) + 1
            features = []
            for i in range(0, n):
                s = "%d" % i
                s = s.zfill(l)
                features += ["feature_%s" % s]
        return np.array(features).astype(str)

    def get_data(self, pairs, X, keys, features, data_file, key_type, agg_method):
        if data_file is not None:
            if not os.path.isfile(data_file):
                raise Exception("File not found: %s" % data_file)
            dh5 = h5py.File(data_file, 'r')
            if "pairs" in dh5.keys():
                pairs = dh5["pairs"][:]
                if "values" in dh5.keys():
                    pairs = zip(pairs, dh5["values"][:])
            if "X" in dh5.keys():
                X = dh5["X"][:]
            if "keys" in dh5.keys():
                keys = dh5["keys"][:]
            if "features" in dh5.keys():
                features = dh5["features"][:]
            dh5.close()
            if pairs is None and X is None:
                raise Exception(
                    "H5 file " + data_file + " does not contain datasets 'pairs' or 'X'")
        if pairs is not None:
            if X is not None:
                raise Exception(
                    "If you input pairs, X should not be specified!")
            if len(pairs[0]) == 2:
                has_values = False
            else:
                has_values = True
            self.__log.info("Input data were pairs")
            keys = list(set([x[0] for x in pairs]))
            features = list(set([x[1] for x in pairs]))
            self.__log.debug("Processing keys and features")
            self.__log.debug("Before processing:")
            self.__log.debug("KEYS: {}".format(keys))
            self.__log.debug("key_type: {}".format(key_type))

            keys, keys_raw, _ = self.process_keys(keys, key_type)
            features = self.process_features(features, len(features))
            keys_dict = dict((k, i) for i, k in enumerate(keys_raw))
            features_dict = dict((k, i) for i, k in enumerate(features))
            self.__log.debug("Iterating over pairs and doing matrix")
            pairs_ = collections.defaultdict(list)
            if not has_values:
                self.__log.debug("Binary pairs")
                for p in pairs:
                    if p[0] not in keys_dict or p[1] not in features_dict:
                        continue
                    pairs_[(keys_dict[p[0]], features_dict[p[1]])] += [1]
            else:
                self.__log.debug("Valued pairs")
                for p in pairs:
                    if p[0] not in keys_dict or p[1] not in features_dict:
                        continue
                    pairs_[(keys_dict[p[1]], features_dict[p[1]])] += [p[2]]
            X = np.zeros((len(keys), len(features)))
            self.__log.debug("Aggregating duplicates")
            if agg_method == "average":
                def do_agg(v): return np.mean(v)
            if agg_method == "first":
                def do_agg(v): return v[0]
            if agg_method == "last":
                def do_agg(v): return v[-1]
            for k, v in pairs_.items():
                X[k[0], k[1]] = do_agg(v)
            self.__log.debug("Setting input type")
            input_type = "pairs"
        else:
            if X is None:
                raise Exception(
                    "No data were provided! X cannot be None if pairs aren't provided")
            if keys is None:
                raise Exception("keys cannot be None")
            if features is None:
                raise Exception("features cannot be None")
            if X.shape[0] != len(keys):
                raise Exception(
                    "number of rows of X must equal length of keys")
            if X.shape[1] != len(features):
                raise Exception(
                    "number of columns of X must equal length of features")
            if len(features) != len(set(features)):
                raise Exception("features must be unique")
            self.__log.debug("Processing keys")
            keys, keys_raw, idxs = self.process_keys(keys, key_type)
            self.__log.debug("Processing features")
            features = self.process_features(features, X.shape[1])
            self.__log.debug("Only keeping idxs of relevance")
            self.__log.debug("keys is {}".format(keys))
            self.__log.debug("keys_raw is {}".format(keys_raw))
            self.__log.debug("idxs is {}".format(idxs))
            print("idxs is {}".format(idxs))
            X = X[idxs]
            self.__log.debug("Setting input type")
            input_type = "matrix"
        if X.shape[0] != len(keys):
            raise Exception(
                "after processing, number of rows does not equal number of columns")
        X, keys, keys_raw, features = self.sort(X, keys, keys_raw, features)
        results = {
            "X": X,
            "keys": keys,
            "keys_raw": keys_raw,
            "features": features,
            "input_type": input_type
        }
        return results

    @cached_property
    def agg_method(self):
        """Get the agg method of the signature."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if "agg_method" not in hf.keys():
                self.__log.warn("No agg_method available for this signature!")
                return None
            if hasattr(hf["agg_method"][0], 'decode'):
                return [k.decode() for k in hf["agg_method"][:]][0]
            else:
                return hf["agg_method"][0]

    @cached_property
    def input_type(self):
        """Get the input type done at fit time."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if "input_type" not in hf.keys():
                self.__log.warn("No input_type available for this signature!")
                return None
            if hasattr(hf["input_type"][0], 'decode'):
                return [k.decode() for k in hf["input_type"][:]][0]
            else:
                return hf["input_type"][0]

    @cached_property
    def key_type(self):
        """Get the key type done at fit time."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if "key_type" not in hf.keys():
                self.__log.warn("No key_type available for this signature!")
                return None
            if hasattr(hf["key_type"][0], 'decode'):
                return [k.decode() for k in hf["key_type"][:]][0]
            else:
                return hf["key_type"][0]

    @property
    def preprocessed(self):
        """Get the path to the corresponding preprocessed.h5."""
        dirname= os.path.dirname(self.data_path)
        preprocess= os.path.join(dirname,'raw','preprocess.h5')
        if os.path.exists(preprocess):
            return preprocess
        else:
            self.__log.warning("No preprocessed file has been found for {}!!".format(self.dataset))
            return None

    def refesh(self):
        DataSignature.refesh()
        self._refresh("key_type")
        self._refresh("input_type")
        self._refresh("agg_method")

    def sort(self, X, keys, keys_raw, features):
        self.__log.debug("Sorting")
        key_idxs = np.argsort(keys)
        feature_idxs = np.argsort(features)
        # sort all data
        X = X[key_idxs]
        for i in range(0, X.shape[0], 2000):
            chunk = slice(i, i + 2000)
            X[chunk] = X[chunk, feature_idxs]
        # sort keys
        keys = keys[key_idxs]
        keys_raw = keys_raw[key_idxs]
        # sort features
        features = features[feature_idxs]
        return X, keys, keys_raw, features


    def fit(self, cc=None, pairs=None, X=None, keys=None, features=None, data_file=None, key_type="inchikey", agg_method="average", do_triplets=True, validations=True, max_features=10000, chunk_size=10000, overwrite=False, **params):
        """Process the input data. We produce a sign0 (full) and a sign0(reference). Data are sorted (keys and features).

        Args:
            cc(Chemical Checker): A CC instance. This is important to produce the triplets. If None specified, the same CC where the signature is present will be used (default=None).
            pairs(array of tuples or file): Data. If file it needs to H5 file with dataset called 'pairs'.
            X(matrix or file): Data. If file it needs to H5 file with datasets called 'X', 'keys' and maybe 'features'.
            keys(array): Row names.
            key_type(str): Type of key. May be inchikey or smiles (default='inchikey').
            features(array): Column names (default=None).
            data_file(str): Input data file in the form of H5 file and it shoud contain the required data in datasets.
            validations(boolean): Create validation files(plots, files,etc)(default=True).
            do_triplets(boolean): Draw triplets from the CC (default=True).
        """
        if not overwrite and BaseSignature.fit(self):
            # NS provides a lock to avoid fitting again if it has been already done
            return

        self.clean()
        if cc is None:
            cc = self.get_cc()
        self.__log.debug("Getting data")
        self.__log.debug("data_file is {}".format(data_file))

        res = self.get_data(pairs=pairs, X=X, keys=keys,
                            features=features, data_file=data_file, key_type=key_type,
                            agg_method=agg_method)

        X = res["X"]
        keys = res["keys"]
        keys_raw = res["keys_raw"]
        features = res["features"]
        input_type = res["input_type"]

        self.__log.debug("Sanitizing")

        # NS we want to keep 2048 features (Morgan fingerprint) for sign0
        trimFeatures= False if self.dataset == 'A1.001' else True

        san = Sanitizer(trim=trimFeatures, max_features=max_features, chunk_size=chunk_size)
        X, keys, keys_raw, features = san.transform(V=X, keys=keys, keys_raw=keys_raw, features=features, sign=None)

        self.__log.debug("Aggregating if necessary")
        agg = Aggregate(method=agg_method, input_type=input_type)
        X, keys, keys_raw = agg.transform(V=X, keys=keys, keys_raw=keys_raw)

        self.__log.debug("Saving dataset")
        with h5py.File(self.data_path, "w") as hf:
            hf.create_dataset("name", data=np.array(
                [str(self.dataset) + "sig"], DataSignature.string_dtype()))
            hf.create_dataset("date", data=np.array([datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
            hf.create_dataset("V", data=X)
            hf.create_dataset("keys", data=np.array(
                keys, DataSignature.string_dtype()))
            hf.create_dataset("features", data=np.array(
                features, DataSignature.string_dtype()))
            hf.create_dataset("keys_raw", data=np.array(
                keys_raw, DataSignature.string_dtype()))
            hf.create_dataset("agg_method", data=np.array(
                [str(agg_method)], DataSignature.string_dtype()))
            hf.create_dataset("input_type", data=np.array(
                [str(input_type)], DataSignature.string_dtype()))

        self.refresh()
        self.__log.info("Removing redundancy")
        sign0_ref = self.get_molset("reference")
        sign0_ref.clean()
        rnd = RNDuplicates(cpu=10)
        rnd.remove(self.data_path, save_dest=sign0_ref.data_path)
        with h5py.File(self.data_path, "r") as hf:
            features = hf["features"][:]
        with h5py.File(sign0_ref.data_path, 'a') as hf:
            hf.create_dataset('features', data=features)
        # Making triplets
        if do_triplets:
            sampler = TripletSampler(cc, self, save=True)
            sampler.sample(**params)
        self.__log.debug("Done.")
        if validations:
            self.__log.debug("Validate")
            self.validate()
            sign0_ref.validate()
        # Marking as ready
        self.__log.debug("Mark as ready")
        sign0_ref.mark_ready()
        self.mark_ready()

    def predict(self, pairs=None, X=None, keys=None, features=None, data_file=None, key_type=None, merge=False, merge_method="new", destination=None):
        """Given data, produce a sign0.

        Args:
            pairs(array of tuples or file): Data. If file it needs to H5 file with dataset called 'pairs'.
            X(matrix or file): Data. If file it needs to H5 file with datasets called 'X', 'keys' and maybe 'features'.
            keys(array): Row names.
            key_type(str): Type of key. May be inchikey or smiles. If None specified, no filtering is applied (default=None).
            features(array): Column names (default=None).
            merge(bool): Merge queried data with the currently existing one.
            merge_method(str): Merging method to be applied when a repeated key is found. Can be 'average', 'old' or 'new' (default=new).
            destination(str): Path to the H5 file. If none specified, a (V, keys, features) tuple is returned.
            validations(boolean): Create validation files(plots, files,etc)(default=False).
        """
        assert self.is_fit(), "Signature is not fitted yet"
        self.__log.debug("Setting up the signature data based on fit")
        if merge:
            self.__log.info("Merging. Loading existing signature.")
            V_ = self[:]
            keys_ = self.keys
            keys_raw_ = self.keys_raw
            if merge_method is not None:
                if merge_method not in ["average", "new", "old"]:
                    raise Exception(
                        "merge_method must be None, 'average', 'new' or 'old'")
        else:
            self.__log.info("Not merging. Just producing signature for the inputted data.")
            V_ = None
            keys_ = None
            keys_raw_ = None
        features_ = self.features
        features_idx = dict((k, i) for i, k in enumerate(features_))
        self.__log.debug("Preparing input data")
        res = self.get_data(pairs=pairs, X=X, keys=keys, features=features, data_file=data_file, key_type=key_type,
                            agg_method=self.agg_method)
        X = res["X"]
        keys = res["keys"]
        keys_raw = res["keys_raw"]
        input_type = res["input_type"]
        features = res["features"]
        if input_type != self.input_type:
            raise Exception("Input type must be %s" % self.input_type)
        self.__log.debug(
            "Putting input in the same features arrangement than the fitted signature.")
        W = np.full((len(keys), len(features_)), np.nan)
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                feat = features[j]
                if feat not in features_idx:
                    continue
                W[i, features_idx[feat]] = X[i, j]
        X = W
        self.__log.debug("Sanitizing if necessary")
        self.refresh()
        san = Sanitizer(trim=False, chunk_size=chunk_size)
        X, keys, keys_raw, features = san.transform(
            V=X, keys=keys, keys_raw=keys_raw, features=features, sign=self)
        self.__log.debug("Aggregating as it was done at fit time")
        agg = Aggregate(method=self.agg_method, input_type=input_type)
        X, keys, keys_raw = agg.transform(V=X, keys=keys, keys_raw=keys_raw)
        features = res["features"]
        features = features_
        if V_ is None:
            V = X
        else:
            self.__log.debug("Stacking")
            V = np.vstack((V_, X))
            keys = np.append(keys_, keys)
            keys_raw = np.append(keys_raw_, keys_raw)
            self.__log.debug("Aggregating (merging) again")
            if merge_method is None:
                agg_method = self.agg_method
            if merge_method == 'new':
                agg_method = 'first'
            if merge_method == 'old':
                agg_method = 'last'
            if merge_method == 'average':
                agg_method = merge_method
            agg = Aggregate(method=agg_method, input_type=input_type)
            V, keys, keys_raw = agg.transform(
                V=V, keys=keys, keys_raw=keys_raw)
        self.__log.debug("Done")
        if destination is None:
            self.__log.debug(
                "Returning a dictionary of V, keys, features and keys_raw")
            results = {
                "V": V,
                "keys": keys,
                "features": features,
                "keys_raw": keys_raw
            }
            return results
        else:
            self.__log.debug("Saving H5 file in %s" % destination)
            with h5py.File(destination, "w") as hf:
                hf.create_dataset(
                    "name", data=np.array([str(self.dataset) + "sig"], DataSignature.string_dtype()))
                hf.create_dataset(
                    "date", data=np.array([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
                hf.create_dataset("V", data=X)
                hf.create_dataset("keys", data=np.array(
                    keys, DataSignature.string_dtype()))
                hf.create_dataset("features", data=np.array(
                    features, DataSignature.string_dtype()))
                hf.create_dataset("keys_raw", data=np.array(
                    keys_raw, DataSignature.string_dtype()))

    def restrict_to_universe(self):
        """
        Nico : 17/09/2020
        - Restricts the keys in the corresponding preprocess.h5 files to the ones contained in the universe,
        defined as the union of all molecules from bioactivity spaces (B and after).
        - Applicable when the signature belongs to one of the A spaces
        """
        cc= self.get_cc()
        universe = cc.universe  # list of inchikeys belonging to the universe
        preprocess= self.preprocessed
        keys_prepro = DataSignature._fetch_keys(preprocess)

        self.__log.debug("--> getting the vectors from s0 corresponding to our (restricted) universe")
        # get the vectors from s0 corresponding to our (restricted) universe
        inchk_univ, _ = self.get_vectors(keys=universe, data_file=preprocess, dataset_name='X')

        # obtain a mask for sign0 in order to obtain a filtered h5 file
        # Strangely, putting lists greatly improves the performances of np.isin
        self.__log.debug("--> Obtaining a mask")
        mask= np.isin(list(keys_prepro), list(inchk_univ))

        del inchk_univ  # avoiding consuming too much memory


        # Make a backup of the current sign0.h5

        
        dirname= os.path.dirname(preprocess)
        backup = os.path.join(dirname,'preprocessBACKUP.h5')
        filtered_h5= os.path.join(dirname, 'preprocess_filtered.h5')

        if not os.path.exists(backup):
            self.__log.debug("Making a backup of preprocess.h5 as {}".format(backup))
            try:
                shutil.copyfile(preprocess, backup)
            except Exception as e:
                self.__log.warning("Cannot backup {}".format(backup))
                self.__log.warning("Please check permissions")
                self.__log.warning(e)
                sys.exit(1)



        self.__log.info("Creating {}".format(filtered_h5))

        self.__log.debug("--> Creating file {}".format(filtered_h5))
        self.make_filtered_copy(filtered_h5, mask, include_all=True, data_file=preprocess)

        # After that check that your file is ok and move it to sign0.h5
        # deleting previous sign0 file
        self.__log.info("Deleting old preprocess.h5 file: {}".format(preprocess))
        try:
            os.remove(preprocess)
        except Exception as e:
            self.__log.warning("Cannot remove {}".format(preprocess))
            self.__log.warning("Please check permissions")
            self.__log.warning(e)
            sys.exit(1)

        self.__log.info("Renaming the new preprocess file:")
        try:
            shutil.copyfile(filtered_h5, preprocess)
        except Exception as e:
            self.__log.warning("Cannot copy {}".format(filtered_h5))
            self.__log.warning("Please check permissions")
            self.__log.warning(e)
            sys.exit(1)

        try:
            self.__log.warning("Removing old {}".format(filtered_h5))
            os.remove(filtered_h5)
        except Exception as e:
            self.__log.warning("Cannot remove {}".format(filtered_h5))
            self.__log.warning("Please check permissions")
            self.__log.warning(e)

        # Now that molecules have been removed, we have to sanitize (remove columns full of 0)
        # and aggregate (if two row vectors are equal, merge them using an approproate method)
        self.__log.info("--> preprocessed file filtered for space {}".format(self.dataset))
        # self.__log.info("Re performing the fit() method")
        # self.fit(data_file= preprocess)

        self.__log.info("Done\n")

    def restrict_to_universe_hpc(self, *args, **kwargs):
        return self.func_hpc("restrict_to_universe", *args, memory=15, **kwargs)

