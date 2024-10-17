import os
import pickle
import re
import shutil
import joblib

import h5py

import numpy as np
from chemicalchecker.util import logged

from chemicalchecker.core import ChemicalChecker
from signaturizer import Signaturizer as SignaturizerExternal
from ..utils import chemistry

from ..utils import HPCUtils
from chemicalchecker.util import Config
import uuid
from sklearn.preprocessing import RobustScaler

MAXQUEUE = 15

@logged
class Prediction(HPCUtils):
    """
    Helper class to carry out external prediction.

    Args:
        directory_root(str): where to search for models
        depth(int): number of directories to search into
        data(list): list of inchis to test
        output_directory(str): folder to output predictions
        datasets(list/str): dataset to search inchis
    """

    def __init__(self,
                 directory_root,
                 data,
                 output_directory="./",
                 depth=2,
                 datasets=None,
                 molrepo="chembl",
                 compressed='',
                 hpc=False,
                 n_jobs_hpc=8,
                 tmp_path=None,
                 chunk_size=None,
                 overwrite=False,
                 wipe=True,
                 log="INFO",
                 use_cc=True,
                 is_classic=False,
                 keytype='InChiKey',
                 cctype="sign3",
                 signature_type=None,
                 pchembl=None,
                 fold=None,
                 running=False,
                 **kwargs):

        HPCUtils.__init__(self, **kwargs)

        self.root = os.path.abspath(directory_root)
        assert type(data) == str or type(data) == np.ndarray or type(data) == list, "Unknown data format"

        self.datasets = datasets
        self.data = data
        self.output = os.path.abspath(output_directory)
        if not os.path.isdir(self.output):
            os.mkdir(self.output)
        self.depth = depth
        self.molrepo = molrepo.upper()
        self.hpc = hpc
        if self.hpc:
            self.n_jobs_hpc = n_jobs_hpc
        if not tmp_path:
            subpath = self.root.rstrip("/").split("/")[-1]
            self.tmp_path = os.path.join(Config().PATH.CC_TMP, "targetmate", subpath, str(uuid.uuid4()))
        else:
            self.tmp_path = os.path.abspath(tmp_path)
        if not os.path.isdir(self.tmp_path):
            os.makedirs(self.tmp_path, exist_ok=True)
        self.chunk_size = chunk_size
        self.wipe = wipe
        self.overwrite = overwrite
        self.compressed = compressed
        self.use_cc = use_cc
        self.is_classic = is_classic
        self.keytype = keytype
        self.cctype = cctype
        if signature_type is not None:
            if signature_type == 'CC':
                self.use_cc = True
            elif signature_type == 'Signaturizer':
                self.use_cc = False
                self.is_classic = False
            elif signature_type == 'ECFP4':
                self.use_cc = False
                self.is_classic = False
            elif type(signature_type) == str:
                self.__log.info("Signature type not available")
        else:
            self.__log.info("Setting signature type from variables")
        if pchembl is not None:
            self.pchembl = "pchembl_{:d}".format(pchembl * 100)
        else:
            self.pchembl = pchembl
        self.fold = fold
        self.running = running

    def _prepare_cc(self, datasets, cctype):
        cc = ChemicalChecker()
        if datasets is None:
            datasets = cc.datasets_exemplary()
        sign = []
        for dataset in datasets:
            sign += [cc.signature(dataset, cctype)]
        return sign

    def _signaturize_fingerprinter(self, smiles):
        """Calculate fingerprints"""
        V = chemistry.morgan_matrix(smiles)
        return V

    def _signaturize_signaturizer(self, molecules, moleculetype='SMILES'):
        V = None
        for ds in self.datasets:
            s3_ = SignaturizerExternal(ds.split(".")[0])
            s = s3_.predict(molecules, keytype=moleculetype)
            v = s.signature[~s.failed]
            if V is None:
                V = v
            else:
                V = np.hstack([V, v])
        return V

    def _read_signatures_by_inchikey_from_cc(self, inchikeys):
        """Read signatures from CC. InChIKeys are used."""
        iks_dict = dict((k, i) for i, k in enumerate(inchikeys))
        V = None
        for s in self.sign:
            keys, v = s.get_vectors_lite(inchikeys)
            if V is None:
                V = v
            else:
                V = np.hstack([V, v])
        idxs = np.array([iks_dict[k] for k in keys if k in iks_dict]).astype(int)
        # self.__log.debug("Signature read: {}".format(V.shape))
        return V, idxs

    def model_iterator(self, mod_dir, uncalib):
        for m in os.listdir(mod_dir):
            m_ = m.split(".z")[0]  # TODO: Consider additional variable so that can change type of compression
            if m_.split("---")[-1] == "base_model": continue
            if uncalib is False:
                if m_.split("-")[1] == "uncalib": continue
            else:
                if m_.split("-")[1] != "uncalib": continue

            fn = os.path.join(mod_dir, m)
            with open(fn, "rb") as f:
                mod = joblib.load(fn)
            yield mod

    def search_models(self, model_directory):
        dat = []
        if type(model_directory) == str:
            for n in os.listdir(model_directory):
                if os.path.isdir(os.path.join(model_directory, n)):
                    dat.append(os.path.join(model_directory, n))
        elif type(model_directory) == list:
            for m in model_directory:
                for n in os.listdir(m):
                    if os.path.isdir(os.path.join(m, n)):
                        dat.append(os.path.join(m, n))
        return dat

    def get_models(self, targets):
        directories = self.search_models(self.root)
        self.depth -= 1
        while self.depth > 0:
            directories = self.search_models(directories)
            self.depth -= 1
        directories = [d for d in directories if os.path.isdir(os.path.join(d, "bases"))]
        if self.pchembl is not None:
            directories = [d for d in directories if self.pchembl in d]
        if targets is not None:
            directories = [d for d in directories for t in targets if t in d]
        if self.running:
            directories = [d for d in directories if os.path.exists(os.path.join(d, "bases", "Stacked---0.z"))]
        return directories

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def load_signatures(self, sign_f):
        with h5py.File(sign_f, "r") as f:
            # List all groups
            X = f["signature"][:] # TODO: Currently only for signaturizer, apply to CC as well
        return X

    def _is_done(self, target):
        if os.path.exists(os.path.join(self.output, target + ".pkl")):
            self.__log.info("File exists in %s" % self.output)
            return True
        else:
            return False

    def _predict(self, mod, X, target, iteration):
        if iteration is not None:
            destination_path = os.path.join(self.output, "%s_%d.pkl" % (target, iteration))
        else:
            destination_path = os.path.join(self.output, "%s.pkl" % target)
        if os.path.exists(destination_path): os.remove(
            destination_path)

        if self.fold is not None:
            if not os.path.isdir(os.path.join(mod, "bases", self.fold)): return None
        else:
            if not os.path.isdir(os.path.join(mod, "bases")): return None

        preds = None

        if self.fold is not None:
            mod_path = os.path.join(mod, "bases", self.fold)
        else:
            mod_path = os.path.join(mod, "bases")
        smoothing = None
        for i, mod in enumerate(self.model_iterator(mod_path, uncalib=False)):
            if smoothing is None:
                smoothing = np.random.uniform(0, 1, size=(
                    len(X), len(mod.classes)))  # Added by Paula: Apply same smoothing to all ccp
            p = mod.predict(X, smoothing=smoothing)
            if preds is None:
                preds = p
            else:
                preds = preds + p
        preds = preds / (i + 1)
        with open(destination_path, "wb") as f:
            pickle.dump(preds, f)

    def predict(self, mod, X, target, iteration=None):
        return self._predict(mod, X, target, iteration)

    def incorporate_background(self, X, mod):
        test_idx = len(X)
        with open(os.path.join(mod, "trained_data.pkl"), "rb") as f:
            td = pickle.load(f)
        if self.use_cc:
            train, idx = self._read_signatures_by_inchikey_from_cc(td.inchikey)
        else:
            if self.is_classic:
                train = self._signaturize_fingerprinter(td.molecule)
            else:
                train = self._signaturize_signaturizer(td.molecule, td.moleculetype)

        X = np.vstack([X, train])
        idx = np.zeros(len(X))
        idx[:test_idx] = 1
        return X, idx

    def predict_background(self, mod, X, target, iteration=None):
        if iteration is not None:
            destination_path = os.path.join(self.output, "%s_%d.pkl" % (target, iteration))
        else:
            destination_path = os.path.join(self.output, "%s.pkl" % target)
        if os.path.exists(destination_path): os.remove(
            destination_path)

        if self.fold is not None:
            if not os.path.isdir(os.path.join(mod, "bases", self.fold)): return None
        else:
            if not os.path.isdir(os.path.join(mod, "bases")): return None

        preds = None
        zscore = None

        if self.fold is not None:
            mod_path = os.path.join(mod, "bases", self.fold)
        else:
            mod_path = os.path.join(mod, "bases")
        X, test_idx = self.incorporate_background(X, mod)
        i = False
        smoothing = None  # Added by Paula: currently set class size to 2, eed to make so that it depends on model
        for i, mod in enumerate(self.model_iterator(mod_path, uncalib=False)):
            if smoothing is None:
                smoothing = np.random.uniform(0, 1, size=(
                    len(X), len(mod.classes)))  # Added by Paula: Apply same smoothing to all ccp

            p = mod.predict(X, smoothing=smoothing)
            if preds is None:
                preds = p
            else:
                preds = preds + p

            RS = RobustScaler().fit(p[test_idx == 0, 1].reshape(-1, 1))
            if zscore is None:
                zscore = RS.transform(p[:, 1].reshape(-1, 1)).flatten()
            else:
                zscore = zscore + RS.transform(p[:, 1].reshape(-1, 1)).flatten()
        if i is False:
            return None
        preds = preds / (i + 1)
        zscore = zscore / (i + 1)
        preds = np.vstack([preds.transpose(), zscore, test_idx]).transpose()
        self.__log.debug("Storing prediciton: {:s}".format(destination_path))
        with open(destination_path, "wb") as f:
            pickle.dump(preds, f)

    def run(self, target_name_location=-2, zscore=False, targets=None):

        if self.molrepo == 'CHEMBL':
            p = re.compile('CHEMBL[0-9]+')
        else:
            assert target_name_location < 0, "Incorrect target name location: must be negative integer"
        self.__log.info("Getting models")

        directories = self.get_models()
        self.__log.info("Will calculate predictions for {:d} proteins".format(len(directories)))



        self.__log.info("Loading compounds to predict")
        jobs = []
        if type(self.data) == str:
            self.__log.info("File to data used: loading signatures")
            self.data = os.path.abspath(self.data)
            data = self.load_signatures(self.data)
        elif (type(self.data) == np.ndarray) and (self.data.dtype == float ):
            self.__log.info("Signatures introduced")
            data = self.data
        elif type(self.data) == list:
            if self.use_cc:
                self.sign = self._prepare_cc(self.datasets, self.cctype)
                data, idx = self._read_signatures_by_inchikey_from_cc(self.data)
            else:
                if self.is_classic:
                    data = self._signaturize_fingerprinter(self.data)
                else:
                    data = self._signaturize_signaturizer(self.data, self.keytype)



        for d in directories:
            if self.molrepo == 'CHEMBL':
                target = p.search(d).group()
            else:
                target = d.split("/")[target_name_location]
            if targets is not None:
                if target not in targets: continue

            if not self.overwrite:
                if self._is_done(target):
                    self.__log.warn("{:s} already exists".format(target))
                    continue

            self.__log.info("Working on {:s}".format(target))
            self.__log.info("Predicting compounds")

            if self.hpc:
                if zscore:

                    jobs += [
                        self.func_hpc("predict_background", d, data, target, None, cpu=self.n_jobs_hpc,
                                      job_base_path=self.tmp_path)]
                else:
                    if self.chunk_size:  # TODO: Add alert so that if compound library is larger than X recommend using chunksize
                        for j, c in tqdm(enumerate(self.chunker(data, self.chunk_size))):
                            jobs += [
                                self.func_hpc("predict", d, c, target, j, cpu=self.n_jobs_hpc,
                                              job_base_path=self.tmp_path)]
                    else:
                        jobs += [self.func_hpc("predict", d, data, target, None, cpu=self.n_jobs_hpc,
                                               job_base_path=self.tmp_path)]


                if len(jobs) > MAXQUEUE:
                    self.waiter(jobs)
                    jobs = []
            else:
                if zscore:
                    self.predict_background(d, data, target)
                else:
                    self.predict(d, data, target)

        if self.hpc:
            self.waiter(jobs)
            if self.wipe:
                self.__log.info("Deleting temporary directories")
                shutil.rmtree(self.tmp_path, ignore_errors=True)
