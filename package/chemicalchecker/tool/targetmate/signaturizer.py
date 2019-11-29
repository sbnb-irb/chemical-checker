import h5py
import os
import numpy as np

from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import logged

from .utils import chemistry
from .utils import HPCUtils
from .tmsetup import TargetMateSetup

# Utils
@logged
class SignUtils(object):

    def __init__(self):
        self.destination_dirs = {}

    def _fingerprinter__init__(self, cls):
        featurizer_func = chemistry.morgan_matrix
        datasets = ["FP"]
        dataset  = datasets[0]
        return featurizer_func, datasets, dataset
    
    def _signaturizer__init__(self, cls, datasets, sign_predict_paths):
        if not datasets:
            # self.datasets = list(self.cc.datasets)
            datasets = ["%s%s.001" % (x, y) for x in "ABCDE" for y in "12345"]
        else:
            datasets = datasets
        # preloaded neural networks
        if not sign_predict_paths:
            sign_predict_paths = {}
            for ds in datasets:
                self.__log.debug("Loading signature predictor for %s" % ds)
                s3 = cls.cc.get_signature("sign3", "full", ds)
                sign_predict_paths[ds] = s3
        else:
            sign_predict_paths = sign_predict_paths
        return datasets, sign_predict_paths

    def _featurizer(self, cls, smiles, destination_dir):
        V = cls.featurizer_func(smiles)
        with h5py.File(destination_dir, "w") as hf:
            hf.create_dataset("V", data = V.astype(np.int8))
            hf.create_dataset("keys", data = np.array(smiles, DataSignature.string_dtype()))

    def _fingerprinter_signaturize(self, cls, smiles, is_tmp=None, wait=True, **kwargs):
        destination_dir = cls.get_destination_dir(dataset=None, is_tmp=is_tmp)
        self.destination_dirs[cls.dataset] = destination_dir
        jobs = []
        if os.path.exists(destination_dir):
            self.__log.debug("Fingerprint file already exists: %s" %  destination_dir)
        else:
            self.__log.debug("Calculating fingerprint")
            if not cls.hpc:
                cls.featurizer(smiles, destination_dir)
            else:
                job = cls.func_hpc("featurizer",
                                    smiles, 
                                    destination_dir,
                                    cpu = 4,
                                    wait = False)
                jobs += [job]
        if wait:
            cls.waiter(jobs)
        return jobs

    def _signaturizer_signaturize(self, cls, smiles, datasets=None, is_tmp=None, chunk_size=1000, wait=True, **kwargs):
        self.__log.info("Calculating sign for every molecule.")
        datasets = cls.get_datasets(datasets)
        jobs  = []
        for dataset in datasets:
            destination_dir = cls.get_destination_dir(dataset=dataset, is_tmp=is_tmp)
            self.destination_dirs[dataset] = destination_dir
            if os.path.exists(destination_dir):
                self.__log.debug("Signature %s file already exists: %s" % (dataset, destination_dir))
                continue
            else:
                self.__log.debug("Calculating sign for %s" % dataset)
                s3 = cls.sign_predict_paths[dataset]
                if not cls.hpc:
                    s3.predict_from_smiles(smiles, destination_dir)
                else:    
                    job = s3.func_hpc("predict_from_smiles", smiles,
                                      destination_dir, chunk_size, None, False,
                                      cpu=np.max([cls.n_jobs_hpc,8]),
                                      memory=16,
                                      wait=False)
                    jobs += [job]
        if wait:
            cls.waiter(jobs)
        return jobs


@logged
class RawBaseSignaturizer(HPCUtils):

    def __init__(self, root, is_classic, **kwargs):
        if is_classic:
            HPCUtils.__init__(self, **kwargs)
        self.root = root
        self.is_classic = is_classic
        self.destination_dirs = {}

    def get_datasets(self, datasets=None):
        if self.is_classic:
            return self.datasets
        else:
            return sorted(set(datasets))

    def get_destination_dir(self, dataset, **kwargs):
        if self.is_classic:
            dataset = self.dataset
        return os.path.join(self.root, dataset)


@logged
class RawFingerprinter(RawBaseSignaturizer):
    """Set up a Fingerprinter. This is usually used as a baseline featurizer to compare with CC signatures."""

    def __init__(self, **kwargs):
        # Inherit
        RawBaseSignaturizer.__init__(self, **kwargs)
        # Featurizer
        utils = SignUtils()
        self.featurizer_func, self.datasets, self.dataset = utils._fingerprinter__init__(self)

    def featurizer(self, smiles, destination_dir):
        utils = SignUtils()
        utils._featurizer(self, smiles, destination_dir)

    def signaturize(self, smiles, wait=True, **kwargs):
        utils = SignUtils()
        jobs = utils._featurizer_signaturize(self, smiles, wait, **kwargs)
        self.destination_dirs = utils.destinations_dirs
        return jobs


@logged
class RawSignaturizer(RawBaseSignaturizer):
    """Set up a Signaturizer"""

    def __init__(self,
                 datasets=None,
                 sign_predict_paths=None,
                 **kwargs):
        """Set up a Signaturizer
        
        Args:
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign predictor are
                used.
            sign_predict_paths(dict): pre-loaded predict_fn, keys are dataset
                codes, values are tuples of (sign, predict_fn)
        """
        # Inherit
        RawBaseSignaturizer.__init__(self, **kwargs)
        # Datasets and signature paths
        utils = SignUtils()
        self.datasets, self.sign_predict_paths = utils._signaturizer__init__(self, datasets, sign_predict_paths)

    # Calculate signatures
    def signaturize(self, smiles, datasets=None, chunk_size=1000, wait=True, **kwargs):
        utils = SignUtils()
        jobs = utils._signaturizer_signaturize(self, smiles, datasets=None, chunk_size=1000, wait=True, **kwargs)
        self.destination_dirs = utils.destination_dirs


class RawSignaturizerSetup(RawSignaturizer, RawFingerprinter):

    def __init__(self, **kwargs):
        if self.is_classic:
            RawFingerprinter.__init__(self, **kwargs)
        else:
            RawSignaturizer.__init__(self, **kwargs)
        
    def signaturize(self, smiles, **kwargs):
        if self.is_classic:
            return RawFingerprinter.signaturize(self, smiles, **kwargs)
        else:
            return RawSignaturizer.signaturize(self, smiles, **kwargs)


@logged
class BaseSignaturizer(TargetMateSetup, HPCUtils):

    #def __init__(self, master_signature_files=None, **kwargs):
    def __init__(self, **kwargs):
        """Initialize base signaturizer
        
        Args:
            master_signature_files(dict): Path to signature files that are not specific to the collection being analysed (default=None).
        """

        TargetMateSetup.__init__(self, **kwargs)
        if self.is_classic:
            HPCUtils.__init__(self, **kwargs)
        #self.master_signature_files=master_signature_files

    def get_datasets(self, datasets=None):
        if self.is_classic:
            return self.datasets
        else:
            if datasets is None:
                return self.datasets
            else:
                return sorted(set(self.datasets).intersection(datasets))

    def get_destination_dir(self, dataset, is_tmp=None):
        if is_tmp is None:
            is_tmp = self.is_tmp
        else:
            is_tmp = is_tmp
        if self.is_classic:
            if is_tmp:
                return os.path.join(self.signatures_tmp_path, self.dataset)
            else:
                return os.path.join(self.signatures_models_path, self.dataset)
        else:
            if is_tmp:
                return os.path.join(self.signatures_tmp_path, dataset)
            else:
                return os.path.join(self.signatures_models_path, dataset)

    def get_master_idx(self, dataset):
        master_idxs = {}
        master_dest = self.master_signature_files[dataset]
        with h5py.File(master_dest, "r") as hf:
            smiles = hf["smiles"][:]
        for i, smi in enumerate(smiles):
            master_idxs[i]
        return master_idx

    def master_mapping(self, smiles, master_idx):
        idx = []
        for smi in smiles:
            idx += [master_idx[smi]]
        return idx


@logged
class Fingerprinter(BaseSignaturizer):
    """Set up a Fingerprinter. This is usually used as a baseline featurizer to compare with CC signatures."""

    def __init__(self, **kwargs):
        # Inherit
        BaseSignaturizer.__init__(self, **kwargs)
        # Featurizer
        self.featurizer_func = chemistry.morgan_matrix
        self.datasets = ["FP"]
        self.dataset  = self.datasets[0]

    def featurizer(self, smiles, destination_dir):
        V = self.featurizer_func(smiles)
        with h5py.File(destination_dir, "w") as hf:
            hf.create_dataset("V", data = V.astype(np.int8))
            hf.create_dataset("keys", data = np.array(smiles, DataSignature.string_dtype()))

    def signaturize(self, smiles, is_tmp=None, wait=True, **kwargs):
        """Calculate fingerprints"""
        destination_dir = self.get_destination_dir(dataset = None, is_tmp = is_tmp)
        jobs = []
        if os.path.exists(destination_dir):
            self.__log.debug("Fingerprint file already exists: %s" %  destination_dir)
        else:
            self.__log.debug("Calculating fingerprint")
            if not self.hpc:
                self.featurizer(smiles, destination_dir)
            else:
                job = self.func_hpc("featurizer",
                                    smiles, 
                                    destination_dir,
                                    cpu = 4,
                                    wait = False)
                jobs += [job]
        if wait:
            self.waiter(jobs)
        return jobs

    # Signature readers
    def read_signatures_from_master(self, idxs=None, **kwargs):
        master_idx = self.get_master_idx(self.dataset)


    def read_signatures(self, idxs=None, is_tmp=None, sign_folder=None, **kwargs):
        """Read a signature from an HDF5 file"""
        if not sign_folder:
            destination_dir = self.get_destination_dir(dataset = None, is_tmp=is_tmp)
        else:
            destination_dir = os.path.join(sign_folder, self.dataset)
        with h5py.File(destination_dir, "r") as hf:
            if idxs is None:
                V = hf["V"][:]
            else:
                V = hf["V"][:][idxs]
        return V


@logged
class Signaturizer(BaseSignaturizer):
    """Set up a Signaturizer"""

    def __init__(self,
                 datasets=None,
                 sign_predict_paths=None,
                 **kwargs):
        """Set up a Signaturizer
        
        Args:
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign predictor are
                used.
            sign_predict_paths(dict): pre-loaded predict_fn, keys are dataset
                codes, values are tuples of (sign, predict_fn)
        """
        # Inherit
        BaseSignaturizer.__init__(self, **kwargs)
        # Datasets
        if not datasets:
            # self.datasets = list(self.cc.datasets)
            self.datasets = ["%s%s.001" % (x, y)
                             for x in "ABCDE" for y in "12345"]
        else:
            self.datasets = datasets
        # preloaded neural networks
        if not sign_predict_paths:
            self.sign_predict_paths = {}
            for ds in self.datasets:
                self.__log.debug("Loading signature predictor for %s" % ds)
                s3 = self.cc.get_signature("sign3", "full", ds)
                self.sign_predict_paths[ds] = s3
        else:
            self.sign_predict_paths = sign_predict_paths

    # Calculate signatures
    def signaturize(self, smiles, datasets=None, is_tmp=None, chunk_size=1000, wait=True, **kwargs):
        self.__log.info("Calculating sign for every molecule.")
        datasets = self.get_datasets(datasets)
        jobs  = []
        for dataset in datasets:
            destination_dir = self.get_destination_dir(dataset, is_tmp)
            if os.path.exists(destination_dir):
                self.__log.debug("Signature %s file already exists: %s" % (dataset, destination_dir))
                continue
            else:
                self.__log.debug("Calculating sign for %s" % dataset)
                s3 = self.sign_predict_paths[dataset]
                if not self.hpc:
                    s3.predict_from_smiles(smiles, destination_dir)
                else:    
                    job = s3.func_hpc("predict_from_smiles", smiles,
                                      destination_dir, chunk_size, None, False,
                                      cpu=np.max([self.n_jobs_hpc,8]),
                                      memory=16,
                                      wait=False)
                    jobs += [job]
        if wait:
            self.waiter(jobs)
        return jobs
     
    # Signature readers
    def read_signature(self, dataset, idxs=None, is_tmp=None, sign_folder=None):
        """Read a signature from an HDF5 file"""
        if not sign_folder:
            destination_dir = self.get_destination_dir(dataset, is_tmp=is_tmp)
        else:
            destination_dir = os.path.join(sign_folder, dataset)
        with h5py.File(destination_dir, "r") as hf:
            if idxs is None:
                V = hf["V"][:]
            else:
                V = hf["V"][:][idxs]
        return V

    def read_signatures_ensemble(self, datasets=None, **kwargs):
        """Return signatures as an ensemble"""
        if not datasets: datasets = self.datasets
        for ds in datasets:
            yield self.read_signature(ds, **kwargs)

    def read_signatures_stacked(self, datasets=None, **kwargs):
        """Return signatures in a stacked form"""
        if not datasets: datasets = self.datasets
        if type(datasets) == str: datasets = [datasets]
        V = []
        for ds in datasets:
            V += [self.read_signature(ds, **kwargs)]
        return np.hstack(V)

    def read_signatures(self, is_ensemble, **kwargs):
        if is_ensemble:
            return self.read_signatures_ensemble(**kwargs)
        else:
            return self.read_signatures_stacked(**kwargs)


class SignaturizerSetup(Signaturizer, Fingerprinter):

    def __init__(self, **kwargs):
        if self.is_classic:
            Fingerprinter.__init__(self, **kwargs)
        else:
            Signaturizer.__init__(self, **kwargs)
        
    def signaturize(self, smiles, **kwargs):
        if self.is_classic:
            return Fingerprinter.signaturize(self, smiles, **kwargs)
        else:
            return Signaturizer.signaturize(self, smiles, **kwargs)

    def read_signatures(self, **kwargs):
        if self.is_classic:
            return Fingerprinter.read_signatures(self, **kwargs)
        else:
            return Signaturizer.read_signatures(self, **kwargs)

