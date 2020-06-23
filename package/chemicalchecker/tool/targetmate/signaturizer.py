import h5py
import os
import numpy as np

from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import logged
from chemicalchecker.core import ChemicalChecker

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
        datasets = ["FP.000"]
        dataset  = datasets[0]
        return featurizer_func, datasets, dataset
    
    def _signaturizer__init__(self, cls, datasets, sign_predict_paths):
        if datasets is None:
            datasets = [ds for ds in cls.cc.datasets_exemplary()]
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
                                    cpu=4,
                                    wait=False)
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
                                      cpu=np.max([cls.n_jobs_hpc, 8]),
                                      memory=16,
                                      wait=False,
                                      delete_job_path=True)
                    jobs += [job]
        if wait:
            cls.waiter(jobs)
        return jobs

    def _fingerprinter_get_destination_dirs(self, cls, is_tmp=None, **kwargs):
        if not self.destination_dirs:
            self.__log.info("Getting destination dirs")
            destination_dir = cls.get_destination_dir(dataset=None, is_tmp=is_tmp)
            self.destination_dirs[cls.dataset] = destination_dir
        return self.destination_dirs

    def _signaturizer_get_destination_dirs(self, cls, datasets=None, is_tmp=None, **kwargs):
        if not self.destination_dirs:
            self.__log.info("Getting destination dirs")
            datasets = cls.get_datasets(datasets)
            for dataset in datasets:
                destination_dir = cls.get_destination_dir(dataset=dataset, is_tmp=is_tmp)
                self.destination_dirs[dataset] = destination_dir
        return self.destination_dirs


# Raw signaturizer
#  Used to do signaturizers *outside* the TargetMate class
#  The following is only used to signaturize. Classes are not prepared to read signatures.

@logged
class RawBaseSignaturizer(HPCUtils):

    def __init__(self, root, is_classic, cc_root=None, hpc=False, n_jobs_hpc=8, **kwargs):
        if is_classic:
            HPCUtils.__init__(self, **kwargs)
        self.root = root
        self.is_classic = is_classic
        self.destination_dirs = {}
        self.cc = ChemicalChecker(cc_root)
        self.hpc = hpc
        self.n_jobs_hpc = n_jobs_hpc

    def get_datasets(self, datasets=None):
        if self.is_classic:
            return self.datasets
        else:
            if datasets is None:
                datasets = self.datasets
            else:
                datasets = set(datasets).intersection(self.datasets)
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
        jobs = utils._fingerprinter_signaturize(self, smiles, wait, **kwargs)
        self.destination_dirs = utils.destination_dirs
        return jobs

    def get_destination_dirs(self, **kwargs):
        utils = SignUtils()
        return utils._fingerprinter_get_destination_dirs(self, **kwargs)


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
        jobs = utils._signaturizer_signaturize(self, smiles, datasets=datasets, chunk_size=chunk_size, wait=wait, **kwargs)
        self.destination_dirs = utils.destination_dirs
        return jobs

    def get_destination_dirs(self, **kwargs):
        utils = SignUtils()
        return utils._signaturizer_get_destination_dirs(self, **kwargs)


class RawSignaturizerSetup(RawSignaturizer, RawFingerprinter):

    def __init__(self, **kwargs):
        self.is_classic = kwargs.get("is_classic")
        self.hpc = kwargs.get("hpc")
        if self.is_classic:
            RawFingerprinter.__init__(self, **kwargs)
        else:
            RawSignaturizer.__init__(self, **kwargs)
        
    def signaturize(self, smiles, **kwargs):
        if self.is_classic:
            return RawFingerprinter.signaturize(self, smiles, **kwargs)
        else:
            return RawSignaturizer.signaturize(self, smiles, **kwargs)

    def get_destination_dirs(self, **kwargs):
        if self.is_classic:
            return RawFingerprinter.get_destination_dirs(self, **kwargs)
        else:
            return RawSignaturizer.get_destination_dirs(self, **kwargs)


# Signaturizer
#  Used to deal with signatures *inside* the TargetMate class
#  The following is prepared to write *and* read signatures.

@logged
class BaseSignaturizer(TargetMateSetup, HPCUtils):

    def __init__(self, use_cc, master_sign_paths=None, cctype="sign3", **kwargs):
        """Initialize base signaturizer
        
        Args:
            use_cc(bool): Use pre-computed CC signatures.
            master_sign_paths(dict): Path to signature files that are not specific to the collection being analysed (default=None).
            cctype(str): CC signature type to be used (sign0, sign1, sign2, sign3) (default='sign3').
        """
        TargetMateSetup.__init__(self, **kwargs)
        if self.is_classic:
            HPCUtils.__init__(self, **kwargs)
        self.use_cc = use_cc
        self.master_sign_paths = master_sign_paths
        if cctype != "sign3" and cctype != "sign4":
            raise Exception("cctype can only be 'sign3' or 'sign4'")
        self.cctype = cctype

    def master_key_type(self):
        """Check master key types"""
        from chemicalchecker.util.keytype.detect import KeyTypeDetector
        if self.master_sign_paths is None:
            raise Exception("master_sign_paths is None")
        ktypes = set()
        for ds, path in self.master_sign_paths.keys():
            with h5py.File(self.master_sign_paths[dataset], "r") as hf:
                keys = hf["keys"][:]
            kd = KeyTypeDetector(keys)
            ktype = kd.detect()
            ktypes.update([ktype])
        if len(ktypes) > 1:
            raise Exception("More than one key type detected")
        return ktype

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

    def get_master_idxs(self, dataset):
        master_idxs = {}
        master_dest = self.master_sign_paths[dataset]
        with h5py.File(master_dest, "r") as hf:
            keys = hf["keys"][:]
        for i, key in enumerate(keys):
            master_idxs[key] = i
        return master_idxs

    def master_mapping(self, keys, master_idxs):
        idxs0 = []
        idxs1 = []
        for i, key in enumerate(keys):
            if key not in master_idxs: continue
            idxs0 += [i]
            idxs1 += [master_idxs[key]]
        idxs0 = np.array(idxs0).astype(np.int)
        idxs1 = np.array(idxs1).astype(np.int)
        return idxs0, idxs1

    # Signature readers
    def _read_signatures_by_inchikey_from_cc(self, dataset, inchikeys):
        """Read signatures from CC. InChIKeys are used."""
        iks_dict = dict((k,i) for i,k in enumerate(inchikeys))
        if dataset is None:
            dataset = self.dataset
        self.__log.info("Reading signature of type %s" % self.cctype)
        sign = self.cc.signature(dataset, self.cctype)
        self.__log.info("...data path: %s" % sign.data_path)
        keys, V = sign.get_vectors_lite(inchikeys)
        self.__log.info("Signature read")
        idxs = np.array([iks_dict[k] for k in keys if k in iks_dict]).astype(np.int)
        return V, idxs

    def _read_signatures_from_master(self, dataset, keys):
        if dataset is None:
            dataset = self.dataset
        master_idxs = self.get_master_idxs(dataset)
        idxs_or, idxs_mp = self.master_mapping(keys)
        destination_dir = self.master_sign_paths[dataset]
        with h5py.File(destination_dir, "r") as hf:
            V = hf["V"][:][idxs_mp]
            idxs = idxs_or
        return V, idxs

    def _read_signatures_by_inchikey_from_master(self, dataset, inchikeys):
        """Read signatures from a master signature file. InChIKeys are used"""
        return self._read_signatures_from_master(dataset, inchikeys)

    def _read_signatures_by_smiles_from_master(self, dataset, smiles):
        """Read signatures from a master signature file. SMILES are used"""
        return self._read_signatures_from_master(dataset, smiles)

    def _read_signatures_by_idxs_from_local(self, dataset, idxs, sign_folder, is_tmp):
        """Read a signature from an HDF5 file. This must be specific to the collection being analyzed."""
        if not sign_folder:
            destination_dir = self.get_destination_dir(dataset, is_tmp=is_tmp)
        else:
            destination_dir = os.path.join(sign_folder, dataset)
        with h5py.File(destination_dir, "r") as hf:
            if idxs is None:
                V = hf["V"][:]
                idxs = np.array([i for i in range(V.shape[0])]).astype(np.int)
            else:
                V = hf["V"][:][idxs]
        return V, idxs

    def read_signatures(self, dataset, idxs, smiles, inchikeys, sign_folder, is_tmp):
        if self.use_cc:
            self.__log.info("Reading signatures from the Chemical Checker (inchikeys are used)")
            if inchikeys is None:
                raise Exception("inchikeys is None, cannot use_cc")
            return self._read_signatures_by_inchikey_from_cc(dataset=dataset, inchikeys=inchikeys)
        else:
            if self.master_sign_paths is None:
                self.__log.info("Reading signatures from a task-specific file")
                return self._read_signatures_by_idxs_from_local(dataset=dataset, idxs=idxs, sign_folder=sign_folder, is_tmp=is_tmp)
            else:
                self.__log.info("Reading signatures from a master signatures file")
                key_type = self.master_key_type()
                if key_type == "inchikey":
                    if inchikeys is None:
                        raise Exception("inchikeys is None, cannot use master signatures")
                    return self._read_signatures_by_inchikey_from_master(dataset=dataset, inchikeys=inchikeys)
                elif key_type == "smiles":
                    if smiles is None:
                        raise Exception("smiles is None, cannot use master signatures")
                    return self._read_signatures_by_smiles_from_master(dataset=dataset, smiles=smiles)


@logged
class Fingerprinter(BaseSignaturizer):
    """Set up a Fingerprinter. This is usually used as a baseline featurizer to compare with CC signatures."""

    def __init__(self, **kwargs):
        # Inherit
        BaseSignaturizer.__init__(self, **kwargs)
        # Featurizer
        self.featurizer_func = chemistry.morgan_matrix
        if not self.use_cc:
            self.datasets = ["FP.000"]
            self.dataset  = self.datasets[0]
        else:
            self.datasets = [self.classic_dataset]
            self.dataset  = self.datasets[0]
            self.cctype   = self.classic_cctype

    def featurizer(self, smiles, destination_dir):
        V = self.featurizer_func(smiles)
        with h5py.File(destination_dir, "w") as hf:
            hf.create_dataset("V", data = V.astype(np.int8))
            hf.create_dataset("keys", data = np.array(smiles, DataSignature.string_dtype()))

    def signaturize(self, smiles, is_tmp=None, wait=True, **kwargs):
        """Calculate fingerprints"""
        if self.use_cc:
            self.__log.info("use_cc was set to True, i.e. signatures are already calculated!")
            return []
        if self.master_sign_paths is not None:
            self.__log.info("Master signature paths exists")
            return []
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

    def read_signatures(self, idxs, smiles, inchikeys, is_tmp, sign_folder):
        """Read signatures"""
        return BaseSignaturizer.read_signatures(self, dataset=self.dataset, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)


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
        self.sign_dim = 128
        # Datasets
        if not datasets:
            self.datasets = []
            for ds in self.cc.datasets_exemplary():
                self.datasets += [ds]
        else:
            self.datasets = datasets
        if sorted(self.datasets) != list(self.datasets):
            raise Exception("Datasets must be sorted!")
        # preloaded neural networks
        if not sign_predict_paths:
            self.sign_predict_paths = {}
            for ds in self.datasets:
                self.__log.debug("Loading signature predictor for %s" % ds)
                s3 = self.cc.get_signature(self.cctype, "full", ds)
                self.sign_predict_paths[ds] = s3
        else:
            self.sign_predict_paths = sign_predict_paths

    def _dataseter(self, datasets):
        if not datasets: datasets = self.datasets
        if type(datasets) == str: datasets = [datasets]
        if sorted(datasets) != list(datasets):
            raise Exception("Datasets not sorted")
        return datasets        

    def _check_prestack_friendly(self, datasets):
        datasets = self._dataseter(datasets)
        s3 = self.cc.signature(self.prestacked_dataset, "sign3")
        with h5py.File(s3.data_path, "r") as hf:
            prestacked_datasets = list(hf["datasets"][:])
        # Check datasets of pre-stacked signature 
        if len(set(datasets).difference(prestacked_datasets)) == 0:
            prestacked_friendly = True
            if list(datasets) == list(prestacked_datasets):
                prestacked_mask = None
            else:
                datasets_set    = set(datasets)
                prestacked_mask = []
                for ds in prestacked_datasets:
                    if ds in datasets_set:
                        prestacked_mask += [True]*self.sign_dim
                    else:
                        prestacked_mask += [False]*self.sign_dim
                prestacked_mask = np.array(prestacked_mask)
        else:
            prestacked_friendly = False
            prestacked_mask   = None
        return prestacked_friendly, prestacked_mask

    # Calculate signatures
    def signaturize(self, smiles, datasets=None, is_tmp=None, chunk_size=1000, wait=True, **kwargs):
        if self.use_cc:
            self.__log.info("use_cc was set to True, i.e. signatures are already calculated!")
            return []
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
                                      wait=False,
                                      delete_job_path=True)
                    jobs += [job]
        if wait:
            self.waiter(jobs)
        return jobs

    # Read signatures
    def read_signatures_ensemble(self, datasets, smiles, inchikeys, idxs, is_tmp, sign_folder):
        """Return signatures as an ensemble"""
        datasets = self._dataseter(datasets)
        for ds in datasets:
            yield BaseSignaturizer.read_signatures(self, dataset=ds, smiles=smiles, inchikeys=inchikeys, idxs=idxs, is_tmp=is_tmp, sign_folder=sign_folder)

    def read_signatures_stacked(self, datasets, smiles, inchikeys, idxs, is_tmp, sign_folder):
        """Return signatures in a stacked form"""
        datasets = self._dataseter(datasets)
        V = []
        idxs = None
        for ds in datasets:
            v, idxs_ = BaseSignaturizer.read_signatures(self, dataset=ds, smiles=smiles, inchikeys=inchikeys, idxs=idxs, is_tmp=is_tmp, sign_folder=sign_folder)
            V += [v]
            if idxs is None:
                idxs = idxs_
            if np.any(idxs_ != idxs):
                raise Exception("When stacking signatures exactly the same keys need to be available for all molecules")
        return np.hstack(V), idxs

    def read_signatures_prestacked(self, mask, datasets, smiles, inchikeys, idxs, is_tmp, sign_folder):
        """Return signatures in a stacke form from an already prestacked file"""
        datasets = self._dataseter(datasets)
        V, idxs = BaseSignaturizer.read_signatures(self, dataset=self.prestacked_dataset, smiles=smiles, inchikeys=inchikeys, idxs=idxs, is_tmp=is_tmp, sign_folder=sign_folder)
        if mask is None:
            return V, idxs
        else:
            return V[:,mask], idxs

    def read_signatures(self, is_ensemble, datasets, idxs, smiles, inchikeys, is_tmp, sign_folder):
        if is_ensemble:
            return self.read_signatures_ensemble(datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)
        else:
            prestack_friendly, prestack_mask = self._check_prestack_friendly(datasets)
            if prestack_friendly:
                return self.read_signatures_prestacked(mask=prestack_mask, datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)
            else:
                return self.read_signatures_stacked(datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)


class SignaturizerSetup(Signaturizer, Fingerprinter):
    """Set up a signaturizer"""

    def __init__(self, **kwargs):
        if self.is_classic:
            Fingerprinter.__init__(self, **kwargs)
        else:
            Signaturizer.__init__(self, **kwargs)
        
    def signaturize(self, smiles, **kwargs):
        if self.is_classic:
            return Fingerprinter.signaturize(self, smiles=smiles, **kwargs)
        else:
            return Signaturizer.signaturize(self, smiles=smiles, **kwargs)

    def _read_signatures_(self, datasets, idxs, smiles, inchikeys, is_tmp, sign_folder):
        if self.is_classic:
            return Fingerprinter.read_signatures(self, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)
        else:
            return Signaturizer.read_signatures(self, is_ensemble=self.is_ensemble, datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)

    def read_signatures(self, datasets=None, idxs=None, smiles=None, inchikeys=None, is_tmp=None, sign_folder=None):
        return self._read_signatures_(datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)


@logged
class Sign3SignatureStacker(object):

    def __init__(self, cc, output_dataset="Z0.001", datasets=None, chunk_size=10000):
        z3 = cc.signature(output_dataset, "sign3")
        self.destination_dir = z3.data_path
        if os.path.exists(self.destination_dir):
            os.remove(self.destination_dir)
        self.cc = cc
        if not datasets:
            self.datasets = []
            for ds in self.cc.datasets_exemplary():
                self.datasets += [ds]
        else:
            self.datasets = datasets
        self.chunk_size = chunk_size
        s3 = self.cc.signature("A1.001", "sign3")
        self.shape = s3.shape
        self.keys  = s3.keys
        self.data_paths = []
        for ds in self.datasets:
            s3 = self.cc.signature(ds, "sign3")
            self.data_paths += [s3.data_path]
        
    def chunker(self, n):
        size = self.chunk_size
        for i in range(0, n, size):
            yield slice(i, i+size)

    def stack(self):
        from tqdm import tqdm
        with h5py.File(self.destination_dir, "a") as hf:
            hf.create_dataset("keys", data=np.array(self.keys, DataSignature.string_dtype()))
            hf.create_dataset("datasets", data=np.array(self.datasets, DataSignature.string_dtype()))
            for chunk in tqdm(self.chunker(self.shape[0])):
                V = None
                for dp in self.data_paths:
                    with h5py.File(dp, "r") as sf:
                        if V is None:
                            V = sf["V"][chunk]
                        else:
                            V = np.hstack((V, sf["V"][chunk]))
                if "V" not in hf.keys():
                    hf.create_dataset("V", data=V, maxshape=(self.shape[0], V.shape[1]))
                else:
                    hf["V"].resize((hf["V"].shape[0]+V.shape[0]), axis=0)
                    hf["V"][-V.shape[0]:] = V



# Signatures in a raw HDF5 format
#  Utils functions to assemble signatures
#  The following is currently not so much used (it was developed to tackle the DREAM challenge)

@logged
class RawSignatureStacker(object):

    def __init__(self, filename, datasets, cctype="sign3", cc=None):
        self.filename = os.path.abspath(filename)
        self.datasets = datasets
        self.sign_type = cctype
        if cc is None:
            self.cc = ChemicalChecker()
        else:
            self.cc = cc

    def stack(self):
        self.__log.debug("Stacked signature will be stored as a simple HDF5 file")
        keys = None
        self.__log.debug("Getting shared keys")
        for ds in self.datasets:
            sign = self.cc.get_signature(self.sign_type, "full", ds)
            if keys is None:
                keys = set(sign.keys)
            else:
                keys = keys.intersection(sign.keys)
        keys = sorted(keys)
        self.__log.debug("Provisionally, I control for a universe of iks")
        cc = ChemicalChecker()
        sign = cc.get_signature("sign3", "full", "A1.001")
        iks_universe = set(sign.keys)
        keys = [k for k in keys if k in iks_universe or k[0]=="_"] # The "_" is relevant to the DREAM challenge
        self.__log.debug("%d keys in common" % len(keys))
        X = None
        confidence = np.zeros((len(keys), len(self.datasets)))
        for j, ds in enumerate(self.datasets):
            sign = self.cc.get_signature(self.sign_type, "full", ds)
            V = sign.get_vectors(keys)[1]
            if X is None:
                X = V
                dtype = V.dtype
            else:
                X = np.hstack((X, V))
            if self.sign_type == "sign3":
                my_keys = set(keys)
                conf_ = sign.get_h5_dataset("confidence")
                keys_ = sign.keys
                conf  = []
                for i, k in enumerate(keys_):
                    if k in my_keys:
                        conf += [conf_[i]]
                confidence[:,j] = np.array(conf)
        self.__log.debug("Saving: %s" % self.filename)
        with h5py.File(self.filename, "w") as hf:
            hf.create_dataset("V", data=X.astype(dtype))
            hf.create_dataset("keys", data=np.array(keys, DataSignature.string_dtype()))
            hf.create_dataset("datasets", data=np.array(self.datasets, DataSignature.string_dtype()))
            hf.create_dataset("confidences", data=confidence)

@logged
class FileByFileSignatureStacker(object):

    def __init__(self, filename, source_folder, files_format="npy"):
        self.filename = os.path.abspath(filename)
        self.files_format = files_format
        self.source_folder = os.path.abspath(source_folder)

    def stack(self):
        self.__log.debug("Reading files one by one from: %s" % self.source_folder)
        V    = []
        keys = []
        dtype = None
        from tqdm import tqdm
        for f in tqdm(os.listdir(self.source_folder)):
            f = os.path.join(self.source_folder, f)
            if self.files_format == "npy":
                v = np.load(f)
                if dtype is None:
                    dtype = v.dtype
                V += [v]
                keys += [f.split("/")[-1].split(".npy")[0]]
        V    = np.array(V)
        keys = np.array(keys)
        idxs = np.argsort(keys)
        keys = keys[idxs]
        V    = V[idxs]
        self.__log.debug("Saving: %s" % self.filename)
        with h5py.File(self.filename, "w") as hf:
            hf.create_dataset("V", data=V.astype(dtype))
            hf.create_dataset("keys", data=np.array(keys, DataSignature.string_dtype()))

