import os

import h5py
import numpy as np
from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import logged

from signaturizer import Signaturizer as SignaturizerExternal
from .tmsetup import TargetMateSetup
from .utils import HPCUtils
from .utils import chemistry

MAXQUEUE = 15

@logged
class BaseSignaturizer(TargetMateSetup, HPCUtils):

    def __init__(self, master_sign_paths=None, cctype="sign3", **kwargs):
        """Initialize base signaturizer

        Args:
            master_sign_paths(dict): Path to signature files that are not specific to the collection being analysed (default=None).
            cctype(str): CC signature type to be used (sign0, sign1, sign2, sign3) (default='sign3').
        """
        TargetMateSetup.__init__(self, **kwargs)
        if self.is_classic:
            HPCUtils.__init__(self, **kwargs)

        self.master_sign_paths = master_sign_paths
        # if cctype != "sign3" and cctype != "sign4":
        #     raise Exception("cctype can only be 'sign3' or 'sign4'")
        self.cctype = cctype

    def master_key_type(self):
        """Check master key types"""
        from chemicalchecker.util.keytype.detect import KeyTypeDetector
        if self.master_sign_paths is None:
            raise Exception("master_sign_paths is None")
        ktypes = set()
        for ds, path in self.master_sign_paths.items():
            with h5py.File(self.master_sign_paths[ds], "r") as hf:
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
        idxs0 = np.array(idxs0).astype(int)
        idxs1 = np.array(idxs1).astype(int)
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
        idxs = np.array([iks_dict[k] for k in keys if k in iks_dict]).astype(int)
        return V, idxs


    def _get_predicted_signatures(self, dataset, inchikeys): # Added by Paula
        iks_dict = dict((k,i) for i,k in enumerate(inchikeys))
        if dataset is None:
            dataset = self.dataset
        self.__log.info("Reading signature of type %s" % self.cctype)

        sign = self.cc.signature(dataset, self.cctype)
        self.__log.info("...data path: %s" % sign.data_path)

        keys, V = sign.get_vectors_lite(inchikeys)
        self.__log.info("Signature read")

        idxs = np.array([iks_dict[k] for k in keys if k in iks_dict]).astype(int)

        return V, idxs

    def _read_signatures_from_master(self, dataset, keys):
        if dataset is None:
            dataset = self.dataset

        master_idxs = self.get_master_idxs(dataset)
        idxs_or, idxs_mp = self.master_mapping(keys, master_idxs)
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

    def _read_signatures_by_idxs_from_local(self, dataset, smiles, idxs, inchikeys, sign_folder, is_tmp):
        """Read a signature from an HDF5 file. This must be specific to the collection being analyzed."""
        if not sign_folder:
            destination_dir = self.get_destination_dir(dataset, is_tmp=is_tmp)
        else:
            destination_dir = os.path.join(sign_folder, dataset)

        if self.use_cc or self.is_classic:
            name = "V"
        else:
            name = "signature"
        with h5py.File(destination_dir, "r") as hf:
            if idxs is None:
                V = hf[name][:]
                idxs = np.array([i for i in range(V.shape[0])]).astype(int)
            else:
                V = hf[name][:][idxs]
                if name == "signature":
                    failed = hf["failed"][:][idxs]
                    V = V[~failed]
                    idxs = np.array([i for i, f in enumerate(~failed) if f]).astype(int)
                else:
                    idxs = np.array([i for i in range(V.shape[0])]).astype(int)
        return V, idxs

    def read_signatures(self, dataset, smiles, inchikeys, sign_folder, is_tmp, idxs=None):
        if self.use_cc:
            self.__log.info("Reading signatures from the Chemical Checker (inchikeys are used)")
            if inchikeys is None:
                raise Exception("inchikeys is None, cannot use_cc")
            return self._read_signatures_by_inchikey_from_cc(dataset=dataset, inchikeys=inchikeys)
        else:

            if self.master_sign_paths is None:
                self.__log.info("Reading signatures from a task-specific file")
                return self._read_signatures_by_idxs_from_local(dataset=dataset, smiles= smiles, inchikeys=inchikeys, idxs=idxs, sign_folder=sign_folder, is_tmp=is_tmp)

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
            hf.create_dataset("V", data = V.astype(int))
            hf.create_dataset("keys", data = np.array(smiles, DataSignature.string_dtype()))

    def _signaturize_fingerprinter(self, smiles, is_tmp=None, wait=True, **kwargs):
        """Calculate fingerprints"""
        if self.use_cc:
            self.__log.info("use_cc was set to True, i.e. signatures are already calculated!")
            return []

        if self.master_sign_paths is not None:
            self.__log.info("Master signature paths exists")
            return []

        destination_dir = self.get_destination_dir(dataset = self.dataset, is_tmp = is_tmp)
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
        return self._read_signatures_by_idxs_from_local(dataset=self.dataset, smiles = smiles, idxs=idxs, inchikeys = inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)

    def signaturize(self, **kwargs):
        return self._signaturize_fingerprinter(**kwargs)


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
        if self.prestacked_dataset is None:
            prestacked_friendly = False
            prestacked_mask = None
            return prestacked_friendly, prestacked_mask
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
    def _predict_from_molecule(self, dataset, smiles, destination_dir, moleculetype):
        s3 = SignaturizerExternal(dataset.split(".")[0])
        s3.predict(smiles.tolist(), destination_dir, keytype=moleculetype)

    def _signaturize_signaturizer(self, smiles, datasets=None, is_tmp=None, wait=True, moleculetype = 'SMILES',
                                  **kwargs):

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
                if not self.hpc:
                    self._predict_from_molecule(dataset, smiles, destination_dir, moleculetype)
                else:
                    job = self.func_hpc("_predict_from_molecule", dataset, smiles,
                                      destination_dir, moleculetype,
                                      cpu=np.max([self.n_jobs_hpc, 8]),
                                      memory=16,
                                      wait=False,
                                      job_base_path = self.tmp_path,
                                      delete_job_path=True)
                    jobs += [job]
                    if len(jobs) > MAXQUEUE:
                        self.waiter(jobs)
                        jobs = []
        if wait:
            self.waiter(jobs)
        return jobs

    def signaturize(self, **kwargs):
        return self._signaturize_signaturizer(**kwargs)

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
        idxs__ = None
        for ds in datasets:
            v, idxs_ = BaseSignaturizer.read_signatures(self, dataset=ds, smiles=smiles, inchikeys=inchikeys, idxs=idxs, is_tmp=is_tmp, sign_folder=sign_folder)
            V += [v]
            if idxs__ is None:
                idxs__ = idxs_
            if np.any(idxs__ != idxs_):
                raise Exception("When stacking signatures exactly the same keys need to be available for all molecules")
        return np.hstack(V), idxs_

    def read_signatures_prestacked(self, mask, datasets, smiles, inchikeys, idxs, is_tmp, sign_folder):
        """Return signatures in a stacked form from an already prestacked file"""
        datasets = self._dataseter(datasets)
        V, idxs = BaseSignaturizer.read_signatures(self, dataset=self.prestacked_dataset, smiles=smiles, inchikeys=inchikeys, idxs=idxs, is_tmp=is_tmp, sign_folder=sign_folder)
        if mask is None:
            return V, idxs
        else:
            return V[:,mask], idxs

    def read_signatures(self, is_ensemble, datasets, idxs, smiles, inchikeys, is_tmp, sign_folder=None): # Changed sign folder to None
        if is_ensemble:
            return self.read_signatures_ensemble(datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)
        else:
            prestack_friendly, prestack_mask = self._check_prestack_friendly(datasets)
            if prestack_friendly:
                return self.read_signatures_prestacked(mask=prestack_mask, datasets=self.prestacked_dataset, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)
            else:
                return self.read_signatures_stacked(datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)


class SignaturizerSetup(Signaturizer, Fingerprinter):
    """Set up a signaturizer"""

    def __init__(self, **kwargs):
        if self.is_classic and not self.use_cc:
            Fingerprinter.__init__(self, **kwargs)
        else:
            Signaturizer.__init__(self, **kwargs)

    def signaturize(self, smiles, **kwargs):
        if self.is_classic and not self.use_cc:
            return Fingerprinter.signaturize(self, smiles=smiles, **kwargs)
        else:
            if self.use_stacked_signature:
                return Signaturizer.stacker(self, smiles=smiles, **kwargs)
            else:
                return Signaturizer.signaturize(self, smiles=smiles, **kwargs)

    def _read_signatures_(self, datasets, idxs, smiles, inchikeys, is_tmp, sign_folder):
        if self.is_classic and not self.use_cc:
            return Fingerprinter.read_signatures(self, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)
        else:
            return Signaturizer.read_signatures(self, is_ensemble=self.is_ensemble, datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)

    def read_signatures(self, datasets=None, idxs=None, smiles=None, inchikeys=None, is_tmp=None, sign_folder=None):
        return self._read_signatures_(datasets=datasets, idxs=idxs, smiles=smiles, inchikeys=inchikeys, is_tmp=is_tmp, sign_folder=sign_folder)


