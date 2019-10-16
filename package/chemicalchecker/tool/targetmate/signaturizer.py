import h5py
from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import logged
from .utils import chemistry
from .tmsetup import TargetMateSetup

@logged
class Fingerprinter(TargetMateSetup):
    """Set up a Fingerprinter. This is usually used as a baseline featurizer to compare with CC signatures."""

    def __init__(self, **kwargs):
        # Inherit TargetMateSetup
        TargetMateSetup.__init__(**kwargs)
        # Featurizer
        self.featurizer = chemistry.morgan_matrix
        self.dataset = "FP"

    def featurize(self, smiles, **kwargs):
        """Calculate fingerprints"""
        if self.is_tmp:
            destination_dir = os.path.join(self.tmp_path, self.dataset)
        else:
            destination_dir = os.path.join(self.models_path, self.dataset)
        V = self.featurizer(smiles)
        with h5py.File(destination_dir, "wb") as hf:
            hf.create_dataset("V", data = V.astype(np.int8))
            hf.create_dataset("keys", data = np.array(smiles, DataSignature.string_dtype()))
    
    def read_fingerprint(self, idxs=None, fp_file=None):
        """Read a signature from an HDF5 file"""
        # Identify HDF5 file
        if not fp_file:
            if self.is_tmp:
                h5file = os.path.join(self.tmp_path, self.dataset)
            else:
                h5file = os.path.join(self.models_path, self.dataset)
        else:
            h5file = os.path.join(fp_file)
        # Read the file
        with h5py.File(h5file, "r") as hf:
            if idxs is None:
                V = hf["V"][:]
            else:
                V = hf["V"][:][idxs]
        return V


@logged
class Signaturizer(TargetMateSetup):
    """Set up a Signaturizer"""

    def __init__(self,
                 datasets=None,
                 sign_predict_fn=None,
                 **kwargs):
        """Set up a Signaturizer
        
        Args:
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign predictor are
                used.
            sign_predict_fn(dict): pre-loaded predict_fn, keys are dataset
                codes, values are tuples of (sign, predict_fn)
        """
        # Inherit TargetMateSetup
        TargetMateSetup.__init__(self, **kwargs)
        # Datasets
        if not datasets:
            # self.datasets = list(self.cc.datasets)
            self.datasets = ["%s%s.001" % (x, y)
                             for x in "ABCDE" for y in "12345"]
        else:
            self.datasets = datasets
        # preloaded neural networks
        if sign_predict_fn is None:
            self.sign_predict_fn = dict()
            for ds in self.datasets:
                self.__log.debug("Loading sign predictor for %s" % ds)
                s3 = self.cc.get_signature("sign3", "full", ds)
                self.sign_predict_fn[ds] = (s3, s3.get_predict_fn())
        else:
            self.sign_predict_fn = sign_predict_fn

    def get_destination_dir(self, dataset):
        if self.is_tmp:
            return os.path.join(self.signatures_tmp_path, dataset)
        else:
            return os.path.join(self.signatures_models_path, dataset)

    # Calculate signatures
    def signaturize(self, smiles, chunk_size=1000):
        self.__log.info("Calculating sign for every molecule.")
        jobs  = []
        for dataset in self.datasets:
            destination_dir = self.get_destination_dir(dataset)
            if os.path.exists(destination_dir):
                continue
            else:
                self.__log.debug("Calculating sign for %s" % dataset)
                s3, predict_fn = self.sign_predict_fn[dataset]
                if not self.hpc:
                    s3.predict_from_smiles(smiles, destination_dir,
                                           predict_fn=predict_fn,
                                           use_novelty_model=False)
                else:    
                    job = s3.func_hpc("predict_from_smiles", smiles,
                                      destination_dir, chunk_size, None, False,
                                      cpu=self.n_jobs_hpc, wait=False)
                    jobs += [job]
        self.waiter(jobs)
     
    # Signature readers
    def read_signature(self, dataset, idxs=None, sign_folder=None):
        """Read a signature from an HDF5 file"""
        if not sign_folder:
            destination_dir = self.get_destination_dir(dataset)
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
