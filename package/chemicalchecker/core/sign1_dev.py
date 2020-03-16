"""Signature type 1.

Signatures type 1 are basically processed signatures. The typical preprocessing is a PCA (continuous data) or TF-IDF LSI (continuous).

Processing steps can be added as a pipeline previous to fitting the data.

A reference (non-redundant) dataset is always used for all of the fits.
"""
import os
import h5py
import numpy as np
import datetime
import shutil
from scipy.signal import savgol_filter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from .signature_data import DataSignature
from .signature_base import BaseSignature
from chemicalchecker.util import logged

from chemicalchecker.util.transform.scale import Scale
from chemicalchecker.util.transform.lsi import Lsi
from chemicalchecker.util.transform.pca import Pca
from chemicalchecker.util.outlier_removal import OutlierRemover
from chemicalchecker.util.transform.metric_learn import NoMetricLearn, UnsupervisedMetricLearn, SemiSupervisedMetricLearn


@logged
class sign1(BaseSignature, DataSignature):
    """Signature type 1 class."""
    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): the path to the signature directory.
            model_path(str): Where the persistent model is.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(self.signature_path, "sign1.h5")
        DataSignature.__init__(self, self.data_path)
        self.data_path_tmp = os.path.join(self.signature_path, "sign1_tmp.h5")

    def copy_sign0_to_sign1(self, s0, s1):
        """Copy from sign0 to sign1"""
        if s0.molset != s1.molset:
            raise Exception("Copying from signature 0 to 1 is only allowed for same molsets (reference or full)")
        self.__log.debug("Copying HDF5 dataset")
        with h5py.File(s1.data_path, "w") as hf:
            hf.create_dataset(
                "name", data=np.array([str(self.dataset) + "sig"], DataSignature.string_dtype()))
            hf.create_dataset(
                "date", data=np.array([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], DataSignature.string_dtype()))
            hf.create_dataset("V", data=s0[:])
            hf.create_dataset("keys", data=np.array(s0.keys, DataSignature.string_dtype()))
            if s0.molset == "reference":
                mappings = s0.get_h5_dataset("mappings")
                hf.create_dataset("mappings", data=np.array(mappings, DataSignature.string_dtype()))
        self.__log.debug("Copying triplets")
        fn0 = os.path.join(s0.model_path, "triplets.h5")
        fn1 = os.path.join(s1.model_path, "triplets.h5")
        shutil.copyfile(fn0, fn1)

    def was_sparse(self, max_keys=1000, zero_prop=0.5):
        """Guess if the matrix was sparse"""
        vals = self.subsample(max_keys)[0].ravel()
        if np.sum(vals != 0)/len(vals) > zero_prop:
            self.__log.debug("Matrix was probably sparse")
            return True
        else:
            self.__log.debug("Matrix was probably not sparse")
            return False

    def pipeline_file(self):
        fn = os.path.join(self.get_molset("reference").model_path, "pipeline.pkl")
        return fn

    def fit(self, sign0, latent=True, scale=True, remove_outliers=False, metric_learning=True, semisupervised=False):
        """Fit a signature 1, given a signature 0

            Args:
                sign0: A signature 0.
        """
        self.__log.debug("Fitting")
        if sign0.cctype != "sign0":
            raise Exception("A signature type 0 is expected..!")
        if sign0.molset != "full":
            raise Exception("Fit should be done with the full signature 0 (even if inside reference is used)")
        sign0_ref = sign0.get_molset("reference")
        sign1_ref = self.get_molset("reference")
        self.__log.debug("Placing sign0 to sign1 (done for reference)")
        self.copy_sign0_to_sign1(sign0_ref, sign1_ref)
        self.__log.debug("Placing sign0 to sign1 (done for full)")
        self.copy_sign0_to_sign1(sign0, self)
        self.__log.debug("Checking if matrix was sparse or not")
        if latent:
            self.__log.debug("Looking for latent variables")
            sparse = sign1_ref.was_sparse(max_keys=max_keys, zero_prop=zero_prop)
            if sparse:
                self.__log.debug("Starting pipeline for sparse matrix (TFIDF LSI)")
                mod = Lsi()
                mod.fit()
            else:
                self.__log.debug("Starting pipeline for dense matrix")
                if scale:
                    self.__log.debug("Scaling")
                    mod = Scale()
                    mod.fit()
                else:
                    self.__log.debug("Not scaling")
                self.__log.debug("PCA")
                mod = Pca()
                mod.fit()
        else:
            self.__log.debug("Not looking for latent variables")
            sparse = None
        if remove_outliers:
            self.__log.debug("Looking for further outliers")
            mod = OutlierRemover()
            mod.fit()
        else:
            self.__log.debug("Not looking for further outliers")
            pass
        self.__log.debug("Pipeline done, now doing metric learning")
        if metric_learning:
            self.__log.debug("Not learning any metric")
            mod = NoMetricLearn()
            mod.fit()
        else:
            if semisupervised:
                self.__log.debug("")
                mod = SemiSupervisedMetricLearn()
                mod.fit()
            else:
                self.__log.debug("Unsupervised metric learning")
                mod = UnsupervisedMetricLearn()
                mod.fit()
        self.__log.debug("Saving pipeline")
        pipeline = {
            "sparse": sparse,
            "latent": latent,
            "scale" : scale,
            "remove_outliers": remove_outliers,
            "metric_learning": metric_learning,
            "semisupervised": semisupervised
        }
        fn = self.pipeline_file()
        with open(fn, "wb") as f:
            pickle.dump(pipeline, f)
        
    def predict(self, sign0):
        """Predict sign1 from sign0"""
        self.__log.debug("Reading pipeline")
        fn = self.pipeline_file()
        with open(fn, "rb") as f:
            pipeline = pickle.load(f)
        self.__log.debug("Starting pipeline")
        if not pipeline["sparse"] and pipeline["scale"]:
            sign0 = ""
        if pipeline["metric_learning"]:
            if pipeline["sparse"]:
                pass
            else:
                if pipeline["scale"]:
                    scale = ""
                else:
                    ""
        else:
            if pipeline["sparse"]:
                ""
            
        if pipeline["sparse"]:
            pass
        else:
            
        if pipeline["metric_learning"]:
            self.__log.debug("Metric learning was done. Using it to project.")
            "XXXX"
        else:
            self.__log.debug("No metric learning was done")
            if pipeline["latent"]:
                "XXXX"
            else:
                "XXXX"

    def get_triplets(self, reference):
        """Read triplets of signature"""
        if reference:
            fn = os.path.join(self.get_molset("reference").model_path, "triplets.h5")
        else:
            fn = os.path.join(self.model_path, "triplets.h5")
        with h5py.File(fn, "r") as hf:
            triplets = hf["triplets"][:]
        return triplets

    def score(self, reference, max_triplets=10000):
        """Score based on triplets.

        Args:
            max_triplets(int): Maximum number of triplets to consider.
        """
        self.__log.debug("Score the transformation based on triplets accuracy")
        if reference:
            sign = self.get_molset("reference")
        else:
            sign = self
        triplets = self.get_triplets(reference)
        idxs = np.random.choice(triplets.shape[0], max_triplets, replace=False)
        triplets = triplets[idxs]
        acc = 0
        for t in tqdm(triplets):
            a = sign[int(t[0])]
            p = sign[int(t[1])]
            n = sign[int(t[2])]
            if cosine(a, p) < cosine(a, n):
                acc += 1
        acc /= len(triplets)
        return acc

    def optimal_t(self, max_triplets=10000):
        """Find optimal (recommended) number of neighbors, based on the accuracy of triplets across the CC.
        Neighbors class needs to be precomputed.
        Only done for the reference set (it doesn't really make sense to do it for the full).

        Args:
            max_triplets(int): Maximum number of triplets to consider (default=10000).
        """
        self.__log.debug("Getting neighbors instance")
        neig = self.get_molset("reference").get_neig()
        self.__log.debug("Reading triplets")
        triplets = self.get_triplets(reference=True)
        self.__log.debug("Selecting available anchors")
        if len(triplets) > max_triplets:
            idxs = np.random.choice(len(triplets), max_triplets, replace=False)
            triplets = triplets[idxs]
        anchors = sorted(set(triplets[:,0]))
        anchors_idxs = dict((k,i) for i,k in enumerate(anchors))
        self.__log.debug("Reading from nearest neighbors")
        with h5py.File(neig.data_path, "r") as hf:
            nn = hf["indices"][anchors][:,1:101]
        self.__log.debug("Negatives and positives")
        positives = [(anchors_idxs[t[0]], t[1]) for t in triplets]
        negatives = [(anchors_idxs[t[0]], t[2]) for t in triplets]
        pairs     = positives + negatives
        truth     = [1]*len(positives) + [0]*len(negatives)
        self.__log.debug("Setting the range of search")
        N = neig.shape[0]
        kranges = []
        for i in range(5, 10):
            kranges += [i]
        for i in range(10, 50, 2):
            kranges += [i]
        for i in range(50, 100, 5):
            kranges += [i]
        self.__log.debug("Screening")
        accus = []
        for k in kranges:
            n_pairs = []
            for i in range(0, nn.shape[0]):
                n_pairs += [(i, n) for n in nn[i,:k]]
            pred = []
            n_pairs = set(n_pairs)
            for p in pairs:
                if p in n_pairs:
                    pred += [1]
                else:
                    pred += [0]
            accus += [(k, accuracy_score(truth, pred))]
        idx = np.argmax(savgol_filter([x[1] for x in accus], 11, 3))
        opt_k = accus[idx][0]
        opt_t = opt_k/neig.shape[0]
        with h5py.File(os.path.join(self.get_molset("reference").model_path, "opt_t.h5"), "w") as hf:
            hf.create_dataset("accuracies", data=np.array(accus).astype(np.int))
            hf.create_dataset("opt_t", data=np.array([opt_t]))
            hf.create_dataset("opt_k", data=np.array([opt_k]))
        return opt_t