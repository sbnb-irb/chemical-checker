"""Signature type 1.

Signatures type 1 are basically processed signatures.

This imply mild compression (usually latent) of the signatures,
with a dimensionality that typically retains 90% of the original variance.
They keep most of the complexity of the original data and they can be used for
similarity calculations.

The typical preprocessing is a PCA (continuous data) or TF-IDF LSI.
"""
import os
import h5py
import uuid
import shutil
import pickle
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score

from .chemcheck import ChemicalChecker
from chemicalchecker.util import Config
from .signature_data import DataSignature
from .signature_base import BaseSignature

from chemicalchecker.util import logged
from chemicalchecker.util.transform.pca import Pca
from chemicalchecker.util.transform.scale import Scale


DEFAULT_T = 0.01


@logged
class sign1(BaseSignature, DataSignature):
    """Signature type 1 class."""

    def __init__(self, signature_path, dataset, **kwargs):
        """Initialize a Signature.

        Args:
            signature_path(str): the path to the signature directory.
            model_path(str): Where the persistent model is.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, signature_path, dataset, **kwargs)
        self.data_path = os.path.join(self.signature_path, "sign1.h5")
        DataSignature.__init__(self, self.data_path)

    def copy_sign0_to_sign1(self, s0, s1, just_data=False):
        """Copy from sign0 to sign1"""
        is_basesig = False
        if isinstance(s0, BaseSignature):
            if isinstance(s1, BaseSignature):
                is_basesig = True

        if is_basesig:
            if s0.molset != s1.molset:
                raise Exception(
                    "Copying from signature 0 to 1 is only allowed for "
                    "same molsets (reference or full)")

        self.__log.debug("Copying HDF5 dataset")
        shutil.copyfile(s0.data_path, s1.data_path)
        with h5py.File(s1.data_path, "a") as hf:
            if 'name' in hf.keys():
                del hf['name']
            hf.create_dataset("name", data=np.array(
                [str(self.dataset) + "sig"], DataSignature.string_dtype()))

        if not just_data:
            fn0 = os.path.join(s0.model_path, "triplets.h5")
            if os.path.exists(fn0):
                self.__log.debug("Copying triplets")
                fn1 = os.path.join(s1.model_path, "triplets.h5")
                shutil.copyfile(fn0, fn1)

    def duplicate(self, s1):
        self.__log.debug("Duplicating V matrix to V_tmp")
        with h5py.File(s1.data_path, "a") as hf:
            if "V_tmp" in hf.keys():
                self.__log.debug("Deleting V_tmp")
                del hf["V_tmp"]
            hf.create_dataset("V_tmp", hf["V"].shape, dtype=hf["V"].dtype)
            for i in range(0, hf["V"].shape[0]):
                hf["V_tmp"][i] = hf["V"][i][:]

    def was_sparse(self, max_keys=1000, zero_prop=0.5):
        """Guess if the matrix was sparse"""
        vals = self.subsample(max_keys)[0].ravel()
        if np.sum(vals == 0) / len(vals) > zero_prop:
            self.__log.debug("Matrix was probably sparse")
            return True
        else:
            self.__log.debug("Matrix was probably not sparse")
            return False

    def pipeline_file(self):
        fn = os.path.join(self.get_molset(
            "reference").model_path, "pipeline.pkl")
        return fn

    def load_model(self, name):

        fn = os.path.join(self.get_molset(
            "reference").model_path, "%s.pkl" % name)

        with open(fn, "rb") as f:
            mod = pickle.load(f)

        self.__log.debug("\n----> Loading model:" + fn)
        return mod

    def delete_tmp(self, s1):
        self.__log.debug("Deleting V_tmp")
        with h5py.File(s1.data_path, "r+") as hf:
            if "V_tmp" in hf.keys():
                del hf["V_tmp"]

    def fit(self, sign0=None, latent=True, scale=True, metric_learning=False,
            semisupervised=False, scale_kwargs={}, pca_kwargs={}, 
            lsi_kwargs={}, **kwargs):
        """Fit signature 1 given signature 0

            Args:
                sign0: A signature 0.
        """
        try:
            from chemicalchecker.util.transform.metric_learn import \
                UnsupervisedMetricLearn, SemiSupervisedMetricLearn
        except ImportError:
            raise ImportError("requires tensorflow https://tensorflow.org")
        try:
            from chemicalchecker.util.transform.lsi import Lsi
        except ImportError as ex:
            raise ex
        BaseSignature.fit(self, **kwargs)
        self.clear()
        # signature specific checks
        if sign0 is None:
            sign0 = self.get_sign('sign0').get_molset("full")
        if sign0.cctype != "sign0":
            raise Exception("A signature type 0 is expected..!")
        if sign0.molset != "full":
            raise Exception(
                "Fit should be done with the full signature 0 "
                "(even if inside reference is used)")
        # preparing signatures
        self.update_status("Getting data")
        s0_ref = sign0.get_molset("reference")
        s1_ref = self.get_molset("reference")
        s1_ref.clear()
        self.__log.debug("Placing sign0 to sign1 (done for reference)")
        self.copy_sign0_to_sign1(s0_ref, s1_ref)
        self.__log.debug("Placing sign0 to sign1 (done for full)")
        self.copy_sign0_to_sign1(sign0, self)
        self.__log.debug("Duplicating signature (tmp) (done for reference)")
        self.duplicate(s1_ref)
        self.__log.debug("Duplicating signature (tmp) (done for full)")
        self.duplicate(self)
        self.__log.debug("Checking if matrix was sparse or not")
        sparse = s1_ref.was_sparse()

        tmp = False
        if metric_learning:
            tmp = True

        if sparse:
            self.__log.debug("Sparse matrix pipeline")
            if latent:
                self.update_status("LSI")
                self.__log.debug("Looking for latent variables with"
                                 "TFIDF-LSI (done for tmp)")
                mod = Lsi(self, tmp=tmp, **lsi_kwargs)
                mod.fit()
            else:
                self.__log.debug("Not looking for latent variables")
        else:
            self.__log.debug("Dense matrix pipeline")
            if scale:
                self.update_status("Scaling")
                mod = Scale(self, tmp=False, **scale_kwargs)
                mod.fit()
            else:
                self.__log.debug("Not scaling")
            if latent:
                self.update_status("PCA")
                mod = Pca(self, **pca_kwargs)
                mod.fit()
            else:
                self.__log.debug("Not looking for latent variables")
        if metric_learning:
            self.update_status("Metric Learning")
            if semisupervised:
                self.__log.debug("Semi-supervised metric learning")
                mod = SemiSupervisedMetricLearn(self, tmp=False)
                mod.fit()
            else:
                self.__log.debug("Unsupervised metric learning")
                self.__log.debug("First doing neighbors")
                mod = UnsupervisedMetricLearn(self, tmp=False)
                mod.fit()
        # save pipeline
        pipeline = {
            "sparse": sparse,
            "latent": latent,
            "scale": scale,
            "metric_learning": metric_learning,
            "semisupervised": semisupervised
        }
        self.delete_tmp(self)
        self.delete_tmp(s1_ref)
        fn = self.pipeline_file()
        with open(fn, "wb") as f:
            pickle.dump(pipeline, f)

        with h5py.File(s1_ref.data_path, "a") as hf:
            hf.create_dataset("metric", data=np.array(
                ["cosine"], DataSignature.string_dtype()))
        # finalize signature
        BaseSignature.fit_end(self, **kwargs)

    def predict(self, sign0, destination):
        """Use the learned model to predict the signature.

        Args:
            sign1(signature): A valid Signature type 1
            destination(None|path|signature): If None the prediction results are
                returned as dictionary, if str then is used as path for H5 data,
                if empty Signature type 2 its data_path is used as destination.
        """
        if not isinstance(destination, BaseSignature):
            """
            tag = str(uuid.uuid4())
            tmp_path = os.path.join(self.model_path, tag)
            cc = ChemicalChecker(tmp_path)
            s1 = cc.get_signature(self.cctype, self.molset, self.cctype)
            destination = s1
            """
            raise NotImplementedError(
                "'destination' must be a valid signature object.")
        if not os.path.isfile(sign0.data_path):
            raise Exception("The file " + sign0.data_path + " does not exist")
        self.copy_sign0_to_sign1(sign0, destination, just_data=True)
        self.__log.debug("Reading pipeline")
        fn = self.pipeline_file()
        with open(fn, "rb") as f:
            pipeline = pickle.load(f)
        self.__log.debug("Starting pipeline")
        if not pipeline["sparse"] and pipeline["scale"]:
            self.__log.debug("Scaling")
            mod = self.load_model("scale")
            mod.model_path = self.model_path
            # using the config path for the CC_TMP and not the generated model one
            mod.tmp_path = Config().PATH.CC_TMP
            mod.predict(destination)

        self.__log.debug("Transformation")
        if pipeline["metric_learning"]:
            if pipeline["semisupervised"]:
                mod = self.load_model("semiml")
            else:
                mod = self.load_model("unsupml")
        else:
            if pipeline["latent"]:
                if pipeline["sparse"]:
                    mod = self.load_model("lsi")
                else:
                    mod = self.load_model("pca")
            else:
                mod = None
        if mod is not None:
            # avoid taking the info from pickle in case it is copied
            mod.model_path = self.model_path
            # using the config path for the CC_TMP and not the generated model one
            mod.tmp_path = Config().PATH.CC_TMP
            mod.predict(destination)

        destination.refresh()
        self.__log.debug("Prediction done!")

    def neighbors(self, tmp, metric="cosine", k_neig=1000, cpu=4):
        """Neighbors"""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "requires faiss  https://github.com/facebookresearch/faiss")

        s1 = self.get_molset("reference")
        if metric not in ["cosine", "euclidean"]:
            raise Exception("Metric must be 'cosine' or 'euclidean'")
        faiss.omp_set_num_threads(cpu)
        if tmp:
            V_name = "V_tmp"
        else:
            V_name = "V"
        data_path = os.path.join(s1.model_path, "neig.h5")
        self.__log.debug(
            "Calculating nearest neighbors. Saving in: %s" % data_path)
        with h5py.File(s1.data_path, 'r') as dh5, \
                h5py.File(data_path, 'w') as dh5out:
            datasize = dh5[V_name].shape
            # data_type = dh5[V_name].dtype
            self.__log.debug("...data size is (%d, %d)" %
                             (datasize[0], datasize[1]))
            k = min(datasize[0], k_neig)
            dh5out.create_dataset("row_keys", data=dh5["keys"][:])
            dh5out["col_keys"] = h5py.SoftLink('/row_keys')
            dh5out.create_dataset(
                "indices", (datasize[0], k), dtype='int32')
            dh5out.create_dataset(
                "distances", (datasize[0], k), dtype='float32')
            dh5out.create_dataset("shape", data=(datasize[0], k))
            dh5out.create_dataset(
                "metric", data=[metric.encode(encoding='UTF-8',
                                              errors='strict')])
            if metric == "euclidean":
                index = faiss.IndexFlatL2(datasize[1])
            else:
                index = faiss.IndexFlatIP(datasize[1])
            for chunk in s1.chunker():
                data_temp = np.array(dh5[V_name][chunk], dtype='float32')
                if metric == "cosine":
                    normst = LA.norm(data_temp, axis=1)
                    index.add( np.array( data_temp / normst[:, None], dtype='float32') )
                else:
                    index.add( np.array(data_temp, dtype='float32') )
            for chunk in s1.chunker():
                data_temp = np.array(dh5[V_name][chunk], dtype='float32')
                if metric == "cosine":
                    normst = LA.norm(data_temp, axis=1)
                    Dt, It = index.search( np.array(data_temp / normst[:, None], dtype='float32'), k)
                else:
                    Dt, It = index.search( np.array(data_temp, dtype='float32'), k)
                dh5out["indices"][chunk] = It
                if metric == "cosine":
                    dh5out["distances"][chunk] = np.maximum(0.0, 1.0 - Dt)
                else:
                    dh5out["distances"][chunk] = Dt

    def get_triplets(self, reference):
        """Read triplets of signature across the CC"""
        if reference:
            fn = os.path.join(self.get_molset(
                "reference").model_path, "triplets.h5")
        else:
            fn = os.path.join(self.model_path, "triplets.h5")
        if not os.path.exists(fn):
            return None
        self.__log.debug("Getting triplets from %s" % fn)
        with h5py.File(fn, "r") as hf:
            triplets = hf["triplets"][:]
        return triplets

    def get_self_triplets(self, local_neig_path, num_triplets=10000000):
        """Get triplets of signatures only looking at itself"""
        s1 = self.get_molset("reference")
        if local_neig_path:
            neig_path = os.path.join(s1.model_path, "neig.h5")
        opt_t = self.optimal_t(local_neig_path=local_neig_path)
        # Heuristic to correct opt_t, dependent on the size of the data
        LB = 10000
        UB = 100000
        TMAX = 50
        TMIN = 10

        def get_t_max(n):
            n = np.clip(n, LB, UB)
            a = (TMAX - TMIN) / (LB - UB)
            b = TMIN - a * UB
            return a * n + b

        with h5py.File(neig_path, "r") as hf:
            N, kn = hf["indices"].shape
            opt_t = np.min([opt_t, 0.01])
            k = np.clip(opt_t * N, 5, 100)
            k = np.min([k, kn * 0.5 + 1])
            k = np.max([k, 5])
            k = np.min([k, get_t_max(N)])
            k = int(k)
            self.__log.debug("... selected T is %d" % k)
            nn_indices = hf["indices"][:]
            nn_pos = nn_indices[:, 1:(k + 1)]
            nn_neg = nn_indices[:, (k + 1):]
        self.__log.debug("Starting sampling (pos:%d, neg:%d)" %
                         (nn_pos.shape[1], nn_neg.shape[1]))
        n_sample = np.clip(int(num_triplets / N), 1, 100)
        triplets = []
        med_neg = nn_neg.shape[1]
        nn_pos_prob = [(len(nn_pos) - i) for i in range(0, nn_pos.shape[1])]
        nn_neg_prob = [(len(nn_neg) - i) for i in range(0, nn_neg.shape[1])]
        nn_pos_prob = np.array(nn_pos_prob) / np.sum(nn_pos_prob)
        nn_neg_prob = np.array(nn_neg_prob) / np.sum(nn_neg_prob)
        for i in range(0, N):
            # sample positives with replacement
            pos = np.random.choice(
                nn_pos[i], n_sample, p=nn_pos_prob, replace=True)
            if n_sample > med_neg:
                # sample "medium" negatives
                neg = np.random.choice(
                    nn_neg[i], med_neg, p=nn_neg_prob, replace=False)
                # for the rest, sample "easy" negatives
                forb = set(list(nn_pos[i]) + list(nn_neg[i]))
                cand = [i for i in range(0, N) if i not in forb]
                if len(cand) > 0:
                    neg_ = np.random.choice(
                        cand, min(len(cand), n_sample - med_neg), replace=True)
                    neg = np.array(list(neg) + list(neg_))
            else:
                neg = np.random.choice(nn_neg[i], n_sample, replace=False)
            if len(pos) > len(neg):
                neg = np.random.choice(neg, len(pos), replace=True)
            elif len(pos) < len(neg):
                neg = np.random.choice(neg, len(pos), replace=False)
            else:
                pass
            for p, n in zip(pos, neg):
                triplets += [(i, p, n)]
        triplets = np.array(triplets).astype(int)
        fn = os.path.join(s1.model_path, "triplets_self.h5")
        self.__log.debug("Triplets path: %s" % fn)
        with h5py.File(fn, "w") as hf:
            hf.create_dataset("triplets", data=triplets)
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

    def optimal_t(self, max_triplets=10000, min_triplets=1000,
                  local_neig_path=False, save=True):
        """Find optimal (recommended) number of neighbors.

        Based on the accuracy of triplets across the CC.
        Neighbors class needs to be precomputed.
        Only done for the reference set (it doesn't really make sense to do it
        for the full).

        Args:
            max_triplets(int): Maximum number of triplets to consider
                (default=10000).
            save(bool): Store an opt_t.h5 file (default=True).
        """
        # lazily loading if already computed
        fname = os.path.join(
            self.get_molset("reference").model_path, "opt_t.h5")
        if os.path.isfile(fname):
            with h5py.File(fname, "r") as hf:
                opt_t = hf['opt_t'][0]
            return opt_t
        self.__log.debug("Reading triplets")
        triplets = self.get_triplets(reference=True)
        if triplets is None:
            self.__log.debug("No triplets were found. Returning ")
            return DEFAULT_T
        if len(triplets) < min_triplets:
            self.__log.warning("Not enough triplets... t is %f" % DEFAULT_T)
            return DEFAULT_T
        self.__log.debug("Selecting available anchors")
        if len(triplets) > max_triplets:
            idxs = np.random.choice(len(triplets), max_triplets, replace=False)
            triplets = triplets[idxs]
        anchors = sorted(set(triplets[:, 0]))
        anchors_idxs = dict((k, i) for i, k in enumerate(anchors))
        self.__log.debug("Reading from nearest neighbors")
        if not local_neig_path:
            self.__log.debug(
                "Getting neighbors data from a proper neig instance")
            neig = self.get_molset("reference").get_neig()
            neig_path = neig.data_path
        else:
            neig_path = os.path.join(self.get_molset(
                "reference").model_path, "neig.h5")
            self.__log.debug(
                "Getting neighbors data from file: %s" % neig_path)
        with h5py.File(neig_path, "r") as hf:
            nn = hf["indices"][anchors][:, 1:101]
        self.__log.debug("Negatives and positives")
        positives = [(anchors_idxs[t[0]], t[1]) for t in triplets]
        negatives = [(anchors_idxs[t[0]], t[2]) for t in triplets]
        pairs = positives + negatives
        truth = [1] * len(positives) + [0] * len(negatives)
        self.__log.debug("Setting the range of search")
        # N = nn.shape[0]
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
                n_pairs += [(i, n) for n in nn[i, :k]]
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
        opt_t = opt_k / nn.shape[0]
        if save:
            self.__log.debug("Saving")
            with h5py.File(fname, "w") as hf:
                hf.create_dataset(
                    "accuracies", data=np.array(accus).astype(np.int))
                hf.create_dataset("opt_t", data=np.array([opt_t]))
                hf.create_dataset("opt_k", data=np.array([opt_k]))
        return opt_t
