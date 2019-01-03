import os
import gc
from scipy.sparse import lil_matrix
from sklearn.utils.sparsefuncs import mean_variance_axis
from gensim import corpora, models
import random
import tempfile
import shelve
import datetime
import numpy as np
import h5py
import glob
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from chemicalchecker.util import logged, Plot
from .signature_base import BaseSignature
from chemicalchecker.util import Config


@logged
class sign1(BaseSignature):
    """Signature type 1 class.

    Signature type 1 is...
    """

    def __init__(self, data_path, model_path, stats_path, dataset_info, **params):
        """Initialize the signature.

        Args:
            data_path(str): Where the h5 file is.
            model_path(str): Where the persistent model is.
        """
        self.__log.debug('data_path: %s', data_path)
        self.data_path = data_path
        self.model_path = os.path.join(model_path, "sig")
        self.__log.debug('model_path: %s', self.model_path)
        self.stats_path = stats_path
        self.__log.debug('stats_path: %s', self.stats_path)
        self.min_freq = 5
        self.max_freq = 0.25
        self.num_topics = None
        self.B_val = 10
        self.N_val = 1000
        self.B_distances = 1000000
        self.multipass = False
        self.variance_cutoff = 0.9
        self.integerize = False
        self.not_normalized = False
        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)
            if "min_freq" in params:
                self.min_freq = params["min_freq"]
            if "max_freq" in params:
                self.max_freq = params["max_freq"]
            if "num_topics" in params:
                self.num_topics = params["num_topics"]
            if "B" in params:
                self.B = params["B"]
            if "N" in params:
                self.N = params["N"]
            if "B_distances" in params:
                self.B_distances = params["B_distances"]
            if "multipass" in params:
                self.multipass = params["multipass"]
            if "variance_cutoff" in params:
                self.variance_cutoff = params["variance_cutoff"]
            if "integerize" in params:
                self.integerize = params["integerize"]
            if "not_normalized" in params:
                self.not_normalized = params["not_normalized"]

        # Calling base class to trigger file existence checks
        BaseSignature.__init__(
            self, data_path, model_path, stats_path, dataset_info)

    def fit(self, sign0, validations=True):
        """Take `sign0` and learn an unsupervised `sign1` predictor.

        Args:
            sign0(sign0): a `sign0` instance.
            validations(boolean):Create validation files(plots, files,etc)(default:True)
        """
        # Calling base class to trigger file existence checks
        BaseSignature.fit(self)
        # if not isinstance(sign0, Sign0.__class__):
        #     raise Exception("Fit method expects an instance of signature0")
        plot = Plot(self.dataset_info, self.stats_path)
        self.__log.debug('LSI/PCA fit %s' % sign0)
        FILE = os.path.join(self.model_path, "procs.txt")
        with open(FILE, "w") as f:
            if self.integerize:
                f.write("integerize\n")
            else:
                f.write("not_integerize\n")
            if self.not_normalized:
                f.write("not_normalized\n")
            else:
                f.write("normalized\n")

        input_data = str(sign0)

        with h5py.File(input_data, "r") as hf:
            keys = hf["keys"][:]
            V = hf["V"][:]

        tmp_dir = tempfile.mkdtemp(
            prefix='sign1_' + self.dataset_info.code + "_", dir=Config().PATH.CC_TMP)

        self.__log.debug("Temporary files saved in " + tmp_dir)

        if self.dataset_info.is_discrete:

            plain_corpus = os.path.join(tmp_dir, "sign1.corpus.txt")
            tfidf_corpus = os.path.join(tmp_dir, "sign1.mm")

            f = open(plain_corpus, "w")

            for i in range(0, len(keys)):
                f.write("%s %s\n" % (keys[i], V[i]))

            f.close()

            self.__log.info("Getting dictionary")

            dictionary = corpora.Dictionary(l.rstrip("\n").split(" ")[1].split(
                ",") for l in open(plain_corpus, "r"))

            dictionary.filter_extremes(
                no_below=self.min_freq, no_above=self.max_freq)
            dictionary.compactify()

            dictionary.save(self.model_path + "/dictionary.pkl")

            self.__log.info("Terms: %d" % len(dictionary))

            c = MyCorpus(plain_corpus, dictionary)

            Mols = len(c)

            self.__log.info("Corpus length: " + str(Mols))

            self.__log.info("Calculating TFIDF model")

            tfidf = models.TfidfModel(c)

            tfidf.save(self.model_path + "/tfidf.pkl")

            c_tfidf = tfidf[c]

            corpora.MmCorpus.serialize(tfidf_corpus, c_tfidf)

            self.__log.info("LSI model with %d topics..." % self.num_topics)

            if self.multipass:
                onepass = False
            else:
                onepass = True

            lsi = models.LsiModel(
                c_tfidf, id2word=dictionary, num_topics=self.num_topics, onepass=onepass)

            lsi.save(self.model_path + "/lsi.pkl")

            self.__log.info("LSI transformation of the TF-IDF corpus...")

            c_lsi = lsi[c_tfidf]

            self.__log.info("Deciding number of topics")

            exp_var_ratios = self._lsi_variance_explained(
                tfidf_corpus, lsi, B=self.B_val, N=self.N_val, num_topics=self.num_topics)

            cut_i, elb_i = plot.variance_plot(
                exp_var_ratios, variance_cutoff=self.variance_cutoff)

            with open(self.model_path + "/cut.txt", "w") as f:
                f.write("%d\n%d\n" % (cut_i, elb_i))

            self.__log.info("%.1f topics: %d" %
                            (self.variance_cutoff, cut_i + 1))
            self.__log.info("Elbow topics: %d" % (elb_i + 1))

            # Get inchikeys

            inchikeys = np.array([k for k in c.inchikeys()])

            V = np.empty((len(inchikeys), cut_i + 1))

            i = 0
            for l in c_lsi:
                v = np.zeros(cut_i + 1)
                for x in l[:cut_i + 1]:
                    if x[0] > cut_i:
                        continue
                    v[x[0]] = x[1]
                k = inchikeys[i]
                V[i, :] = v
                i += 1

            if self.not_normalized:
                pass
            else:
                self.__log.info("Normalizing")
                V = self._normalizer(V, False)

            if self.integerize:
                self.__log.info("Integerizing")
                V = self._integerize(V, False)

            self.__log.info("Saving to %s" % self.data_path)

            inchikey_sig = shelve.open(
                os.path.join(tmp_dir, "sign1.dict"), "n")
            for i in xrange(len(inchikeys)):
                inchikey_sig[str(inchikeys[i])] = V[i]
            inchikey_sig.close()
            f.close()

            self.__log.info("... but sorting before!")
            sort_idxs = np.argsort(inchikeys)

            with h5py.File(self.data_path, 'w') as hf:
                hf.create_dataset("keys", data=inchikeys[sort_idxs])
                hf.create_dataset("V", data=V[sort_idxs])
                hf.create_dataset("shape", data=V.shape)

            V = None
            c_lsi = None

            try:
                os.remove(plain_corpus)
                os.remove(tfidf_corpus)
                os.remove(tfidf_corpus + ".index")
            except:
                pass

        else:

            RowNames = []
            X = []

            for r in V:
                RowNames += [r[0]]
                X += [[float(x) for x in r[1].split(",")]]
            X = np.array(X)

            if self.dataset_info.coordinate == 'A5':
                scl = RobustScaler()
                scl.fit(X)
                joblib.dump(scl, self.model_path + "/robustscaler.pkl")
                X = scl.transform(X)

                scl = None

            InitMols = X.shape[0]
            Mols = InitMols

            self.__log.info("Fitting PCA")

            pca = PCA(n_components=self.variance_cutoff)

            pca.fit(X)

            joblib.dump(pca, self.model_path + "/pca.pkl")

            self.__log.info("Looking for variance")
            cut_i, elb_i = plot.variance_plot(
                pca.explained_variance_ratio_, variance_cutoff=self.variance_cutoff)
            with open(self.model_path + "/cut.txt", "w") as f:
                f.write("%d\n%d\n" % (cut_i, elb_i))

            self.__log.info("Projecting")

            V = pca.transform(X)

            self.__log.info("Saving stuff")

            if self.not_normalized:
                pass
            else:
                V = self._normalizer(V, False)

            if self.integerize:
                V = self._integerize(V, False)

            inchikeys = []
            inchikey_sig = shelve.open(
                os.path.join(tmp_dir, "sign1.dict"), "n")
            for i in xrange(len(RowNames)):
                inchikey = RowNames[i]
                inchikeys += [inchikey]
                inchikey_sig[str(inchikey)] = V[i]
            inchikey_sig.close()
            inchikeys = np.array(inchikeys)

            self.__log.info("... but sorting before!")
            sort_idxs = np.argsort(inchikeys)

            with h5py.File(self.data_path, "w") as hf:
                hf.create_dataset("keys", data=inchikeys[sort_idxs])
                hf.create_dataset("V", data=V[sort_idxs])
                hf.create_dataset("shape", data=V[sort_idxs].shape)

            V = []
            pca = []

        with h5py.File(self.data_path, "a") as hf:
            hf.create_dataset(
                "name", data=[str(self.dataset_info.code) + "_sig"])
            hf.create_dataset(
                "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            hf.create_dataset("metric", data=["cosine"])
            hf.create_dataset("normed", data=[not self.not_normalized])
            hf.create_dataset("integerized", data=[self.integerize])
            hf.create_dataset("principal_components", data=[True])

        with h5py.File(self.model_path + "/bg_distances.h5", "a") as hf:

            if "distance" not in hf.keys() or "pvalue" not in hf.keys():
                self.__log.info("Computing distance empirical P-values")

                inchikey_sig = shelve.open(
                    os.path.join(tmp_dir, "sign1.dict"), "r")
                pvals = plot.distance_background(
                    inchikey_sig, inchikeys, B=self.B_distances)
                inchikey_sig.close()

                hf.create_dataset(
                    "distance", data=np.array([p[0] for p in pvals]))
                hf.create_dataset(
                    "pvalue", data=np.array([p[1] for p in pvals]))

        self.__log.info("Cleaning")
        gc.collect()

        if validations:

            # Validation

            self.__log.info("MOA and ATC Validations")

            inchikey_sig = shelve.open(
                os.path.join(tmp_dir, "sign1.dict"), "r")
            # ks_moa, auc_moa = plot.vector_validation(
            #     inchikey_sig, "sig", prefix="moa")
            # ks_atc, auc_atc = plot.vector_validation(
            #     inchikey_sig, "sig", prefix="atc")
            inchikey_sig.close()
            for filename in glob.glob(os.path.join(tmp_dir, "sign1.dict*")):
                os.remove(filename)

            # Cleaning

            self.__log.info("Matrix plot")

            plot.matrix_plot(self.data_path)

    def predict(self, sign0, destination=None, validations=False):
        """Take `sign0` and predict `sign1`.

        Args:
            sign0(sign0): a `sign0` instance.
            destination(str): where to save the prediction by default the
                current signature data path.
            validations(boolean):Create validation files(plots, files,etc)(default:False)
        """
        # Calling base class to trigger file existence checks
        BaseSignature.predict(self)
        plot = Plot(self.dataset_info, self.stats_path)
        self.__log.debug('loading model from %s' % self.model_path)
        self.__log.debug('LSI/PCA predict %s' % sign0)
        FILE = os.path.join(self.model_path, "procs.txt")
        if destination is None:
            destination = self.data_path
        with open(FILE, "r") as f:
            i = f.next()
            n = f.next()
            if "not_integerize" in i:
                self.integerize = False
            else:
                self.integerize = True
            if "not_normalized" in n:
                self.not_normalized = True
            else:
                self.not_normalized = False

        input_data = str(sign0)

        with h5py.File(input_data, "r") as hf:
            keys = hf["keys"][:]
            V = hf["V"][:]

        tmp_dir = tempfile.mkdtemp(
            prefix='sign1_' + self.dataset_info.code + "_", dir=Config().PATH.CC_TMP)

        if self.dataset_info.is_discrete:

            plain_corpus = os.path.join(tmp_dir, "sign1.corpus.txt")
            tfidf_corpus = os.path.join(tmp_dir, "sign1.mm")

            f = open(plain_corpus, "w")

            for i in range(0, len(keys)):
                f.write("%s %s\n" % (keys[i], V[i]))

            f.close()

            self.__log.info("Getting dictionary")

            dictionary = corpora.Dictionary.load(
                self.model_path + "/dictionary.pkl")

            self.__log.info("Terms: %d" % len(dictionary))

            c = MyCorpus(plain_corpus, dictionary)

            Mols = len(c)

            self.__log.info("Corpus length: " + str(Mols))

            self.__log.info("Loading TFIDF model")

            tfidf = models.TfidfModel.load(self.model_path + "/tfidf.pkl")

            c_tfidf = tfidf[c]

            corpora.MmCorpus.serialize(tfidf_corpus, c_tfidf)

            lsi = models.LsiModel.load(self.model_path + "/lsi.pkl")

            self.__log.info("LSI transformation of the TF-IDF corpus...")

            c_lsi = lsi[c_tfidf]

            self.__log.info("Reading number of topics")

            with open(self.model_path + "/cut.txt", "r") as f:
                cut_i = int(f.next().rstrip())
                elb_i = int(f.next().rstrip())

            self.__log.info("%.1f topics: %d" %
                            (self.variance_cutoff, cut_i + 1))
            self.__log.info("Elbow topics: %d" % (elb_i + 1))

            # Get inchikeys

            inchikeys = np.array([k for k in c.inchikeys()])

            V = np.empty((len(inchikeys), cut_i + 1))

            i = 0
            for l in c_lsi:
                v = np.zeros(cut_i + 1)
                for x in l[:cut_i + 1]:
                    if x[0] > cut_i:
                        continue
                    v[x[0]] = x[1]
                k = inchikeys[i]
                V[i, :] = v
                i += 1

            if self.not_normalized:
                pass
            else:
                self.__log.info("Normalizing")
                V = self._normalizer(V, False)

            if self.integerize:
                self.__log.info("Integerizing")
                V = self._integerize(V, False)

            self.__log.info("Saving to %s" % destination)

            inchikey_sig = shelve.open(
                os.path.join(tmp_dir, "sign1.dict"), "n")
            for i in xrange(len(inchikeys)):
                inchikey_sig[str(inchikeys[i])] = V[i]
            inchikey_sig.close()
            f.close()

            self.__log.info("... but sorting before!")
            sort_idxs = np.argsort(inchikeys)

            with h5py.File(destination, 'w') as hf:
                hf.create_dataset("keys", data=inchikeys[sort_idxs])
                hf.create_dataset("V", data=V[sort_idxs])
                hf.create_dataset("shape", data=V.shape)

            V = None
            c_lsi = None

            try:
                os.remove(plain_corpus)
                os.remove(tfidf_corpus)
                os.remove(tfidf_corpus + ".index")
            except:
                pass

        else:

            RowNames = []
            X = []

            for r in V:
                RowNames += [r[0]]
                X += [[float(x) for x in r[1].split(",")]]
            X = np.array(X)

            if self.dataset_info.coordinate == 'A5':
                scl = RobustScaler()
                scl.fit(X)
                joblib.dump(scl, self.model_path + "/robustscaler.pkl")
                X = scl.transform(X)

                scl = None

            InitMols = X.shape[0]
            Mols = InitMols

            pca = joblib.load(self.model_path + "/pca.pkl")

            self.__log.info("Reading number of topics")
            with open(self.model_path + "/cut.txt", "r") as f:
                cut_i = int(f.next().rstrip())
                elb_i = int(f.next().rstrip())

            self.__log.info("Projecting")

            V = pca.transform(X)

            self.__log.info("Saving stuff")

            if self.not_normalized:
                pass
            else:
                V = self._normalizer(V, True)

            if self.integerize:
                V = self._integerize(V, True)

            inchikeys = []
            inchikey_sig = shelve.open(
                os.path.join(tmp_dir, "sign1.dict"), "n")
            for i in xrange(len(RowNames)):
                inchikey = RowNames[i]
                inchikeys += [inchikey]
                inchikey_sig[str(inchikey)] = V[i]
            inchikey_sig.close()
            inchikeys = np.array(inchikeys)

            self.__log.info("... but sorting before!")
            sort_idxs = np.argsort(inchikeys)

            with h5py.File(destination, "w") as hf:
                hf.create_dataset("keys", data=inchikeys[sort_idxs])
                hf.create_dataset("V", data=V[sort_idxs])
                hf.create_dataset("shape", data=V[sort_idxs].shape)

            V = []
            pca = []

        with h5py.File(destination, "a") as hf:
            hf.create_dataset(
                "name", data=[str(self.dataset_info.code) + "_sig"])
            hf.create_dataset(
                "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            hf.create_dataset("metric", data=["cosine"])
            hf.create_dataset("normed", data=[not self.not_normalized])
            hf.create_dataset("integerized", data=[self.integerize])
            hf.create_dataset("principal_components", data=[True])

        self.__log.info("Cleaning")
        gc.collect()

        if validations:

            # Validation

            self.__log.info("MOA and ATC Validations")

            inchikey_sig = shelve.open(
                os.path.join(tmp_dir, "sign1.dict"), "r")
            # ks_moa, auc_moa = plot.vector_validation(
            #     inchikey_sig, "sig", prefix="moa")
            # ks_atc, auc_atc = plot.vector_validation(
            #     inchikey_sig, "sig", prefix="atc")
            inchikey_sig.close()
            for filename in glob.glob(os.path.join(tmp_dir, "sign1.dict*")):
                os.remove(filename)

            # Cleaning

            self.__log.info("Matrix plot")

            plot.matrix_plot(destination)

    def statistics(self):
        """Perform a statistics."""
        # Calling base class to trigger file existence checks
        BaseSignature.validate(self)

    def _integerize(self, V, recycle):

        FILE = self.model_path + "/integerizer_ab.txt"

        def callibrator(lb, ub):
            # Returns a*x + b to convert V to an integer scale
            a = float(255) / (ub - lb)
            b = 127 - a * ub
            return a, b

        # Convert to integer type from -128 to 127
        if not recycle or not os.path.exists(FILE):
            lb = np.min(V)
            ub = np.max(V)
            a, b = callibrator(lb, ub)
            with open(FILE, "w") as f:
                f.write("%f\n%f" % (a, b))
        else:
            with open(FILE, "r") as f:
                a, b = [float(x) for x in f.read().split("\n")]

        def callibration(x):
            return a * x + b
        V = callibration(V)
        V = V.astype(np.int8)

        return V

    def _normalizer(self, V, recycle):

        FILE = self.model_path + "/normalizer.pkl"

        if not recycle or not os.path.exists(FILE):
            nlz = Normalizer(copy=True, norm="l2")
            V = nlz.fit_transform(V)
            joblib.dump(nlz, FILE)
        else:
            nlz = joblib.load(FILE)
            V = nlz.transform(V)

        return V.astype(np.float32)

    def _lsi_variance_explained(self, tfidf_corpus, lsi, B, N, num_topics):

            # Variance estimation (this may take a while...)
            # B: Number of runs, to ensure robustness
            # N: Size of the random sample sample (1000 should be enough, 100
            # works)

        mm = corpora.MmCorpus(tfidf_corpus)

        exp_var_ratios = []
        for _ in xrange(B):
            xt = []
            sm = lil_matrix((N, mm.num_terms))
            for i in xrange(N):
                io = random.randint(0, mm.num_docs - 1)
                terms = mm[io]
                # Transformed matrix
                tops = np.zeros(num_topics)
                for x in lsi[terms]:
                    if x[0] >= num_topics:
                        continue
                    tops[x[0]] = x[1]
                xt += [tops]
                # Sparse original matrix
                for t in terms:
                    sm[i, t[0] - 1] = t[1]
            xt = np.array(xt)
            sm = sm.tocsr()
            full_var = mean_variance_axis(sm, 0)[1].sum()

            try:
                exp_var = np.var(xt, axis=0)
                exp_var_ratios += [exp_var / full_var]
            except:
                continue

        exp_var_ratios = np.mean(np.array(exp_var_ratios), axis=0)

        return exp_var_ratios


# Corpus class

class MyCorpus(object):

    def __init__(self, plain_corpus, dictionary):
        self.plain_corpus = plain_corpus
        self.dictionary = dictionary

    def __iter__(self):
        for l in open(self.plain_corpus, "r"):
            l = l.rstrip("\n").split(" ")[1].split(",")
            bow = self.dictionary.doc2bow(l)
            if not bow:
                continue
            yield bow

    def __len__(self):
        return len([_ for _ in self.inchikeys()])

    def inchikeys(self):
        for l in open(self.plain_corpus, "r"):
            inchikey = l.split(" ")[0]
            l = l.rstrip("\n").split(" ")[1].split(",")
            bow = self.dictionary.doc2bow(l)
            if not bow:
                continue
            yield inchikey
