import os
import gc
import h5py
import glob
import random
import shelve
import tempfile
import datetime
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, cosine
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.preprocessing import Normalizer, RobustScaler

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util.plot import Plot


@logged
class sign1(BaseSignature, DataSignature):
    """Signature type 1 class.

    Signature type 1 is...
    """

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
        self.discrete = True

        if "discrete" not in params:
            self.__log.warning(
                'Sign1 using default value (True) for discrete parameter')

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
            if "discrete" in params:
                self.discrete = params["discrete"]

    def fit(self, sign0, validations=True):
        """Take `sign0` and learn an unsupervised `sign1` predictor.

        Args:
            sign0(sign0): a `sign0` instance.
            validations(boolean):Create validation files(plots, files,etc)(default:True)
        """
        try:
            from gensim import corpora, models
        except ImportError:
            raise ImportError("requires gensim " +
                              "https://radimrehurek.com/gensim/")
        # Calling base class to trigger file existence checks
        BaseSignature.fit(self)
        # if not isinstance(sign0, Sign0.__class__):
        #     raise Exception("Fit method expects an instance of signature0")
        plot = Plot(self.dataset, self.stats_path)
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

        input_data = sign0.data_path
        mappings = None

        tmp_dir = tempfile.mkdtemp(
            prefix='sign1_' + self.dataset + "_", dir=Config().PATH.CC_TMP)

        self.__log.debug("Temporary files saved in " + tmp_dir)

        if self.discrete:

            plain_corpus = os.path.join(tmp_dir, "sign1.corpus.txt")
            tfidf_corpus = os.path.join(tmp_dir, "sign1.mm")

            f = open(plain_corpus, "w")

            with h5py.File(input_data, "r") as hf:
                features = hf["features"][:]
                if "mappings" in hf.keys():
                    mappings = hf["mappings"][:]
                for chunk in sign0.chunker():
                    V = hf["V"][chunk]
                    keys = hf['keys'][chunk]
                    for key, row in zip(keys, V):
                        mask = np.where(row > 0)
                        val = ",".join([",".join([features[x]] * row[x])
                                        for x in mask[0]])
                        f.write("%s %s\n" % (key, val))

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

            if self.num_topics is None:
                num_topics = np.min([int(0.67 * len(dictionary)), 5000])
            else:
                num_topics = self.num_topics

            self.__log.info("LSI model with %d topics..." % num_topics)

            if self.multipass:
                onepass = False
            else:
                onepass = True

            lsi = models.LsiModel(
                c_tfidf, id2word=dictionary, num_topics=num_topics, onepass=onepass)

            lsi.save(self.model_path + "/lsi.pkl")

            self.__log.info("LSI transformation of the TF-IDF corpus...")

            c_lsi = lsi[c_tfidf]

            self.__log.info("Deciding number of topics")

            exp_var_ratios = self._lsi_variance_explained(
                tfidf_corpus, lsi, B=self.B_val, N=self.N_val, num_topics=num_topics)

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
                hf.create_dataset("elbow", data=[elb_i])

            V = None
            c_lsi = None

            try:
                os.remove(plain_corpus)
                os.remove(tfidf_corpus)
                os.remove(tfidf_corpus + ".index")
            except:
                pass

        else:

            with h5py.File(input_data, "r") as hf:
                keys = hf["keys"][:]
                V = hf["V"]
                if "mappings" in hf.keys():
                    mappings = hf["mappings"][:]

                RowNames = []
                X = []

                for i in range(0, V.shape[0]):
                    r = V[i]
                    RowNames += [keys[i]]
                    X += [r]
            X = np.array(X)

            if self.dataset[:2] == 'A5':
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
                hf.create_dataset("elbow", data=[elb_i])

            V = []
            pca = []

        with h5py.File(self.data_path, "a") as hf:
            hf.create_dataset(
                "name", data=[str(self.dataset) + "_sig"])
            hf.create_dataset(
                "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            hf.create_dataset("metric", data=["cosine"])
            hf.create_dataset("normed", data=[not self.not_normalized])
            hf.create_dataset("integerized", data=[self.integerize])
            hf.create_dataset("principal_components", data=[True])
            if mappings is not None:
                # Sometimes sign1 might reduce the number of output signatures
                # If so, we should update the mappings in the sign1.h5 file
                mask = np.isin(mappings[:, 1], hf["keys"][:])
                hf.create_dataset("mappings", data=mappings[mask])

        with h5py.File(self.model_path + "/bg_cosine_distances.h5", "a") as hf:

            if "distance" not in hf.keys() or "pvalue" not in hf.keys():
                self.__log.info("Computing cosine distance empirical P-values")

                inchikey_sig = shelve.open(
                    os.path.join(tmp_dir, "sign1.dict"), "r")
                pvals = self.background_distances("cosine",
                                                  inchikey_sig, inchikeys, B=self.B_distances)
                inchikey_sig.close()

                hf.create_dataset(
                    "distance", data=np.array([p[0] for p in pvals]))
                hf.create_dataset(
                    "pvalue", data=np.array([p[1] for p in pvals]))

        with h5py.File(self.model_path + "/bg_euclidean_distances.h5", "a") as hf:

            if "distance" not in hf.keys() or "pvalue" not in hf.keys():
                self.__log.info(
                    "Computing euclidean distance empirical P-values")

                inchikey_sig = shelve.open(
                    os.path.join(tmp_dir, "sign1.dict"), "r")
                pvals = self.background_distances("euclidean",
                                                  inchikey_sig, inchikeys, B=self.B_distances)
                inchikey_sig.close()

                hf.create_dataset(
                    "distance", data=np.array([p[0] for p in pvals]))
                hf.create_dataset(
                    "pvalue", data=np.array([p[1] for p in pvals]))

        self.__log.info("Cleaning")
        gc.collect()

        if validations:
            self.validate()

        for filename in glob.glob(os.path.join(tmp_dir, "sign1.dict*")):
            os.remove(filename)
        os.rmdir(tmp_dir)

        self.mark_ready()

    def predict(self, sign0, destination=None, validations=False):
        """Take `sign0` and predict `sign1`.

        Args:
            sign0(sign0): a `sign0` instance.
            destination(str): where to save the prediction by default the
                current signature data path.
            validations(boolean):Create validation files(plots, files,etc)(default:False)
        """
        try:
            from gensim import corpora, models
        except ImportError:
            raise ImportError("requires gensim " +
                              "https://radimrehurek.com/gensim/")
        # Calling base class to trigger file existence checks
        BaseSignature.predict(self)
        plot = Plot(self.dataset, self.stats_path)
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

        input_data = sign0.data_path
        mappings = None

        tmp_dir = tempfile.mkdtemp(
            prefix='sign1_' + self.dataset + "_", dir=Config().PATH.CC_TMP)

        if self.discrete:

            with h5py.File(input_data, "r") as hf:
                keys = hf["keys"][:]
                V = hf["V"]
                features = hf["features"][:]
                if "mappings" in hf.keys():
                    mappings = hf["mappings"][:]

                plain_corpus = os.path.join(tmp_dir, "sign1.corpus.txt")
                tfidf_corpus = os.path.join(tmp_dir, "sign1.mm")

                f = open(plain_corpus, "w")

                for i in range(0, len(keys)):
                    row = V[i]
                    mask = np.where(row > 0)
                    val = ",".join([",".join([features[x]] * row[x])
                                    for x in mask[0]])
                    f.write("%s %s\n" % (keys[i], val))

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

            with h5py.File(input_data, "r") as hf:
                keys = hf["keys"][:]
                V = hf["V"]
                if "mappings" in hf.keys():
                    mappings = hf["mappings"][:]

                RowNames = []
                X = []

                for i in range(0, V.shape[0]):
                    r = V[i]
                    RowNames += [keys[i]]
                    X += [r]
            X = np.array(X)

            if self.dataset[:2] == 'A5':
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
                "name", data=[str(self.dataset) + "_sig"])
            hf.create_dataset(
                "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            hf.create_dataset("metric", data=["cosine"])
            hf.create_dataset("normed", data=[not self.not_normalized])
            hf.create_dataset("integerized", data=[self.integerize])
            hf.create_dataset("principal_components", data=[True])
            if mappings is not None:
                # Sometimes sign1 reduce the number of output signatures
                # If so, we should update the mappings in the sign1.h5 file
                mask = np.isin(mappings[:, 1], hf["keys"][:])
                hf.create_dataset("mappings", data=mappings[mask])

        self.__log.info("Cleaning")
        gc.collect()

        if validations:

            # Validation

            self.__log.info("MOA and ATC Validations")

            if mappings is not None:
                inchikey_mappings = dict(mappings)
            else:
                inchikey_mappings = None

            inchikey_sig = shelve.open(
                os.path.join(tmp_dir, "sign1.dict"), "r")
            ks_moa, auc_moa, frac_moa = plot.vector_validation(
                self, "sign1", prefix="moa", mappings=inchikey_mappings)
            ks_atc, auc_atc, frac_atc = plot.vector_validation(
                self, "sign1", prefix="atc", mappings=inchikey_mappings)
            inchikey_sig.close()

            # Cleaning

            self.__log.info("Matrix plot")

            plot.matrix_plot(destination)

        for filename in glob.glob(os.path.join(tmp_dir, "sign1.dict*")):
            os.remove(filename)
        os.rmdir(tmp_dir)

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
            nlz = Normalizer(copy=False, norm="l2")
            nlz.fit_transform(V)
            joblib.dump(nlz, FILE)
        else:
            nlz = joblib.load(FILE)
            nlz.transform(V)

        return V.astype(np.float32)

    def _lsi_variance_explained(self, tfidf_corpus, lsi, B, N, num_topics):

            # Variance estimation (this may take a while...)
            # B: Number of runs, to ensure robustness
            # N: Size of the random sample sample (1000 should be enough, 100
            # works)
        try:
            from gensim import corpora
        except ImportError:
            raise ImportError("requires gensim " +
                              "https://radimrehurek.com/gensim/")

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
