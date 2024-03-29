"""Do TFIDF-LSI."""
import os
import h5py
import random
import tempfile
import numpy as np
from gensim import corpora, models
from scipy.sparse import lil_matrix
from sklearn.utils.sparsefuncs import mean_variance_axis

from .base import BaseTransform

from chemicalchecker.util import Config, logged
from chemicalchecker.core.signature_data import DataSignature


class Corpus(object):
    """Corpus class."""

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
        return len([_ for _ in self.keys()])

    def keys(self):
        for l in open(self.plain_corpus, "r"):
            key = l.split(" ")[0]
            l = l.rstrip("\n").split(" ")[1].split(",")
            bow = self.dictionary.doc2bow(l)
            if not bow:
                continue
            yield key


@logged
class Lsi(BaseTransform):
    """Lsi class."""

    def __init__(self, sign1, *args, tmp=False, variance_explained=0.9,
                 num_topics=None, B_val=10, N_val=1000, multipass=True,
                 min_freq=5, max_freq=0.25,
                 max_keys=100000, tmp_path=None, **kwargs):
        """Initialize a Lsi instance."""
        BaseTransform.__init__(self, sign1, "lsi", max_keys, tmp)
        self.variance_explained = variance_explained
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.multipass = multipass
        self.num_topics = num_topics
        self.B_val = B_val
        self.N_val = N_val
        if tmp_path is None:
            tmp_path = Config().PATH.CC_TMP
        self.tmp_path = tmp_path

    def _lsi_variance_explained(self, tfidf_corpus, lsi, num_topics):
        mm = corpora.MmCorpus(tfidf_corpus)
        exp_var_ratios = []
        for _ in range(self.B_val):
            xt = []
            sm = lil_matrix((self.N_val, mm.num_terms))
            for i in range(self.N_val):
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
            except Exception as ex:
                self.__log.warning(str(ex))
                continue
        exp_var_ratios = np.mean(np.array(exp_var_ratios), axis=0)
        return exp_var_ratios

    def fit(self):
        if not self.categorical:
            raise Exception("TFIDF-LSI only allowed for categorical matrices")
        V, keys, features = self.subsample()
        self.features = features
        self.plain_corpus = os.path.join(
            self.model_path, self.name + ".plain.txt")
        self.tfidf_corpus = os.path.join(
            self.model_path, self.name + ".tfidf.mm")
        # plain corpus
        with open(self.plain_corpus, "w") as f:
            for chunk in self.chunker(V.shape[0]):
                vs = V[chunk]
                ks = keys[chunk]
                for key, row in zip(ks, vs):
                    mask = np.where(row > 0)
                    val = ",".join([",".join([features[x]] * int(row[x]))
                                    for x in mask[0]])
                    f.write("%s %s\n" % (key, val))
        del V
        # get dictionary
        self.__log.info('Generating dictionary.')
        self.__log.info('min_freq: %s', self.min_freq)
        self.__log.info('max_freq: %s', self.max_freq)
        dictionary = corpora.Dictionary(l.rstrip("\n").split(" ")[1].split(
            ",") for l in open(self.plain_corpus, "r"))
        # filter extremes
        dictionary.filter_extremes(
            no_below=self.min_freq, no_above=self.max_freq)
        # save
        dictionary.compactify()
        dictionary.save(os.path.join(self.model_path, self.name + ".dict.pkl"))
        # corpus
        c = Corpus(self.plain_corpus, dictionary)
        # tfidf model
        tfidf = models.TfidfModel(c)
        tfidf.save(os.path.join(self.model_path, self.name + ".tfidf.pkl"))
        c_tfidf = tfidf[c]
        corpora.MmCorpus.serialize(self.tfidf_corpus, c_tfidf)
        # getting ready for lsi
        if self.num_topics is None:
            self.num_topics = int(0.67 * len(dictionary))
        if self.multipass:
            onepass = False
        else:
            onepass = True
        # lsi
        self.__log.info('Fitting LSI model.')
        only_zeros = 1
        while only_zeros > 0:
            self.__log.info('num_topics: %s', self.num_topics)
            lsi = models.LsiModel(c_tfidf, id2word=dictionary,
                                  num_topics=self.num_topics, onepass=onepass,
                                  chunksize=2500)
            lsi.save(os.path.join(self.model_path, self.name + ".lsi.pkl"))
            # variance explained
            exp_var_ratios = self._lsi_variance_explained(
                self.tfidf_corpus, lsi, self.num_topics)
            for cut_i, cum_var in enumerate(np.cumsum(exp_var_ratios)):
                if cum_var > self.variance_explained:
                    break
            self.cut_i = cut_i

            c_lsi = lsi[c_tfidf]
            # get keys
            keys = np.array([k for k in c.keys()])
            only_zeros = 0
            for line in c_lsi:
                v = np.zeros(self.cut_i + 1)
                for x in line[:self.cut_i + 1]:
                    if x[0] > self.cut_i:
                        continue
                    v[x[0]] = x[1]
                if np.sum(v) == 0:
                    only_zeros += 1
            # in some corner cases we might get full zero rows after LSI
            if only_zeros > 0:
                self.__log.warning(
                    'Getting only zero rows: %s', str(only_zeros))
                self.num_topics += 50
                self.variance_explained = min(
                    self.variance_explained + 0.05, 1)
                self.__log.warning(
                    'Repeating LSI with: variance_explained: %.2f num_topics: %s',
                    self.variance_explained, str(self.num_topics))
        self.predict(self.sign_ref)
        self.predict(self.sign)
        self.save()

    def predict(self, sign1):
        self.predict_check(sign1)
        # corpus for the predict
        tmp_dir = tempfile.mkdtemp(prefix="lsi_", dir=self.tmp_path)
        plain_corpus = os.path.join(tmp_dir, self.name + ".plain.txt")
        tfidf_corpus = os.path.join(tmp_dir, self.name + ".tfidf.mm")
        # write corpus (dense feature)
        with open(plain_corpus, "w") as f:
            # Read the provided sign1 by chunks of n signautres
            for chunk in sign1.chunker():
                # take a chunk of n signatures, together with their keys
                vs = sign1[chunk].astype(int)
                ks = sign1.keys[chunk]
                for i in range(0, len(ks)):
                    # save dense represantation (feat with 1 values only)
                    row = vs[i]
                    mask = np.argwhere(row > 0).ravel()
                    val = ",".join(self.features[mask])
                    f.write("%s %s\n" % (ks[i], val))
            sign1.close_hdf5()
        # load dictionary
        dictionary = corpora.Dictionary.load(
            os.path.join(self.model_path, self.name + ".dict.pkl"))
        # init corpus object
        c = Corpus(plain_corpus, dictionary)
        tfidf = models.TfidfModel.load(os.path.join(
            self.model_path, self.name + ".tfidf.pkl"))
        c_tfidf = tfidf[c]
        corpora.MmCorpus.serialize(tfidf_corpus, c_tfidf)
        lsi = models.LsiModel.load(os.path.join(
            self.model_path, self.name + ".lsi.pkl"))
        c_lsi = lsi[c_tfidf]
        # get keys
        keys = np.array([k for k in c.keys()])
        if len(keys) < len(sign1.keys):
            drop = len(sign1.keys) - len(keys)
            self.__log.warning('Dropped %s molecules (only zeros).' % drop)
        # instead of creating V we need to write iteratively to the H5
        # we run here the operations of the overwrite function
        with h5py.File(sign1.data_path, "r+") as hf:
            if self.tmp:
                del hf["V_tmp"]
                del hf["V"]
            else:
                del hf["V"]
                if "V_tmp" in hf.keys():
                    self.__log.debug("Overwriting tmp with the actual dataset")
                    del hf["V_tmp"]
            del hf["keys"]
            hf.create_dataset("keys", data=np.array(
                keys, DataSignature.string_dtype()))
            hf.create_dataset("V", (len(keys), self.cut_i + 1),
                              dtype=float)
            hf.create_dataset("V_tmp", (len(keys), self.cut_i + 1),
                              dtype=float)

            only_zeros = 0
            for idx, line in enumerate(c_lsi):
                v = np.zeros(self.cut_i + 1)
                for x in line[:self.cut_i + 1]:
                    if x[0] > self.cut_i:
                        continue
                    v[x[0]] = x[1]
                    if np.sum(v) == 0:
                        only_zeros += 1
                hf["V"][idx] = v
                hf["V_tmp"][idx] = v

            # in some corner cases we might get full zero rows after LSI
            if only_zeros > 0:
                self.__log.warning(
                    'Getting only zero rows: %s', str(only_zeros))

        sign1.refresh()
        self.reindex_triplets(sign1, keys)
        self.remap(sign1)

        # self.overwrite(sign1=sign1, V=V, keys=keys)
