"""Do TFIDF-LSI."""
import os
import random
import tempfile
import numpy as np
from gensim import corpora, models
from scipy.sparse import lil_matrix
from sklearn.utils.sparsefuncs import mean_variance_axis

from .base import BaseTransform

from chemicalchecker.util import Config


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


class Lsi(BaseTransform):
    """Lsi class."""

    def __init__(self, sign1, tmp=False, variance_explained=0.9, min_freq=5, max_freq=0.25,
                 num_topics=None, B_val=10, N_val=1000, multipass=True,
                 max_keys=100000, **kwargs):
        """Initialize a Lsi instance."""
        BaseTransform.__init__(self, sign1, "lsi", max_keys, tmp)
        self.variance_explained = variance_explained
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.multipass = multipass
        self.num_topics = num_topics
        self.B_val = B_val
        self.N_val = N_val

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
            except:
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
        # get dictionary
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
            num_topics = np.min([int(0.67 * len(dictionary)), 5000])
        else:
            num_topics = self.num_topics
        if self.multipass:
            onepass = False
        else:
            onepass = True
        # lsi
        lsi = models.LsiModel(
            c_tfidf, id2word=dictionary, num_topics=num_topics, onepass=onepass)
        lsi.save(os.path.join(self.model_path, self.name + ".lsi.pkl"))
        c_lsi = lsi[c_tfidf]
        # variance explained
        exp_var_ratios = self._lsi_variance_explained(
            self.tfidf_corpus, lsi, num_topics)
        for cut_i, cum_var in enumerate(np.cumsum(exp_var_ratios)):
            if cum_var > self.variance_explained:
                break
        self.cut_i = cut_i
        self.predict(self.sign_ref)
        self.predict(self.sign)
        self.save()

    def predict(self, sign1):
        self.predict_check(sign1)
        # corpus for the predict
        tmp_dir = tempfile.mkdtemp(prefix="lsi_", dir=Config().PATH.CC_TMP)
        plain_corpus = os.path.join(tmp_dir, self.name + ".plain.txt")
        tfidf_corpus = os.path.join(tmp_dir, self.name + ".tfidf.mm")
        with open(plain_corpus, "w") as f:
            for chunk in sign1.chunker():
                vs = sign1[chunk].astype(np.int)
                ks = sign1.keys[chunk]
                for i in range(0, len(ks)):
                    row = vs[i]
                    mask = np.where(row > 0)
                    val = ",".join([",".join([self.features[x]] * row[x])
                                    for x in mask[0]])
                    f.write("%s %s\n" % (ks[i], val))
        # load dictionary
        dictionary = corpora.Dictionary.load(
            os.path.join(self.model_path, self.name + ".dict.pkl"))
        # corpus
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
        V = np.empty((len(keys), self.cut_i + 1))
        i = 0
        for l in c_lsi:
            v = np.zeros(self.cut_i + 1)
            for x in l[:self.cut_i + 1]:
                if x[0] > self.cut_i:
                    continue
                v[x[0]] = x[1]
            V[i, :] = v
            i += 1
        self.overwrite(sign1=sign1, V=V, keys=keys)
