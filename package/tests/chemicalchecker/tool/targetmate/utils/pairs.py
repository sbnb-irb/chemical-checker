"""Obtain pairs given an incomplete dataset"""
from chemicalchecker.util import logged
import collections
import numpy as np
import random
import h5py
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
import os
import pickle
from scipy.stats import rankdata

@logged
class Pairs:
    """Pairs class samples/undersamples to accomplish a certain proportion of negative-positive"""
    def __init__(self, neg_pos_ratio=100, max_pos=1000, primary_side="right", test_size=0.2, n_splits=1, random_state=None):
        """Initialize Pairs class

        Args:
            neg_pos_ratio(float): Expected number of negatives per positives (default=10).
            primary_side(str): When doing the sampling, focus on balancing 'right' or 'left' (default='right').
            max_pos(int): Maximum number of positives to take into account (default=1000).
            test_size(float): When splitting, proportion of test samples (default=0.2).
            n_splits(int): When splitting, number of runs (default=1).
            random_state(int): Random state (default=None).
        """
        self.neg_pos_ratio = neg_pos_ratio
        self.max_pos = max_pos
        self.primary_side = primary_side
        if self.primary_side == "right":
            self.primary_idx = 1
            self.secondary_idx = 0
        elif self.primary_side == "left":
            self.primary_idx = 0
            self.secondary_idx = 1
        else:
            self.__log.error("Argument primary_side must be 'right' or 'left'")
            raise
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state

    def _calc_num_neg(self, num_pos):
        return int(num_pos*self.neg_pos_ratio) + 1

    def _choose(self, v, size, p):
        return np.random.choice(list(v), size=size, replace=False, p=p)

    def _tupler(self, k, v, y):
        def to_tuple(k, x):
            if self.primary_idx == 1:
                return (x, k, y)
            else:
                return (k, x, y)
        for x in v:
            yield to_tuple(k, x)

    def index_pairs(self, pairs, keys_left, keys_right):
        self.keys_left  = keys_left
        self.keys_right = keys_right
        self.__log.debug("Filtering pairs (only those being in the list are accepted)")
        kl = dict((k, i) for i, k in enumerate(keys_left))
        kr = dict((k, i) for i, k in enumerate(keys_right))
        self.known_pairs = []
        for p in pairs:
            if p[0] not in kl: continue
            if p[1] not in kr: continue
            self.known_pairs += [(kl[p[0]], kr[p[1]], p[2])]
        self.known_pairs = list(set(self.known_pairs))
        self.known_pairs = np.array(self.known_pairs).astype(int)
        self.__log.debug("Original pairs: %d, Remaining pairs: %d" % (len(pairs), len(self.known_pairs)))
        self.__log.debug("Original left : %d, Remaining left : %d" % (len(set([p[0] for p in pairs])), len(set(self.known_pairs[:,0]))))
        self.__log.debug("Original right: %d, Remaining right: %d" % (len(set([p[1] for p in pairs])), len(set(self.known_pairs[:,1]))))

    def sample_left(self, pairs, keys_left, keys_right, max_pos, bioteque_priors):
        from tqdm import tqdm
        self.index_pairs(pairs, keys_left, keys_right)
        #self.left_known_pairs_to_probabilities(max_size=max_size, pickle_filename=pickle_filename)
        self.__log.debug("Making sure each left instance has the same number of positives")
        A = collections.defaultdict(list)
        I = collections.defaultdict(list)
        for i in range(0, len(self.known_pairs)):
            p = self.known_pairs[i]
            if p[-1] == 1:
                A[p[0]] += [p[1]]
            else:
                I[p[0]] += [p[1]]
        A = dict((k, set(v)) for k,v in A.items())
        I = dict((k, set(v)) for k,v in I.items())
        if bioteque_priors:
            self.__log.debug("Getting bioteque priors")
            priors_dict = {}
            with open("/aloy/home/mduran/myscripts/dream-ctd2-targetmate/data/bioteque_priors.tsv", "r") as f:
                import csv
                reader = csv.reader(f, delimiter="\t")
                for r in reader:
                    priors_dict[r[0]] = float(r[1])
            priors = np.zeros(len(keys_right))
            priors_inv = np.zeros(len(keys_right))
            for i, p in enumerate(keys_right):
                if p in priors_dict:
                    priors[i] = priors_dict[p]
                    priors_inv[i] = 1 - priors_dict[p]
            #priors_inv = 1 - priors
        else:
            self.__log.debug("Getting priors for each on the right")
            priors = np.zeros(len(keys_right))
            for k,v in A.items():
                for i in list(v):
                    priors[i] += 1
            priors     = rankdata(priors)
            priors_inv = rankdata(-priors)
        #priors     += 1e-10
        #priors_inv += 1e-10
        from scipy.stats import pearsonr
        self.__log.debug("Priors and priors-inverse are correct (pearson %.2f)" % pearsonr(priors, priors_inv)[0])
        self.__log.debug("Oversampling positives %d" % max_pos)
        A_ = {}
        for k,v in tqdm(A.items()):
            v = list(v)
            p = priors[v]
            if np.sum(p)==0: continue
            p = p/np.sum(p)
            A_[k] = np.random.choice(v, size=max_pos, replace=True, p=p)
        self.__log.debug("Now sample the negatives (%d ratio)" % self.neg_pos_ratio)
        I_ = {}
        universe = set([i for i in range(0, len(keys_right))])
        for k,v in tqdm(A.items()):
            if k not in A_: continue
            if k in I:
                n = int(max_pos*self.neg_pos_ratio - len(I[k]))
            else:
                n = int(max_pos*self.neg_pos_ratio)
            v_ = list(universe.difference(v))
            p = priors_inv[v_]
            if np.sum(p) == 0: continue
            p = p/np.sum(p)
            I_[k] = np.random.choice(v_, size=n, replace=True, p=p)
        self.__log.debug("Done. Now just appending")
        pairs = []
        from tqdm import tqdm
        import uuid
        tag = str(uuid.uuid4())
        with open(tag, "w") as f:
            for k,v in tqdm(A_.items()):
                for x in v:
                    pairs += [(k, x, 1)]
                    #f.write("%d\t%d\t%d\n" % (k, x, 1))
            for k,v in tqdm(I_.items()):
                for x in v:
                    #f.write("%d\t%d\t%d\n" % (k, x, 1))
                    pairs += [(k, x, 0)]
        #return 
        pairs = np.array(pairs, dtype=int)
        self.__log.debug("Shuffling pairs")
        idxs = np.array([i for i in range(0, pairs.shape[0])], dtype=int)
        random.shuffle(idxs)
        self.pairs = pairs[idxs]
        self.__log.debug("Number of pairs: %d" % len(self.pairs))

    def sample_random(self, pairs, keys_left, keys_right):
        self.index_pairs(pairs, keys_left, keys_right)
        self.__log.debug("Sampling to achieve a negative:positive balance of %d" % self.neg_pos_ratio)
        done = set([(self.known_pairs[i,0], self.known_pairs[i,1]) for i in range(0, self.known_pairs.shape[0])])   
        n_pos = np.sum(self.known_pairs[:,-1]==1)
        n_neg = np.sum(self.known_pairs[:,-1]==0)
        to_sample = int(n_pos*self.neg_pos_ratio) - n_neg
        self.__log.debug("Known positives: %d, Known negatives: %d, Negatives to sample: %d" % (n_pos, n_neg, to_sample))
        L = len(keys_left)
        R = len(keys_right)
        sampled = set()
        for _ in range(0, to_sample*10):
            pair = (np.random.choice(L), np.random.choice(R))
            if pair in done: continue
            if pair in sampled: continue
            sampled.update([pair])
            if len(sampled)/1000000 == int(len(sampled)/1000000):
                self.__log.debug("%d negative pairs sampled (%.2f)" % (len(sampled), len(sampled)/to_sample))
            if len(sampled) > to_sample:
                break
        self.__log.debug("Done with the sampling. Merging and shuffling.")
        pairs = []
        for i in range(0, self.known_pairs.shape[0]):
            pairs += [(self.known_pairs[i,0], self.known_pairs[i,1], self.known_pairs[i,2])]
        for s in list(sampled):
            pairs += [(s[0],s[1],0)]
        pairs = np.array(pairs, dtype=int)
        idxs = np.array([i for i in range(0, pairs.shape[0])], dtype=int)
        random.shuffle(idxs)
        self.pairs = pairs[idxs]
        self.__log.debug("Number of pairs: %d" % len(self.pairs))

    def sample_balanced(self, pairs, keys_left, keys_right):
        """Sample from the known pairs to obtain a longer list

        Args:
            pairs(list): List of (key_left, key_right, 1/0) values.
            keys_left(list): Keys universe of the left side.
            keys_right(list): Keys universe of the right side.
        """
        self.index_pairs(pairs, keys_left, keys_right)
        self.__log.debug("Sampling to achieve a negative:positive balance of %d" % self.neg_pos_ratio)
        samp_counts = collections.defaultdict(int)
        A = collections.defaultdict(list)
        I = collections.defaultdict(list)
        for p in self.known_pairs:
            if p[-1] == 1:
                A[p[self.primary_idx]] += [p[self.secondary_idx]]
            else:
                I[p[self.primary_idx]] += [p[self.secondary_idx]]
        A = dict((k, np.array(v)) for k,v in A.items())
        I = dict((k, np.array(v)) for k,v in I.items())
        self.__log.debug("Looking for overabundant data")
        max_A = self.max_pos
        max_I = self._calc_num_neg(self.max_pos)
        probas_A = None
        probas_I = None
        for k,v in A.items():
            if len(v) > max_A:
                A[k] = self._choose(v, size=max_A, p=probas_A)
                self.__log.debug("Subsampling actives from %d, before %d, now %d" % (k, len(v), len(A[k])))
        for k,v in I.items():
            if k in A:
                _max_I = np.min([self._calc_num_neg(len(A[k])), max_I])
            else:
                _max_I = max_I
            if len(v) > _max_I:
                I[k] = self._choose(v, size=_max_I, p=probas_I)
        self.__log.debug("Oversampling negative class")
        universe = set([x for d in [A, I] for k,v in d.items() for x in v])
        self.__log.debug("Universe has %d entities" % len(universe))
        from tqdm import tqdm
        #i = 0
        ks = []
        for k, v in tqdm(A.items()):
            ## REMOVE THIS
            #i += 1
            #if i > 10: break
            n = self._calc_num_neg(len(v))
            sampling_universe = universe.difference(v)
            n = np.min([len(sampling_universe), n])
            if k in I:
                n -= len(I[k])
                if n > 0:
                    sampling_universe = sampling_universe.difference(I[k])
                    n = np.min([len(sampling_universe), n])
                    if n > 0:
                        samp = self._choose(sampling_universe, size=n, p=probas_I)
                        I[k] = np.concatenate([I[k], samp])
            else:
                I[k] = self._choose(sampling_universe, size=n, p=probas_I)
            ks += [k]
        self.__log.debug("Assembling pairs")
        self.pairs = []
        for k in ks:
            self.pairs += [t for t in self._tupler(k, A[k], 1)]
            self.pairs += [t for t in self._tupler(k, I[k], 0)]
        self.pairs = np.array(self.pairs).astype(int)
        self.__log.debug("Shuffling")             
        idxs = np.array([i for i in range(0, self.pairs.shape[0])])
        random.shuffle(idxs)
        self.pairs = self.pairs[idxs]
        self.__log.debug("Number of pairs: %d" % len(self.pairs))

    def _get_y(self):
        return np.array([p[-1] for p in self.pairs])

    def _to_dict(self, key_col):
        d = collections.defaultdict(list)
        if key_col == 1:
            val_col = 0
        else:
            val_col = 1
        for i in range(0, self.pairs.shape[0]):
            d[self.pairs[i, key_col]] += [(self.pairs[i, val_col], self.pairs[i, 2])]
        return d

    def _column_split(self, col):
        def to_tuple(k, x, y):
            if col == 1:
                return (x, k, y)
            else:
                return (k, x, y)
        def appender(t_idx, idxs, d):
            t = []
            for idx in t_idx:
                k = idxs[idx]
                for x in d[k]:
                    t += [to_tuple(k, x[0], x[1])]
            idxs_ = [i for i in range(0, len(t))]
            random.shuffle(idxs_)
            return np.array(t).astype(int)[idxs_]
        idxs = list(set([p[col] for p in self.pairs]))
        d = self._to_dict(col)
        spl = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
        for tr_idx, te_idx in spl.split(X=idxs):
            train = appender(tr_idx, idxs, d)
            test  = appender(te_idx, idxs, d)
            yield train, test

    def as_indices(self):
        """Pairs iterator, returns indices"""
        for i in range(0, self.pairs.shape[0]):
            yield tuple(pairs[i])

    def as_keys(self):
        """Pairs iterator, remaps to keys"""
        for i in range(0, self.pairs.shape[0]):
            yield self.keys_left[self.pairs[i,0]], self.keys_right[self.pairs[i,1]], self.pairs[i,2]

    def naive_split(self):
        """Split pairs randomly"""
        spl = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
        for train_idx, test_idx in spl.split(X=self.pairs, y=self._get_y()):
            yield self.pairs[train_idx], self.pairs[test_idx]

    def right_split(self):
        """Split pairs by right side"""
        return self._column_split(col=1)

    def left_split(self):
        """Split pairs by left side"""
        return self._column_split(col=0)

    def left_right_split(self):
        """Split pairs by left and right sides"""
        ### TO DO
        return

    def save_h5(self, filename):
        self.__log.debug("Saving to %s" % filename)
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("pairs", data=self.pairs)


def onehot_proteins_signature():
    from chemicalchecker.core.signature_data import DataSignature
    import csv
    import h5py
    with open("/aloy/home/mduran/myscripts/dream-ctd2-targetmate/data/bioteque_priors.tsv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        keys = []
        for r in reader:
            keys += [r[0]]
    keys = np.array(keys)
    V = np.identity(len(keys)).astype(int)
    with h5py.File("/aloy/home/mduran/myscripts/dream-ctd2-targetmate/paired_targetmate/X_r_1h.h5", "w") as hf:
        hf.create_dataset("keys", data=np.array(keys, DataSignature.string_dtype()))
        hf.create_dataset("V", data=V)
