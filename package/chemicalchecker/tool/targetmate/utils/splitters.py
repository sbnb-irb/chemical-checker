import collections
import random
import numpy as np
from sklearn import model_selection

from chemicalchecker.util import logged

from .chemistry import generate_scaffold

TRIALS = 3

class Splitter(object):

    def __init__(self, n_splits=1, test_size=0.2, random_state=42, train_size=None, **kwargs):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size=train_size
        if random_state is None:
            self.random_state = random.randint(1, 99999)
        else:
            self.random_state = random_state

    def calc_sizes(self, n):
        if self.test_size < 1:
            self.n_test  = int(n*self.test_size)
        else:
            self.n_test  = int(test_size)
        self.n_train = n - self.n_test
        self.prop    = self.n_test / n
        self.n = n


def generate_scaffolds(smiles):
    scaffolds = collections.defaultdict(list)
    for i, smi in enumerate(smiles):
        scaffold = generate_scaffold(smi)
        scaffolds[scaffold] += [i]
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    return scaffolds


@logged
class ToppedSampler(object):
    """Sample so that coverage is maximized."""

    def __init__(self, max_samples, max_ensemble_size, chance, try_balance, shuffle, brute=True):
        """Initialize the topped sampler.

        Args:
            max_samples(int): Maximum number of samples allowed per draw.
            max_ensemble_size(int): Maximum number of draws.
            chance(float): Desired probability of drawing a sample at least once.
            try_balance(bool): Try to balance, given the available samples. That is, instead of stratifying, give higher probability to the minority class.
            shuffle(bool): Shuffle indices.
            brute(bool): When trying to balance, be brute and do not sample by probability (default=True).
        """
        self.max_samples = max_samples
        self.max_ensemble_size = max_ensemble_size
        self.chance = chance
        self.try_balance = try_balance
        self.shuffle = shuffle
        self.brute = brute
        self.min_ensemble_size = 3

    @staticmethod
    def get_resamp(s, N, chance):
        p = 1 - float(s) / N
        return np.log(1-chance)/np.log(p)

    def calc_ensemble_size(self):
        if self.N <= self.max_samples:
            self.samples = self.N
            self.ensemble_size = 1
        else:
            self.samples = self.max_samples
            self.ensemble_size = int(
                np.ceil(self.get_resamp(self.samples, self.N, self.chance)))
            self.ensemble_size = np.min([self.max_ensemble_size, self.ensemble_size])
            self.ensemble_size = np.max([self.min_ensemble_size, self.ensemble_size])

    @staticmethod
    def probabilities(y, bins):
        h, b = np.histogram(y, bins)
        min_b = np.min(b)
        max_b = np.max(b)
        b = b[:-1]
        m = (max_b-min_b)/(np.max(b)-np.min(b))
        n = min_b - m*np.min(b)
        b = m*b+n
        w = np.interp(y, b, h).ravel()
        w = (1 - w/np.sum(h))+1e-6
        probas = w/np.sum(w)
        return probas

    def brute_sample(self, y):
        if len(y) <= self.max_samples:
            idx = np.array([i for i in range(0, len(y))], dtype=int)
            np.random.shuffle(idx)
            return idx
        y0_idx = np.argwhere(y == 0).ravel()
        y1_idx = np.argwhere(y == 1).ravel()
        n0 = len(y0_idx)
        n1 = len(y1_idx)
        max_minority = int(self.max_samples/2)
        if n0 >= n1:
            self.__log.info("0 >= 1 : %d %d" % (n0, n1))
            if n1 > max_minority:
                y1_idx = np.random.choice(y1_idx, max_minority, replace=False)
            n = self.max_samples - len(y1_idx)
            if n0 > n:
                y0_idx = np.random.choice(y0_idx, n, replace=False)
        else:
            self.__log.info("0 < 1 : %d %d" % (n0, n1))
            if n0 > max_minority:
                y0_idx = np.random.choice(y0_idx, max_minority, replace=False)
            n = self.max_samples - len(y0_idx)
            if n1 > n:
                y1_idx = np.random.choice(y1_idx, n, replace=False)
        idx = list(y0_idx) + list(y1_idx)
        idx = np.array(idx, dtype=int)
        np.random.shuffle(idx)
        self.__log.info("...1: %d total: %d" % (np.sum(y[idx]), len(idx)))
        return idx

    def ret(self, idxs):
        if self.shuffle:
            return idxs
        else:
            idxs_ = np.argsort(idxs)
            return idxs[idxs_]

    def sample(self, X=None, y=None, bins=10):
        """Main method"""
        self.__log.info("Topped Sampling | max samples %d" % (self.max_samples))
        if X is None and y is None:
            raise Exception("X and y are None")
        if X is None:
            self.N = len(y)
        else:
            self.N = X.shape[0]
        if self.N <= self.max_samples:
            self.__log.info("...topped sampling not necessary")
            idxs = np.array(range(len(y)))
            if self.shuffle:
                random.shuffle(idxs)
            yield self.ret(idxs)
        else:
            self.calc_ensemble_size()
            if self.try_balance:
                if y is None:
                    for _ in range(0, self.ensemble_size):
                        idxs = np.random.choice(self.N, self.samples, replace=False)
                        yield self.ret(idxs)
                else:
                    for _ in range(0, self.ensemble_size):
                        if self.brute:
                            idxs = self.brute_sample(y)
                        else:
                            probas = self.probabilities(y, bins=bins)
                            idxs = np.random.choice([i for i in range(0,self.N)], self.samples, p=probas, replace=False)
                        yield self.ret(idxs)
            else:
                if y is None:
                    splits = ShuffleSplit(
                        n_splits=self.ensemble_size, train_size=self.samples)
                else:
                    splits = StratifiedShuffleSplit(
                        n_splits=self.ensemble_size, train_size=self.samples)
                for split in splits:
                    yield self.ret(split[0])


@logged
class OutOfUniverseStratified(Splitter):
    """In a stratified skaffold split, only molecules that are not present in the universe are accepted at testing time"""

    def __init__(self, cc=None, datasets = ["A1.001"],
                 cctype="sign1", inchikeys_universe=None, **kwargs):
        """Initialize.

            Args:
                cc(ChemicalChecker): ChemicalChecker instance.
                datasets_universe(list): Datasets to consider as the universe.
                cctype(str): Signature type corresponding to the datasets.
                inchikeys_universe(list): List of inchikeys to be considered as the universe. If not None, overwrites datasets (default=None).
        """
        Splitter.__init__(self, **kwargs)
        if inchikeys_universe is None:
            if cc is None:
                raise Exception("ChemicalChecker instance need be specified")
            univ = set()
            for ds in datasets:
                s = cc.signature(ds, cctype)
                univ.update(list(s.keys))
            univ = list(univ)
        else:
            univ = inchikeys_universe
        self.univ = set([k.split("-")[0] for k in univ])

    def one_split(self, ins_idxs, out_idxs, y, seed):
        assert set(y) == set([0,1]), "Labels are different than 0/1"
        y = np.array(y)
        # Expected numbers
        exp_n_te   = int(np.round(len(y)*self.test_size, 0))
        exp_n_tr   = len(y) - exp_n_te
        prop_1_0   = np.sum(y) / len(y)
        exp_n_tr_0 = max(1, int(np.round(exp_n_tr*(1-prop_1_0),0)))
        exp_n_tr_1 = max(1, int(np.round(exp_n_tr*(prop_1_0),0)))
        exp_n_te_0 = max(1, int(np.round(exp_n_te*(1-prop_1_0),0)))
        exp_n_te_1 = max(1, int(np.round(exp_n_te*(prop_1_0),0)))
        # Masks
        ins_mask   = np.array([False]*len(y))
        out_mask   = np.array([False]*len(y))
        ins_mask[ins_idxs] = True
        out_mask[out_idxs] = True
        mask_0 = y == 0
        mask_1 = y == 1
        # Check availability
        obs_ins_0   = np.argwhere(np.logical_and(ins_mask, mask_0)).ravel()
        obs_ins_1   = np.argwhere(np.logical_and(ins_mask, mask_1)).ravel()
        obs_out_0   = np.argwhere(np.logical_and(out_mask, mask_0)).ravel()
        obs_out_1   = np.argwhere(np.logical_and(out_mask, mask_1)).ravel()
        obs_ins_n_0 = len(obs_ins_0)
        obs_ins_n_1 = len(obs_ins_1)
        obs_out_n_0 = len(obs_out_0)
        obs_out_n_1 = len(obs_out_1)
        if obs_out_n_1 == 0 or obs_out_n_0 == 0:
            self.__log.warn("Unfortunately, not enough out-of-universe samples are available")
            return None, None
        # Decide how to sample
        #  start with test set
        self.__log.info("... sampling test first")
        test_idxs = []
        #    negatives
        if obs_out_n_0 >= exp_n_te_0:
            self.__log.info("More 0s than needed, subsampling")
            n = exp_n_te_0
            np.random.seed(seed)
            test_idxs += list(np.random.choice(obs_out_0, size=n, replace=False))
        else:
            self.__log.info("Less 0s than needed, that's ok...")
            test_idxs += list(obs_out_0)
        #    positives
        if obs_out_n_1 >= exp_n_te_1:
            self.__log.info("More 1s than needed, subsampling")
            n = exp_n_te_1
            np.random.seed(seed)
            test_idxs += list(np.random.choice(obs_out_1, size=n, replace=False))
        else:
            self.__log.info("Less 1s than needed, that's ok...")
            test_idxs += list(obs_out_1)
        # rebalance if necessary
        self.__log.debug("Re-balance if necessary")
        test_prop_1_0 = np.sum(y[test_idxs])/len(test_idxs)
        self.__log.info("Observed test prop 1/0: %.3f, full prop 1/0: %.3f" % (test_prop_1_0, prop_1_0))
        idxs_0 = np.argwhere(y == 0).ravel()
        idxs_1 = np.argwhere(y == 1).ravel()
        test_idxs_0 = list(set(test_idxs).intersection(idxs_0))
        test_idxs_1 = list(set(test_idxs).intersection(idxs_1))
        if test_prop_1_0 > prop_1_0:
            n_1 = int(np.round(len(test_idxs)*prop_1_0,0))
            np.random.seed(seed)
            test_idxs_1 = np.random.choice(test_idxs_1, size=min(len(test_idxs_1), n_1), replace=False)
        else:
            n_0 = int(np.round(len(test_idxs)*(1-prop_1_0),0))
            np.random.seed(seed)
            test_idxs_0 = np.random.choice(test_idxs_0, size=min(len(test_idxs_0), n_0), replace=False)
        test_idxs = list(test_idxs_0) + list(test_idxs_1)
        if len(test_idxs_0) == 0 or len(test_idxs_1) == 0:
            return None, None
        #  continue with train set
        self.__log.info("... sampling train now")
        train_idxs  = []
        #    negatives
        if obs_ins_n_0 >= exp_n_tr_0:
            self.__log.info("More 0s than needed, subsampling")
            n = exp_n_tr_0
            np.random.seed(seed)
            train_idxs += list(np.random.choice(obs_ins_0, size=n, replace=False))
        else:
            self.__log.info("Less 0s than needed, sampling from out-of-universe")
            train_idxs += list(obs_ins_0)
            n = exp_n_tr_0 - obs_out_n_0
            eligible = list(set(obs_out_0).difference(test_idxs))
            n = min(n, len(eligible))
            if n > 0:
                np.random.seed(seed)
                train_idxs += list(np.random.choice(eligible, size=min(n, len(eligible)), replace=False))
        #    positives
        if obs_ins_n_1 >= exp_n_tr_1:
            self.__log.info("More 1s than needed, subsampling")
            n = exp_n_tr_1
            np.random.seed(seed)
            train_idxs += list(np.random.choice(obs_ins_1, size=n, replace=False))
        else:
            self.__log.info("Less 1s than needed, sampling from out-of-universe")
            train_idxs += list(obs_ins_1)
            n = exp_n_tr_1 - obs_ins_n_1
            eligible = list(set(obs_out_1).difference(test_idxs))
            n = min(n, len(eligible))
            if n > 0:
                np.random.seed(seed)
                train_idxs += list(np.random.choice(eligible, size=n, replace=False))
        # Sort
        train_idxs = np.array(train_idxs).astype(int)
        test_idxs  = np.array(test_idxs).astype(int)
        np.random.seed(seed)
        np.random.shuffle(train_idxs)
        np.random.seed(seed)
        np.random.shuffle(test_idxs)
        return train_idxs, test_idxs

    def split(self, X, y):
        seed = self.random_state
        inchikeys = X
        inchikeys_conn = np.array([k.split("-")[0] for k in inchikeys])
        ins_idxs = []
        out_idxs = []
        for i, k in enumerate(inchikeys_conn):
            if k in self.univ:
                ins_idxs += [i]
            else:
                out_idxs += [i]
        ins_idxs = np.array(ins_idxs)
        out_idxs = np.array(out_idxs)
        self.__log.info("Inside universe: %d, Outside universe: %d" % (len(ins_idxs), len(out_idxs)))
        for _ in range(0, self.n_splits):
            train_idx, test_idx = self.one_split(ins_idxs, out_idxs, y, seed)
            seed += 1
            yield train_idx, test_idx


@logged
class ShuffleScaffoldSplit(Splitter):
    """Random sampling based on scaffolds. It tries to satisfy the desired proportion"""

    def __init__(self, trials=TRIALS, **kwargs):
        Splitter.__init__(self, **kwargs)
        self.trials = trials

    def one_split(self, smiles, seed):
        self.calc_sizes(len(smiles))
        scaffolds = generate_scaffolds(smiles)
        scaffolds = [scaffolds[k] for k in sorted(scaffolds.keys())]
        cur_train_idx = None
        cur_test_idx  = None
        cur_prop      = None
        for _ in range(0, self.trials):
            train_idx = []
            test_idx  = []
            scaff_idxs = [i for i in range(0, len(scaffolds))]
            random.seed(seed)
            random.shuffle(scaff_idxs)
            seed += 1
            for scaff_idx in scaff_idxs:
                idxs = scaffolds[scaff_idx]
                if len(train_idx) + len(idxs) > self.n_train:
                    test_idx += idxs
                else:
                    train_idx += idxs
            prop = len(test_idx) / self.n
            if cur_train_idx is None:
                cur_train_idx = train_idx
                cur_test_idx  = test_idx
                cur_prop      = prop
            else:
                if np.abs(prop - self.prop) < np.abs(cur_prop - self.prop):
                    cur_train_idx = train_idx
                    cur_test_idx  = test_idx
                    cur_prop      = prop
        train_idx = np.array(cur_train_idx).astype(int)
        test_idx  = np.array(cur_test_idx).astype(int)
        random.seed(seed)
        random.shuffle(train_idx)
        random.seed(seed)
        random.shuffle(test_idx)
        self.__log.info("Train size: %d, Test size: %d, Prop : %.2f" % (len(train_idx), len(test_idx), cur_prop))
        return train_idx, test_idx

    def split(self, X, y = None):
        seed = self.random_state
        smiles = X
        for _ in range(0, self.n_splits):
            train_idx, test_idx = self.one_split(smiles, seed)
            seed += 1
            yield train_idx, test_idx

@logged
class StratifiedShuffleScaffoldSplit(Splitter):
    """Stratified shuffle split based on scaffolds.
    It tries to preserve the proportion of samples in the train and test sets.
    The current algorithm is very rough..."""

    def __init__(self, trials=TRIALS, **kwargs):
        ShuffleScaffoldSplit.__init__(self, trials=10, **kwargs)
        self.trials = trials

    def one_split(self, smiles, y, seed):
        assert set(y) == set([0,1]), "Labels are different than 0/1"
        y = np.array(y)
        self.calc_sizes(len(smiles))
        scaffolds = generate_scaffolds(smiles)
        scaffolds = [scaffolds[k] for k in sorted(scaffolds.keys())]
        balance   = np.sum(y) / len(y)
        cur_train_idx     = None
        cur_test_idx      = None
        cur_prop          = None
        cur_balance_train = None
        for _ in range(0, self.trials):
            train_idx = []
            test_idx  = []
            scaff_idxs = [i for i in range(0, len(scaffolds))]
            random.seed(seed)
            random.shuffle(scaff_idxs)
            seed += 1
            for scaff_idx in scaff_idxs:
                idxs = scaffolds[scaff_idx]
                if len(train_idx) + len(idxs) > self.n_train:
                    test_idx += idxs
                else:
                    train_idx += idxs
            prop = len(test_idx) / self.n
            balance_train = np.sum(y[train_idx]) / len(train_idx)
            if cur_train_idx is None:
                cur_train_idx     = train_idx
                cur_test_idx      = test_idx
                cur_prop          = prop
                cur_balance_train = balance_train
            else:
                if (np.abs(balance - balance_train) + np.abs(prop - self.prop)) < (np.abs(balance - cur_balance_train) + np.abs(cur_prop - self.prop)):
                    cur_train_idx     = train_idx
                    cur_test_idx      = test_idx
                    cur_prop          = prop
                    cur_balance_train = balance_train
        train_idx = np.array(cur_train_idx).astype(int)
        test_idx  = np.array(cur_test_idx).astype(int)
        random.seed(seed)
        random.shuffle(train_idx)
        random.seed(seed)
        random.shuffle(test_idx)
        self.__log.info("Train size: %d, Test size: %d, Prop: %.2f, Balance: %.2f (%.2f)" % (len(train_idx), len(test_idx), cur_prop, cur_balance_train, balance))
        return train_idx, test_idx

    def split(self, X, y):
        seed = self.random_state
        smiles = X
        for _ in range(0, self.n_splits):
            train_idx, test_idx = self.one_split(smiles, y, seed)
            seed += 1
            yield train_idx, test_idx


@logged
class DeepchemScaffoldSplit(Splitter):
    """Analogous to DeepChem implementation. First it sorts by scaffold set size."""

    def __init__(self, **kwargs):
        Splitter.__init__(self, **kwargs)

    def split(self, X, y=None):
        seed = self.random_state
        smiles = X
        self.calc_sizes(len(smiles))
        scaffolds = generate_scaffolds(smiles)
        scaffolds = [v for k,v in sorted(scaffolds.items(), key = lambda x: (len(x[1]), x[1][0]), reverse=True)]
        train_idx = []
        test_idx  = []
        for idxs in scaffolds:
            if len(train_idx) + len(idxs) > self.n_train:
                test_idx  += idxs
            else:
                train_idx += idxs
        train_idx = np.array(train_idx).astype(int)
        test_idx  = np.array(test_idx).astype(int)
        random.seed(seed)
        random.shuffle(train_idx)
        random.seed(seed)
        random.shuffle(test_idx)
        prop = len(test_idx) / (len(test_idx) + len(train_idx))
        self.__log.info("Train size: %d, Test size: %d, prop: %.2f" % (len(train_idx), len(test_idx), prop))
        yield train_idx, test_idx


@logged
class StandardStratifiedShuffleSplit(Splitter):
    """A wrapper for the sklearn stratified shuffle split"""

    def __init__(self, **kwargs):
        self.__log.info("Standard stratified shuffle splitter")
        Splitter.__init__(self, **kwargs)

    def split(self, X, y):
        spl = model_selection.StratifiedShuffleSplit(n_splits=self.n_splits,
                                                     test_size=self.test_size,
                                                     random_state=self.random_state)
        for train_idx, test_idx in spl.split(X=X, y=y):
            yield train_idx, test_idx


def GetSplitter(is_cv, is_classifier, is_stratified, scaffold_split, outofuniverse_split):
    """Select the splitter, depending on the characteristics of the problem"""
    if is_classifier:
        if is_cv:
            if is_stratified:
                spl = model_selection.StratifiedKFold
            else:
                spl = model_selection.KFold
        else:
            if outofuniverse_split:
                spl = OutOfUniverseStratified
            else:
                if scaffold_split:
                    if is_stratified:
                        spl = StratifiedShuffleScaffoldSplit
                    else:
                        spl = ShuffleScaffoldSplit
                else:
                    if is_stratified:
                        spl = StandardStratifiedShuffleSplit
                    else:
                        spl = model_selection.ShuffleSplit
    else:
        # TO-DO
        pass
    return spl
