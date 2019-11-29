import collections
import random
import numpy as np
from sklearn import model_selection

from chemicalchecker.util import logged

from .chemistry import generate_scaffold


class Splitter(object):

    def __init__(self, n_splits=1, test_size=0.2, random_state=42, **kwargs):
        self.n_splits = n_splits
        self.test_size = test_size
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
class ShuffleScaffoldSplit(Splitter):
    """Random sampling based on scaffolds. It tries to satisfy the desired proportion"""

    def __init__(self, trials=10, **kwargs):
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
        train_idx = np.array(cur_train_idx).astype(np.int)
        test_idx  = np.array(cur_test_idx).astype(np.int)
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

    def __init__(self, trials=1000, **kwargs):
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
        train_idx = np.array(cur_train_idx).astype(np.int)
        test_idx  = np.array(cur_test_idx).astype(np.int)
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
        train_idx = np.array(train_idx).astype(np.int)
        test_idx  = np.array(test_idx).astype(np.int)
        random.seed(seed)
        random.shuffle(train_idx)
        random.seed(seed)
        random.shuffle(test_idx)
        yield train_idx, test_idx


def GetSplitter(is_cv, is_classifier, is_stratified, scaffold_split):
    """Select the splitter, depending on the characteristics of the problem"""
    if is_classifier:
        if is_cv:
            if is_stratified:
                spl = model_selection.StratifiedKFold
            else:
                spl = model_selection.KFold
        else:
            if scaffold_split:
                if is_stratified:
                    spl = StratifiedShuffleScaffoldSplit
                else:
                    spl = ShuffleScaffoldSplit
            else:
                if is_stratified:
                    spl = model_selection.ShuffleSplit
                else:
                    spl = model_selection.StratifiedShuffleSplit
    else:
        # TO-DO
        pass
    return spl