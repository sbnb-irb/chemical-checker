"""Sample triplets globally and do a train-test split globally.

.. note::
    THIS IS UNDER DEVELOPMENT. IT WAS INTENDED TO BE USED IN THE PRE-TRAINING
    OF AN ACCEPTABLE SIGN3 ACROSS THE CC, BUT WE FINALLY DIDN'T USE IT TOO
    EXTENSIVELY.
"""

import numpy as np
import h5py
from scipy.stats import rankdata
from tqdm import tqdm

from chemicalchecker.util import logged

@logged
class GlobalNeighborTripletTraintest(object):

    def __init__(self, cc, cctype="sign1", nr_neigh=10, attempts=10):
        self.cc = cc
        if datasets is None:
            self.__log.info("Getting exemplary datasets")
            datasets = []
            for ds in cc.datasets_exemplary():
                datasets += [ds]
        self.datasets = datasets
        self.cctype = cctype
        self.nr_neigh = nr_neigh
        self.attempts = attempts

    def read_across(self, keys, split_fractions):
        self.__log.debug("Reading across")
        results = {}
        for ds in tqdm(self.datasets):
            sign_full = self.cc.signature(ds, self.cctype)
            sign_ref  = sign_full.get_molset("reference")
            keys_full = sign_full.keys
            keys_ref  = sign_ref.keys
            maps_ref_ = sign_ref.mappings
            maps_ref  = []
            for i in range(0, maps_ref_.shape[0]):
                maps_ref += [[maps_ref_[i][0], maps_ref_[i][1]]]
            maps_ref  = np.array(maps_ref)
            results[ds] = (keys_full, keys_ref, maps_ref)
        self.__log.debug("Refactoring")
        res_refact = {}
        for k,v in results.items():
            keys_full = set(v[0])
            keys_ref  = set(v[1])
            maps      = dict((v[2][i,0],v[2][i,1]) for i in range(0, v[2].shape[0]))
            res_refact[k] = {
                "keys_full": keys_full,
                "keys_ref": keys_ref,
                "maps": maps
            }
        return res_refact

    def collisions(self, keys, idxs, maps):
        split_fractions = np.array([len(idx) for idx in idxs])
        split_fractions = split_fractions/split_fractions.sum()
        ref_sets = []
        keys_found = 0
        for idx in idxs:
            ref_set = set()
            keys_ = keys[idx]
            for k in keys_:
                if k not in maps: continue
                ref_set.update([maps[k]])
                keys_found += 1
            ref_sets += [ref_set]
        co = 0
        for i in range(0, len(ref_sets)-1):
            for j in range(i+1, len(ref_sets)):
                co += len(ref_sets[i].intersection(ref_sets[j]))
        co_frac = co/keys_found
        sizes = np.array([len(ref_sets[i]) for i in range(0, len(ref_sets))])
        fractions = [s/sizes.sum() for s in sizes]
        error = np.sum([np.abs(fractions[i] - split_fractions[i]) for i in range(0, len(split_fractions))])
        results = {
            "collisions": co,
            "collisions_fraction": co/keys_found,
            "sizes": sizes,
            "fractions": fractions,
            "error": error
        }
        return results

    def collisions_all(self, keys, idxs, ra):
        results = {}
        for k,v in ra.items():
            results[k] = self.collisions(keys, idxs, v["maps"])
        return results

    def score_collisions(self, collis):
        x = []
        y = []
        for k,v in collis.items():
            x += [v["collisions_fraction"]]
            y += [v["error"]]
        cf = np.mean(x)
        er = np.mean(y)
        return cf, er

    def get_splits(self, keys, fractions):
        if not sum(fractions) == 1.0:
            raise Exception("Split fractions should sum to 1.0")
        idxs = list(range(keys))
        np.random.shuffle(idxs)
        splits = np.cumsum(fractions)
        splits = splits[:-1]
        splits *= len(idxs)
        splits = splits.round().astype(int)
        return np.split(idxs, splits)

    def acceptable_split(self, keys, split_fractions):
        keys = np.array(keys)
        ra = self.read_across(keys, split_fractions)
        # doing attempts
        atts = []
        for _ in tqdm(range(0, self.attempts)):
            idxs = self.get_splits(len(keys), [0.8, 0.1, 0.1])
            atts += [idxs]
        # scoring
        cfs = []
        ers = []
        for idxs in tqdm(atts):
            collis = self.collisions_all(keys, idxs, ra)
            cf, er = self.score_collisions(collis)
            cfs += [cf]
            ers += [er]
        # rank attempts
        cfs_ranks = rankdata(cfs)
        ers_ranks = rankdata(ers)
        ranks = np.mean(np.array([cfs_ranks, ers_ranks]), axis=0)
        # choose idxs
        idxs = atts[np.argmin(ranks)]
        return idxs
