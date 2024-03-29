"""Triplet sampler.

Given a signature, sample triplets.
"""
import os
import h5py
import random
import numpy as np
import collections

from chemicalchecker.util import logged


@logged
class TripletSampler(object):
    """TripletSampler class."""

    def __init__(self, cc, sign0, max_sampled_keys=10000, save=True):
        """Initialize a TripletSampler instance."""
        # if sign0.cctype != "sign0":
        #    raise Exception("Triplet sampling only makes sense for sign0!")
        self.cc = cc
        self.sign = sign0
        self.max_sampled_keys = max_sampled_keys
        self.save = save

    def choice(self, row, fmref_idxs, num_samp):
        """Choose from a list of candidates"""
        n = len(row)
        cands = []
        probs = []
        for i, r in enumerate(row):
            if r not in fmref_idxs:
                continue
            for k in fmref_idxs[r]:
                cands += [k]
                probs += [n - i]
        if len(cands) < 1:
            return []
        if len(cands) == 1:
            return cands
        probs = np.array(probs) / np.sum(probs)
        if np.sum(probs) != 1:
            return []
        cands = np.array(cands)
        cands = np.random.choice(cands, num_samp, replace=True, p=probs)
        return list(cands)

    def sample_triplets_from_dataset(self, keys, dataset, num_triplets, p_pos,
                                     p_neg, min_pos, max_pos, max_neg):
        # Represent our keys as indices
        keys_idxs = dict((k, i) for i, k in enumerate(keys))
        # Start with mappings
        sign_ds = self.cc.get_signature("sign1", "reference", dataset)
        #maps_ds = np.array([(x[0].decode("ascii"), x[1].decode("ascii")) for x in sign_ds.get_h5_dataset("mappings")])
        maps_ds = np.array([(x[0], x[1])
                            for x in sign_ds.get_h5_dataset("mappings")])
        keys_ds = sign_ds.keys
        keys_ds_idxs = dict((k, i) for i, k in enumerate(keys_ds))
        toref_idxs = {}
        for i, m in enumerate(maps_ds):
            k = m[0]
            r = m[1]
            if k not in keys_idxs:
                continue
            key_idx = keys_idxs[k]
            key_ds_idx = keys_ds_idxs[r]
            toref_idxs[key_idx] = key_ds_idx
        fmref_idxs = collections.defaultdict(list)
        for k, v in toref_idxs.items():
            fmref_idxs[v] += [k]
        # Focus on the neighbors class
        neig_ds = self.cc.get_signature("neig1", "reference", dataset)
        # Decide number of neighbors
        n_pos = int(np.max([neig_ds.shape[0] * p_pos, min_pos]))
        n_pos = int(np.min([n_pos, max_pos]))
        n_neg = int(
            np.min([neig_ds.shape[0] * p_neg, neig_ds.shape[1], max_neg]))
        self.__log.debug("Limiting the number of molecules")
        # Limit the number of molecules to search in
        toref_idxs_list = [(tidx, ridx) for tidx, ridx in toref_idxs.items()]
        toref_idxs_list_asdict = dict((r[1], r[0]) for r in toref_idxs_list)
        if self.max_sampled_keys < len(toref_idxs_list_asdict):
            keys_toref = [k for k in toref_idxs_list_asdict.keys()]
            keys_toref = random.sample(keys_toref, self.max_sampled_keys)
            toref_idxs_list_asdict = dict(
                (k, toref_idxs_list_asdict[k]) for k in keys_toref)
        toref_idxs_list = [(v, k) for k, v in toref_idxs_list_asdict.items()]
        toref_idxs_list = np.array(toref_idxs_list)
        toref_idxs_list = toref_idxs_list[np.argsort(toref_idxs_list[:, 1])]
        # Decide how much to sample
        num_samp = int(num_triplets / len(toref_idxs_list))
        if num_samp == 0:
            num_samp = int(num_triplets)
        # Sample from neighbors
        self.__log.debug("Sampling from nearest neighbors")
        with h5py.File(neig_ds.data_path, "r") as hf:
            nn = hf["indices"][toref_idxs_list[:, 1]]
        self.__log.debug("Iterating over the nearest neighbors")
        triplets = []
        for i in range(0, toref_idxs_list.shape[0]):
            tidx = toref_idxs_list[i, 0]
            ridx = toref_idxs_list[i, 1]
            pidxs = self.choice(nn[i, :n_pos], fmref_idxs, num_samp)
            nidxs = self.choice(nn[i, n_pos:n_neg], fmref_idxs, num_samp)
            if len(pidxs) == 0 or len(nidxs) == 0:
                continue
            for pidx, nidx in zip(pidxs, nidxs):
                if tidx == pidx:
                    continue
                triplets += [(tidx, pidx, nidx)]
        return set(triplets)

    def sample_triplets(self, datasets, num_triplets, p_pos, p_neg, min_pos,
                        max_pos, max_neg, max_rounds):
        """Sample triplets from multiple exemplary datasets of the CC."""
        self.__log.debug("Sampling triplets")
        keys = self.sign.keys
        if datasets == None:
            datasets = [ds for ds in self.cc.datasets_exemplary()]
        num_triplets_per_ds = num_triplets / len(datasets)
        triplets = set()
        for _ in range(max_rounds):
            random.shuffle(datasets)
            for ds in datasets:
                triplets_ds = self.sample_triplets_from_dataset(
                    keys=keys, dataset=ds, num_triplets=num_triplets_per_ds,
                    p_pos=p_pos, p_neg=p_neg, min_pos=min_pos, max_pos=max_pos,
                    max_neg=max_neg)
                triplets.update(triplets_ds)
            if len(triplets) >= num_triplets:
                break
        if len(triplets) > num_triplets:
            triplets = random.sample(triplets, num_triplets)
        return set(triplets)

    def map_triplets_to_reference(self, triplets):
        """Map triplets from full to reference indices"""
        self.__log.debug("Mapping triplets to reference")
        sign_ref = self.sign.get_molset("reference")
        triplets_ref = list()
        key2idx = dict((k, i) for i, k in enumerate(sign_ref.keys))
        mappings_ = sign_ref.mappings
        mappings = []
        for m in mappings_[:, 1]:
            #mappings += [key2idx[m.decode()]]
            mappings += [key2idx[m]]
        for triplet in list(triplets):
            i, j, k = triplet
            triplets_ref += [(mappings[i], mappings[j], mappings[k])]
        return set(triplets_ref)

    def save_triplets(self, triplets, fn):
        """Save triplets"""
        self.__log.debug("Writing triplets to %s" % fn)
        triplets = np.array(sorted(triplets), dtype=int)
        with h5py.File(fn, "w") as hf:
            hf.create_dataset("triplets", data=triplets)

    def sample(self, datasets=None, num_triplets=1000000, p_pos=0.001,
               p_neg=0.1, min_pos=10, max_pos=100, max_neg=1000, max_rounds=3,**kwargs):
        """Sample triplets from multiple exemplary datasets of the CC.

            Args:
                datasets (list): Datasets to be used for the triplet sampling.
                    In none specified, all exemplary are used (default=None).
                num_triplets (int): Number of triplets to sample
                    (default=1000000).
                p_pos (float): P-value for positive cases (default=0.001).
                p_neg (float): P-value for negative cases. In order to provide
                    'hard' cases, it is recommended to put a relatively low
                    p-value (default=0.1).
                min_pos (int): Minimum number of neighbors considered to be
                    positives.
                max_neg (int): Maximum number of neighbors considered to be
                    negatives.
                max_rounds (int): Triplets may be sampled redundantly. Number
                    of rounds to be done before stopping trying (default=10).
        """
        triplets_full = self.sample_triplets(
            datasets=datasets, num_triplets=num_triplets, p_pos=p_pos,
            p_neg=p_neg, min_pos=min_pos, max_pos=max_pos, max_neg=max_neg,
            max_rounds=max_rounds)
        triplets_reference = self.map_triplets_to_reference(triplets_full)
        results = {
            "full": triplets_full,
            "reference": triplets_reference
        }
        if self.save:
            fn_full = os.path.join(self.sign.model_path, "triplets.h5")
            self.save_triplets(triplets_full, fn_full)
            fn_reference = os.path.join(self.sign.get_molset(
                "reference").model_path, "triplets.h5")
            self.save_triplets(triplets_reference, fn_reference)
            results = {
                "full": fn_full,
                "reference": fn_reference
            }
        return results
