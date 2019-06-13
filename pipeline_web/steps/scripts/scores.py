from __future__ import division
import sys
import os
import h5py
import numpy as np
import bisect
import pickle
import uuid

from chemicalchecker.database import Dataset, Molecule
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker

# Variables

cutoff = 0.001

# Functions


def weights(coords):
    idxs = [coord_idxs[c] for c in coords]
    return np.mean(W[idxs, :][:, idxs], axis=1) + 1e-6


def get_props(cc, iks):

    sign0 = cc.get_signature("sign1", "full", "A5.001")

    _, props = sign0.get_vectors(iks)

    return props


def get_scores(ik, cc, dataset_keys, dataset_pairs, cut_idx):

    # Fetch data
    coords_obs = []
    for dataset_code in dataset_keys.keys():
        if ik in dataset_keys[dataset_code]:
            coords_obs += [dataset_code]
    w_obs = weights(coords_obs)

    coords_prd = []
    for dataset_code in dataset_keys.keys():
        coords_prd += [dataset_code]
    w_prd = weights(coords_prd)

    # Popularity
    popu = (np.sum(w_obs) - min_popu) / (max_popu - min_popu)

    if popu > 1:
        popu = 1.
    if popu < 0:
        popu = 0.

    # Singularity
    v = []
    for c in coords_obs:
        similars = cc.get_signature("neig1", "full", dataset_pairs[c])
        sign1 = cc.get_signature("sign1", "reference", dataset_pairs[c])
        metric = sign1.get_h5_dataset('metric')
        bg_vals = sign1.background_distances(metric[0])
        values = similars[ik]
        sums = max(0, np.sum(values["distances"] <= bg_vals["distance"][cut_idx]) - 1)
        v += [sums / (values["distances"] - 1)]
    if np.sum(v) == 0:
        sing = 0.
    else:
        sing = np.average(v, weights=w_obs)

    # Mappability
    v = []
    for c in coords_prd:
        neig2_ref = cc.get_signature("neig2", "reference", dataset_pairs[c])
        sign3 = cc.get_signature("sign3", "full", dataset_pairs[c])
        sign2 = cc.get_signature("sign2", "reference", dataset_pairs[c])
        bg_vals = sign2.background_distances("cosine")
        signature = sign3[ik]
        neig_predictions_distances = neig2_ref.get_kth_nearest([signature])
        sums = max(0, np.sum(neig_predictions_distances[
                   "distances"] <= bg_vals["distance"][cut_idx]) - 1)
        v += [sums / (len(neig_predictions_distances["distances"]) - 1)]

    if np.sum(v) == 0:
        mapp = 0.
    else:
        mapp = np.average(v, weights=w_prd)

    return popu, sing, mapp

task_id = sys.argv[1]
filename = sys.argv[2]
consensus = sys.argv[3]
output_path = sys.argv[4]
inputs = pickle.load(open(filename, 'rb'))
iks = inputs[task_id]

max_popu = np.sum(weights(
    ["%s%d" % (j, i) for j in ["A", "B", "C", "D", "E"] for i in [1, 2, 3, 4, 5]]))
min_popu = np.sum(weights(["A%d" % i for i in [1, 2, 3, 4, 5]]))


# Read generic data

cut_idx = None

with h5py.File(consensus, "r") as hf:
    W = 1. - hf["correlations"][:]
    W[np.diag_indices(W.shape[0])] = 0.
    coord_idxs = dict((c, i) for i, c in enumerate(hf["coords"][:]))

all_datasets = Dataset.get()
config_cc = Config()

cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

dataset_pairs = dict()
dataset_codes = list()
dataset_keys = dict()
for ds in all_datasets:
    if not ds.exemplary:
        continue
    dataset_pairs[ds.coordinate] = ds.dataset_code
    dataset_codes.append(ds.coordinate)

    sign1 = cc.get_signature("sign1", "full", str(ds.dataset_code))

    if cut_idx is None:
        cut_idx = bisect.bisect_left(sign1.PVALRANGES, cutoff)

    dataset_keys[ds.coordinate] = sign1.unique_keys


dataset_codes.sort()


outfile = os.path.join(output_path, uuid.uuid4())

mappings_inchi = Molecule.get_inchikey_inchi_mapping(iks)

props = get_props(cc, iks)

with open(outfile, "w") as f:
    for i, ik in enumerate(iks):
        s = get_scores(ik, cc, dataset_keys, dataset_pairs, cut_idx)
        inchi = mappings_inchi[ik]
        start_pos = inchi.find('/') + 1
        end_pos = inchi.find('/', start_pos)
        formula = inchi[start_pos:end_pos]
        prop = props[i]
        f.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n" %
                (ik, formula, s[0], s[1], s[2], prop[0], prop[14], prop[16]))
