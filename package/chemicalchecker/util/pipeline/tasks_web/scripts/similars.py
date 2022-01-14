import os
import sys
import time
import json
import pickle
import collections
import numpy as np

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.database import Dataset

cutoff_idx = 5  # what we consider similar (dist < p-value 0.02)
best = 20  # top molecules in libraries
dummy = 999  # distance bin for non-similar


def index_sign(dataset):
    offset = {'A': 0, 'B': 5, 'C': 10, 'D': 15, 'E': 20}
    if dataset.endswith('prd'):
        sign = -1
        pts = 1
    else:
        sign = 1
        pts = 2
    char, num = dataset[0], int(dataset[1])
    num -= 1
    col = offset[char] + num
    return col, sign, pts


# get script arguments
task_id = sys.argv[1]
filename = sys.argv[2]
names_jason = sys.argv[3]
lib_bio_file = sys.argv[4]
save_file_path = sys.argv[5]
dbname = sys.argv[6]
version = sys.argv[7]
CC_ROOT = sys.argv[8]
overwrite = False

# input is a chunk of universe inchikey
inchikeys = pickle.load(open(filename, 'rb'))[task_id]
# for each molecule check if json is already available
if not overwrite:
    notdone = list()
    for index, inchikey in enumerate(inchikeys):
        PATH = save_file_path + "/%s/%s/%s/" % (
            inchikey[:2], inchikey[2:4], inchikey)
        filename = PATH + '/explore_' + version + '.json'
        if os.path.isfile(filename):
            try:
                json.load(open(filename, 'r'))
            except Exception:
                notdone.append(inchikey)
                continue
        else:
            notdone.append(inchikey)
    if len(notdone) == 0:
        print('All molecules already present, nothing to do.')
        sys.exit()
    else:
        inchikeys = notdone

# for each molecule which spaces are available in sign1?
print('for each molecule which spaces are available in sign1?')
t0 = time.time()
cc = ChemicalChecker(CC_ROOT)
metric_obs = None
metric_prd = None
map_coords_obs = collections.defaultdict(list)
dataset_pairs = {}
for ds in Dataset.get(exemplary=True):
    print(ds)
    dataset_pairs[ds.coordinate] = ds.dataset_code
    if metric_obs is None:
        neig1 = cc.get_signature("neig1", "reference", ds.dataset_code)
        metric_obs = neig1.get_h5_dataset('metric')[0]
    if metric_prd is None:
        neig3 = cc.get_signature("neig3", "reference", ds.dataset_code)
        metric_prd = neig3.get_h5_dataset('metric')[0]
    sign1 = cc.get_signature("sign1", "full", ds.dataset_code)
    keys = sign1.unique_keys
    for ik in keys:
        map_coords_obs[ik] += [ds.coordinate]
print('took', time.time() - t0)

# get relevant background distances and mappings
print('get relevant background distances')
t0 = time.time()
bg_vals = dict()
bg_vals['obs'] = dict()
bg_vals['prd'] = dict()
signatures = dict()
signatures['obs'] = dict()
signatures['prd'] = dict()
for coord in dataset_pairs.keys():
    print(coord)
    sign1 = cc.get_signature("sign1", "reference", dataset_pairs[coord])
    bg_vals['obs'][coord] = sign1.background_distances(metric_obs)["distance"]
    signatures['obs'][coord] = sign1
    sign3 = cc.get_signature("sign3", "reference", dataset_pairs[coord])
    bg_vals['prd'][coord] = sign3.background_distances(metric_prd)["distance"]
    signatures['prd'][coord] = sign3
print('took', time.time() - t0)

# for both observed (sign1) and predicted (sign3) get significant neighbors
print('get significant neighbors')
t0 = time.time()
keys = [k + "_obs" for k in dataset_pairs.keys()] + \
    [k + "_prd" for k in dataset_pairs.keys()]
ds_inks_bin = {}
neig_cctype = {
    'obs': 'neig1',
    'prd': 'neig3',
}
for dataset in keys:
    print(dataset)
    coord, type_data = dataset.split("_")
    dist_cutoffs = bg_vals[type_data][coord]
    neig = cc.get_signature(
        neig_cctype[type_data], "full", dataset_pairs[coord])
    _, nn_dist = neig.get_vectors(
        inchikeys, include_nan=True, dataset_name='distances')
    _, nn_inks = neig.get_vectors(
        inchikeys, include_nan=True, dataset_name='indices')
    # mask to keep only neighbors below cutoff
    masks = nn_dist <= dist_cutoffs[cutoff_idx]
    # get binned data according to distance cutoffs
    dist_bin = np.digitize(nn_dist, dist_cutoffs)
    # get close neighbors inchikeys and distance bins and apply mapping
    mappings = signatures[type_data][coord].get_h5_dataset('mappings')
    all_inks = list()
    all_dbins = list()
    # couldn't find a way to avoid iterating on molecules
    for ref_nn_ink, ref_dbin, mask in zip(nn_inks, dist_bin, masks):
        # apply distance cutoff
        ref_nn_ink = ref_nn_ink[mask]
        ref_dbin = ref_dbin[mask]
        # iterate on bins to aggregate mappings
        full_inks = list()
        full_dbins = list()
        for dbin in np.unique(ref_dbin):
            # get inks in the bin
            ink_dbin = ref_nn_ink[ref_dbin == dbin]
            # get idx bassed on redundant 'reference' column
            full_idxs = np.isin(mappings[:,1], ink_dbin)
            # get non redundnt 'full' inks
            full_nn_ink = mappings[:,0][full_idxs]
            # append to molecule lists
            full_inks.extend(full_nn_ink)
            full_dbins.extend([dbin] * len(full_nn_ink))
        all_inks.append(full_inks)
        all_dbins.append(full_dbins)

    # keep neighbors and bins for later
    ds_inks_bin[dataset] = (all_inks, all_dbins)
print('took', time.time() - t0)

# read inchikey to pubmed names mapping
with open(names_jason) as json_data:
    inchies_names = json.load(json_data)

# read library bioactive
with open(lib_bio_file) as json_data:
    ref_bioactive = json.load(json_data)
libs = set(ref_bioactive.keys())
libs.add("All Bioactive Molecules")

print('save json')
t0_tot = time.time()
# save in each molecule path the file the explore json (ranked neighbors)
for index, inchikey in enumerate(inchikeys):
    t0 = time.time()
    # only consider spaces where the molecule is present
    keys = [k + "_obs" for k in map_coords_obs[inchikey]] + \
        [k + "_prd" for k in dataset_pairs.keys()]

    # check if there are neighbors and keep their distance bin
    all_neig = set()
    neig_ds = dict()
    empty_spaces = list()
    for dataset in keys:
        inks = ds_inks_bin[dataset][0][index]
        if len(inks) == 0:
            empty_spaces.append(dataset)
            continue
        # iterate on each neighbor and expand to full set
        all_neig.update(set(inks))
        dbins = ds_inks_bin[dataset][1][index]
        neig_ds[dataset] = dict(zip(inks, dbins))
    for ds in empty_spaces:
        keys.remove(ds)

    # join and sort all neighbors from all spaces obs and pred
    all_neig = np.array(sorted(list(all_neig)))
    ink_pos = dict(zip(all_neig, np.arange(len(all_neig))))
    M = np.full((len(all_neig), 26), np.nan)
    M[:, 25] = 0

    # keep track of neigbors in reference libraries
    inchies = dict()
    ref_counts = dict()
    for lib in libs:
        ref_counts[lib] = [0] * 25
        inchies[lib] = set()

    # rank all neighbors
    selected = set()
    for dataset in keys:
        square, type_data = dataset.split("_")
        pos, sign, pts = index_sign(dataset)

        # iterate on all generic neigbors
        for t, ik in enumerate(all_neig):
            val = M[t, pos]
            # if generic neighbor has value from obs leave it
            if val > 0:
                continue
            # if generic neighbor doesn't have a value set it
            # if it is in current space, update points matrix
            if ik in neig_ds[dataset]:
                dist = neig_ds[dataset][ik]
                M[t, pos] = sign * dist
                M[t, 25] += pts
            # otherwise check if we can say they are different
            else:
                # if dataset is obs check against molecules in sign1
                if type_data == 'obs':
                    if square in map_coords_obs[ik]:
                        M[t, pos] = sign * dummy
                # if dataset is prd check against universe
                else:
                    if ik in map_coords_obs:
                        M[t, pos] = sign * dummy

        # select top neighbors in current space that are also part of libraries
        for ik in neig_ds[dataset]:
            # never select self
            if ik == inchikey:
                continue
            for lib in libs:
                # if we already selected enought stop
                if ref_counts[lib][pos] >= best:
                    break
                if lib == 'All Bioactive Molecules':
                    found = True
                else:
                    found = ik in ref_bioactive[lib]
                if found:
                    ref_counts[lib][pos] += 1
                    selected.add(ik)
                    inchies[lib].add(ik)

    # convert to lists
    selected = list(selected)
    for lib in libs:
        inchies[lib] = list(inchies[lib])

    # save neigbors data for explore page
    for sel in selected:
        inchies[sel] = {}
        inchies[sel]["inchikey"] = sel
        inchies[sel]["data"] = [None if np.isnan(x) else x for x in M[
            ink_pos[sel]]]
        if sel in inchies_names:
            inchies[sel]["name"] = inchies_names[sel]
        else:
            inchies[sel]["name"] = ""
    PATH = save_file_path + "/%s/%s/%s/" % (
        inchikey[:2], inchikey[2:4], inchikey)
    with open(PATH + '/explore_' + version + '.json', 'w') as outfile:
        json.dump(inchies, outfile)

    print(inchikey, 'took', time.time() - t0)
print('saving all took', time.time() - t0_tot)
