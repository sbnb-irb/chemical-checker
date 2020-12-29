import os
import sys
import time
import json
import pickle
import collections
import numpy as np

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.database import Dataset

#inchikey = 'ZZVUWRFHKOJYTH-UHFFFAOYSA-N'

cutoff = 5
best = 20
dummy = 999

# HAVE THIS IN MEMMORY!

def index_sign(coord, pred):
    if pred:
        sign = -1
        pts = 1
    else:
        sign = 1
        pts = 2
    char, num = coord[0], int(coord[1])
    if char == "A":
        return (0 + num - 1), sign, pts
    if char == "B":
        return (5 + num - 1), sign, pts
    if char == "C":
        return (10 + num - 1), sign, pts
    if char == "D":
        return (15 + num - 1), sign, pts
    if char == "E":
        return (20 + num - 1), sign, pts


# get script arguments
task_id = sys.argv[1]
filename = sys.argv[2]
names_jason = sys.argv[3]
lib_bio_file = sys.argv[4]
save_file_path = sys.argv[5]
dbname = sys.argv[6]
version = sys.argv[7]
CC_ROOT = sys.argv[8]

# input is a chunk of universe inchikey
inchikeys = pickle.load(open(filename, 'rb'))[task_id]

# for each molecule check if json is already available
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
    inchikey = notdone

# for each molecule which spaces are available in sign1?
print('for each molecule which spaces are available in sign1?')
t0 = time.time()
cc = ChemicalChecker(CC_ROOT)
metric_obs = None
metric_prd = None
map_coords_obs = collections.defaultdict(list)
dataset_pairs = {}
for ds in Dataset.get(exemplary=True):
    dataset_pairs[ds.coordinate] = ds.dataset_code
    if metric_obs is None:
        neig1 = cc.get_signature("neig1", "reference", ds.dataset_code)
        metric_obs = neig1.get_h5_dataset('metric')[0]
    if metric_prd is None:
        neig3 = cc.get_signature("neig3", "reference", ds.dataset_code)
        metric_prd = neig3.get_h5_dataset('metric')[0]
    sign1 = cc.get_signature("sign1", "full", ds.dataset_code)
    keys = sign1.unique_keys
    for ik in inchikeys:
        if ik in keys:
            map_coords_obs[ik] += [ds.coordinate]
print('took', time.time() - t0)

# get relevant background distances
print('get relevant background distances')
t0 = time.time()
bg_vals = dict()
bg_vals['obs'] = dict()
bg_vals['prd'] = dict()
for coord in dataset_pairs.keys():
    sign1 = cc.get_signature("sign1", "reference", dataset_pairs[coord])
    bg_vals['obs'][coord] = sign1.background_distances(metric_obs)["distance"]
    sign3 = cc.get_signature("sign3", "reference", dataset_pairs[coord])
    bg_vals['prd'][coord] = sign3.background_distances(metric_prd)["distance"]
print('took', time.time() - t0)

# for both observed (sign1) and predicted (sign3) get significant neighbors
print('get significant neighbors')
t0 = time.time()
keys = [k + "_obs" for k in dataset_pairs.keys()] + \
    [k + "_prd" for k in dataset_pairs.keys()]
# FIXME this variable gets pretty heavy, can we save memory?
data_keys_map = {}
neig_cctype = {
    'obs': 'neig1',
    'prd': 'neig3',
}
for dataset in keys:
    coord, type_data = dataset.split("_")
    sim_values = {}
    neig = cc.get_signature(
        neig_cctype[type_data], "full", dataset_pairs[coord])
    _, sim_values["distances"] = neig.get_vectors(
        inchikeys, include_nan=True, dataset_name='distances')
    _, sim_values["keys"] = neig.get_vectors(
        inchikeys, include_nan=True, dataset_name='indices')
    cutoff_sim = bg_vals[type_data][coord][cutoff]
    # mask filter at cutoff (NaN remain NaN)
    mask = sim_values["distances"] <= cutoff_sim
    # set to NaN those above threshold
    sim_values["distances"][~mask] = np.nan
    # get ink matrix (leave empty string for NaN and above cutoff)
    iksm = np.where(mask, sim_values["keys"], '')
    # get binned data according to distance cutoffs
    dist_bin = np.digitize(sim_values["distances"],
                           bg_vals[type_data][coord]) - 1
    dist_bin[~mask] = -1
    data_keys_map[dataset] = (iksm, dist_bin)
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
# save in each molecule path the file the explore json (100 similar molecules)
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
        coord, type_data = dataset.split("_")
        iksm = data_keys_map[dataset][0][index]
        iksm = iksm[iksm != ''].tolist()
        if len(iksm) == 0:
            empty_spaces.append(dataset)
            continue
        all_neig.update(iksm)
        dist_bin = data_keys_map[dataset][1][index]
        dist_bin = dist_bin[dist_bin >= 0].tolist()
        dict_iks_dist = dict(zip(iksm, dist_bin))
        neig_ds[dataset] = dict_iks_dist
    for ds in empty_spaces:
        keys.remove(ds)

    #all_neig = np.array(sorted(list(all_neig)))
    all_neig = np.array(list(all_neig))
    M = np.full((len(all_neig), 26), np.nan)
    M[:, 25] = 0

    # keep track of neigbors in reference libraries
    inchies = dict()
    ref_counts = dict()
    for lib in libs:
        ref_counts[lib] = [0] * 25
        inchies[lib] = set()

    # rank all neighbors
    map_inchies_pos = dict(zip(all_neig, np.arange(len(all_neig))))
    selected = set()
    for dataset in keys:
        square, type_data = dataset.split("_")
        pos, sign, pts = index_sign(square, type_data != 'obs')
        dict_iks_dist = neig_ds[dataset]
        ordered_iksm = list(dict_iks_dist)
        ordered_iksm.sort(key=lambda x: x[1])

        # update point matrix
        for t, ik in enumerate(all_neig):
            if ik in dict_iks_dist and (np.isnan(M[t, pos]) or M[t, pos] == dummy):
                M[t][pos] = sign * (dict_iks_dist[ik] + 1)
                M[t, 25] += pts
            else:
                if square in map_coords_obs[ik] and np.isnan(M[t, pos]):
                    M[t, pos] = sign * dummy

        # update reference library counts
        for ik in ordered_iksm:
            if ik == inchikey:
                continue
            if np.isnan(M[map_inchies_pos[ik], pos]) or abs(M[map_inchies_pos[ik], pos]) == dummy:
                continue
            for lib in libs:
                if ref_counts[lib][pos] >= best:
                    continue
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
            map_inchies_pos[sel]]]
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
