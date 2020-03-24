import bisect
import time
import numpy as np
import os
import sys
import collections
import json
import math
import pickle

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql

#inchikey = 'ZZVUWRFHKOJYTH-UHFFFAOYSA-N'

cutoff = 5
best = 20
dummy = 999
text_lib = "select  library_description.name as lib,lib.inchikey, pubchem.name from libraries as lib INNER JOIN library_description on lib.lib = library_description.lib  LEFT JOIN pubchem ON lib.inchikey = pubchem.inchikey_pubchem where is_landmark = '1' order by  library_description.rank"
text_bio = "select  library_description.name as lib,lib.inchikey from libraries as lib INNER JOIN library_description on lib.lib = library_description.lib   where lib.is_bioactive = '1' order by  library_description.rank"


# HAVE THIS IN MEMMORY!

def get_integer(d):
    if d < 0:
        return d
    return bisect.bisect_left(bg_vals, d)


def index_sign(coord, pred):
    if pred:
        sign = -1
    else:
        sign = 1
    char, num = coord[0], int(coord[1])
    if char == "A":
        return (0 + num - 1), sign
    if char == "B":
        return (5 + num - 1), sign
    if char == "C":
        return (10 + num - 1), sign
    if char == "D":
        return (15 + num - 1), sign
    if char == "E":
        return (20 + num - 1), sign


task_id = sys.argv[1]
filename = sys.argv[2]
names_jason = sys.argv[3]
save_file_path = sys.argv[4]
dbname = sys.argv[5]
version = sys.argv[6]
CC_ROOT = sys.argv[7]
inputs = pickle.load(open(filename, 'rb'))
inchikeys = inputs[task_id]

references = {}
ref_bioactive = {}
dataset_pairs = {}
all_datasets = Dataset.get()

cc = ChemicalChecker(CC_ROOT)

get_integers = np.vectorize(get_integer)
metric_obs = None
map_coords_obs = collections.defaultdict(list)

for ds in all_datasets:
    if not ds.exemplary:
        continue

    dataset_pairs[ds.coordinate] = ds.dataset_code

    if metric_obs is None:
        sign1 = cc.get_signature("sign1", "reference", ds.dataset_code)
        metric_obs = sign1.get_h5_dataset('metric')

    sign1 = cc.get_signature("sign1", "full", ds.dataset_code)

    keys = sign1.unique_keys
    for ik in inchikeys:
        if ik in keys:
            map_coords_obs[ik] += [ds.coordinate]

bg_vals_obs = {}
bg_vals_pred = {}

for coord in dataset_pairs.keys():
    sign1 = cc.get_signature("sign1", "reference", dataset_pairs[coord])
    bg_vals_obs[coord] = sign1.background_distances(
        metric_obs[0])["distance"]

    sign2 = cc.get_signature("sign2", "reference", dataset_pairs[coord])
    bg_vals_pred[coord] = sign2.background_distances("cosine")["distance"]

keys = [k + "_obs" for k in dataset_pairs.keys()] + \
    [k + "_prd" for k in dataset_pairs.keys()]


data_keys_map = {}

for dataset in keys:
    coord, type_data = dataset.split("_")

    sim_values = {}

    if type_data == 'obs':
        similars = cc.get_signature("neig1", "full", dataset_pairs[coord])
        _, sim_values["distances"] = similars.get_vectors(
            inchikeys, include_nan=True, dataset_name='distances')
        _, sim_values["keys"] = similars.get_vectors(
            inchikeys, include_nan=True, dataset_name='indices')
        bg_vals = bg_vals_obs[coord]
        cutoff_sim = bg_vals[cutoff]

    else:
        # Actually, we are not using neig2 but the class methods to access
        # this kind of data
        similars = cc.get_signature("neig2", "full", dataset_pairs[coord])
        sign3 = cc.get_signature("sign3", "full", dataset_pairs[coord])
        similars.data_path = os.path.join(
            sign3.signature_path, "similars.h5")
        _, sim_values["distances"] = similars.get_vectors(
            inchikeys, include_nan=True, dataset_name='distances')
        _, sim_values["keys"] = similars.get_vectors(
            inchikeys, include_nan=True, dataset_name='indices')
        bg_vals = bg_vals_pred[coord]
        cutoff_sim = bg_vals[cutoff]

    mask = sim_values["distances"] <= cutoff_sim
    iksm = np.where(mask, sim_values["keys"], '')
    # iksm = sim_values["keys"][mask].tolist()

    # list1 = np.array(iksm)
    # list2 = np.array(sim_values["distances"][mask])
    # idx = np.argsort(list2)
    # new_iksm = np.array(list1)[idx]
    max_idx = len(bg_vals) - 1
    integers = get_integers(np.where(mask, sim_values["distances"], -1.0))
    integers[integers > max_idx] = max_idx
    data_keys_map[dataset] = (iksm, integers)


lib_refs = psql.qstring(text_lib, dbname)


for lib in lib_refs:
    if lib[0] not in references:
        references[lib[0]] = []
    if len(references[lib[0]]) < 100:
        references[lib[0]].append({'inchikey': lib[1], 'name': lib[2]})


print (references.keys())
lib_bio = psql.qstring(text_bio, dbname)

for lib in lib_bio:
    if lib[0] not in ref_bioactive:
        ref_bioactive[lib[0]] = set()

    ref_bioactive[lib[0]].add(lib[1])

with open(names_jason) as json_data:
    inchies_names = json.load(json_data)
# UNTIL HERE

for index, inchikey in enumerate(inchikeys):
    t0 = time.time()
    PATH = save_file_path + "/%s/%s/%s/" % (
        inchikey[:2], inchikey[2:4], inchikey)
    print(PATH)
    keys = [k + "_obs" for k in map_coords_obs[inchikey]] + \
        [k + "_prd" for k in dataset_pairs.keys()]

    S = set()
    X = []

    empty_spaces = list()

    for dataset in keys:
        coord, type_data = dataset.split("_")

        iksm = data_keys_map[dataset][0][index][
            data_keys_map[dataset][0][index] != ''].tolist()

        if len(iksm) == 0:
            print (dataset)
            empty_spaces.append(dataset)
            continue

        S.update(iksm)

        integers = data_keys_map[dataset][1][index][
            data_keys_map[dataset][1][index] >= 0]

        dict_iks_dist = dict(zip(iksm, integers))

        X += [(dict_iks_dist, iksm)]

    for ds in empty_spaces:
        keys.remove(ds)

    S = np.array(list(S))

    M = np.full((len(S), 26), np.nan)

    for i in range(0, len(S)):
        M[i, 25] = 0

    libs = set(ref_bioactive.keys())
    libs.add("All Bioactive Molecules")

    c = 0
    inchies = {}
    map_inchies_pos = {}
    t = 0

    for ele in S:
        map_inchies_pos[ele] = t
        t += 1

    selected = set()

    inchi_points = set()
    ref_counts = {}

    for lib in libs:

        ref_counts[lib] = [0] * 25
        inchies[lib] = set()

    for dataset in keys:
        # print dataset
        square, type_data = dataset.split("_")
        if type_data == 'obs':
            pos, sign = index_sign(square, False)
        else:
            pos, sign = index_sign(square, True)

        dict_iks_dist = X[c][0]
        # simsm = X[c][1]
        ordered_iksm = X[c][1]
        # iksmset = set(iksm)
        t = 0
        index = 0
        found = False
        for ik in S:
            if ik in dict_iks_dist and (np.isnan(M[t, pos]) or M[t, pos] == dummy):
                # index2 = iksm.index(ik)

                # index = bisect.bisect_left(iksm, ik)
                # print index2, index
                M[t][pos] = sign * (dict_iks_dist[ik] + 1)
                if sign > 0:
                    M[t, 25] += 2
                else:
                    M[t, 25] += 1

            else:
                if square in map_coords_obs[ik] and np.isnan(M[t, pos]):
                    M[t, pos] = sign * dummy

            t += 1

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

        c += 1

    new_selected = list(selected)
    for lib in libs:
        inchies[lib] = list(inchies[lib])

    for sel in new_selected:
        inchies[sel] = {}
        inchies[sel]["inchikey"] = sel
        inchies[sel]["data"] = [None if math.isnan(x) else x for x in M[
            map_inchies_pos[sel]]]
        if sel in inchies_names:
            inchies[sel]["name"] = inchies_names[sel]
        else:
            inchies[sel]["name"] = ""

    with open(PATH + '/explore_' + version + '.json', 'w') as outfile:
        json.dump(inchies, outfile)

    print time.time() - t0
