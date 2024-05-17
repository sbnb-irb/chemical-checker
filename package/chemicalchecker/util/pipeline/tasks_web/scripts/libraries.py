from __future__ import division
import sys
import os
import h5py
import numpy as np
import pickle
import collections
from chemicalchecker.util.download import Downloader
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.parser import Parser
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql

INSERT = "INSERT INTO libraries VALUES %s"

SELECT_INFO = """
SELECT inchikey, popularity FROM molecular_info WHERE inchikey IN %s
""".replace('\n', ' ').strip()


def chunker(data, size=2000):
    for i in range(0, len(data), size):
        yield data[slice(i, i + size)]


def select_landmarks(inchikeys, N, dbname):
    sqiks_ = dict((k, v.intersection(inchikeys)) for k, v in sqiks.items())
    # Start with the more popular molecule
    done_connects = set()
    done_clusts = set()
    square_mols = dict((k, 0) for k in sqiks_.keys())
    s = "(%s)" % ",".join(["'%s'" % ik for ik in inchikeys])
    pop = np.array(
        [r for r in psql.qstring(SELECT_INFO % s, dbname)],
        dtype=np.dtype(
            [('ik', h5py.special_dtype(vlen=str)), ('pop', 'float' )]))
    pop = np.sort(pop, order="pop")[::-1]
    ik = pop['ik'][0]
    landmarks = set([ik])
    done_connects.update([ik.split("-")[0]])
    done_clusts.update(clusts[ik])
    for sq in squares[ik]:
        square_mols[sq] += 1
    for sq in sqiks_.keys():
        sqiks_[sq] = set([i for i in list(sqiks_[sq]) if i.split(
            "-")[0] not in done_connects]).difference(landmarks)

    # Keep adding molecules
    while len(landmarks) < N:
        S = [v for k, v in square_mols.items() if len(sqiks_[k]) > 0]
        if len(S) == 0:
            break
        minsize = np.min(S)
        cand_mols = set()
        for coord, v in square_mols.items():
            if v != minsize:
                continue
            cand_mols.update(sqiks_[coord])

        # Use, also, popularity score (already sorted!)
        # Only consider top-10 by popularity (reasonable enough).
        pop_score = [r['ik'] for r in pop if r['ik'] in cand_mols][:10]
        # Count number of covered clusters per molecule (overlap)
        clust_score = np.array(
            [(i, len(clusts[i].intersection(done_clusts)) / len(clusts[i]))
             for i in pop_score],
            dtype=np.dtype(
                [('ik', h5py.special_dtype(vlen=str)), ('o', 'float' )]))
        clust_score = [r['ik'] for r in np.sort(clust_score, order="o")]

        # Merge the two rankings
        rankings = collections.defaultdict(list)
        for i, ik in enumerate(clust_score):
            rankings[ik] += [i]
        for i, ik in enumerate(pop_score):
            rankings[ik] += [i]
        rankings = dict((k, np.mean(v)) for k, v in rankings.items())
        ik = min(rankings, key=lambda k: rankings[k])

        landmarks.update([ik])
        done_connects.update([ik.split("-")[0]])
        done_clusts.update(clusts[ik])
        for sq in squares[ik]:
            square_mols[sq] += 1
        for sq in sqiks_.keys():
            sqiks_[sq] = set([i for i in list(sqiks_[sq]) if i.split(
                "-")[0] not in done_connects]).difference(landmarks)
    return landmarks


task_id = sys.argv[1]
filename = sys.argv[2]
universe = sys.argv[3]
save_file_path_root = sys.argv[4]
dbname = sys.argv[5]
CC_ROOT = sys.argv[6]
inputs = pickle.load(open(filename, 'rb'))
lib = inputs[task_id]

N = 100  # Number of landmarks

# libraries info
lib_id = list(lib.keys())[0]
lib_name = lib[lib_id][0]
lib_desc = lib[lib_id][1]
lib_urls = lib[lib_id][2]
lib_parser = lib[lib_id][3]

# download libraries
files = list()
save_file_path = os.path.join(save_file_path_root, lib_id)
for idx, url in enumerate(lib_urls.split(";"), 1):
    filename = lib_id + "_" + str(idx) + ".smi"
    file_path = os.path.join(save_file_path, filename)
    files.append(file_path)
    if os.path.exists(file_path):
        continue
    down = Downloader(url, save_file_path, dbname=None, file=filename)
    down.download()
for fpath in files:
    if not os.path.exists(fpath):
        raise Exception("The library " + lib_id +
                        " file did not download correctly")

# group by atom connectivity
bioconn = list()
with h5py.File(universe, "r") as hf:
    bioactive = hf["keys"][:]
    bioactive = set( [ el.decode('utf-8') for el in bioactive ] )
    bioconn_bioactive = collections.defaultdict(list)
    for ik in bioactive:
        bioconn_bioactive[ik.split("-")[0]] += [ik]
    bioconn = set(bioconn_bioactive.keys())

# parse libraries
parse_fn = Parser.parse_fn(lib_parser)
map_files = {}
keys = set()
for fpath in files:
    map_files[lib_id] = fpath
    print(fpath)
    for chunk in parse_fn(map_files, lib_id, 1000):
        for data in chunk:
            if data["inchikey"] is not None:
                keys.add(data["inchikey"])

# get coordinate-code mapping
dataset_pairs = dict()
coords = list()
for ds in Dataset.get(exemplary=True):
    dataset_pairs[ds.coordinate] = ds.dataset_code
    coords.append(ds.coordinate)
coords.sort()

# Count number of squares where the molecule appears
cc = ChemicalChecker(CC_ROOT)
squares = collections.defaultdict(set)
clusts = collections.defaultdict(set)
sqiks = collections.defaultdict(set)
for coord in coords:
    clus1 = cc.get_signature("clus1", "full", dataset_pairs[coord])
    iks = clus1.keys
    clus = clus1.get_h5_dataset("labels")
    for ik, clu in zip(iks, clus):
        squares[ik].update([coord])
        clusts[ik].update(["%s_%d" % (coord, clu)])
        sqiks[coord].update([ik])


# Prepare the R vector


# Get landmarks
aux = bioactive.intersection(keys)
print(len(aux), len(keys))
landmark = select_landmarks(aux, N, dbname)
# Start iterating
alreadies = set()
R = []
for x in keys:

    xs = x.split("-")[0]
    if xs in bioconn:
        b = 1
        for y in bioconn_bioactive[xs]:
            if y in landmark:
                l = 1
            else:
                l = 0
            if (y, lib_id) in alreadies:
                continue
            R += ["('%s', '%s', %d, %d)" % (y, lib_id, b, l)]
            alreadies.update([(y, lib_id)])
    else:
        b = 0
        l = 0
        if (x, lib_id) in alreadies:
            continue
        R += ["('%s', '%s', %d, %d)" % (x, lib_id, b, l)]
        alreadies.update([(x, lib_id)])
for c in chunker(R, 1000):
    psql.query(INSERT % ",".join(c), dbname)

print('JOB DONE')
