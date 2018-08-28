# coding: utf-8

# Imports

import subprocess
from scipy.spatial.distance import euclidean
import sys, os
import json
import h5py
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
from checkerUtils import inchikey2webrepo
from Psql import  querylite
import checkerconfig
import numpy as np
import bisect

# Compare molecules

coordinate, inchikey, infolder_infile, infile, bgfile, outfile, vname, VERSION = sys.argv[1].split('---')

chunksize = 1000000

print coordinate, inchikey

# Read pvalues

with h5py.File(bgfile, "r") as hf:
    distances = hf["distance"][:]
    max_idx = len(distances) - 1

def get_integer(distances, d):
    i = bisect.bisect_left(distances, d)
    if i > max_idx: return max_idx
    return i

# Compute similarities

with h5py.File(infolder_infile, "r") as hf0:
    iks = hf0["keys"][:]
    with h5py.File(infile, "r") as hf:
        inchikeys = hf["keys"][:]
        N = len(inchikeys)
        mysig = hf0["V"][bisect.bisect_left(iks, inchikey)]
        INTEGERS = []
        from_i = 0
        while from_i < N:
            to_i = from_i + chunksize
            V = hf["V"][from_i:to_i]
            for j in xrange(V.shape[0]):
                d = euclidean(mysig, V[j])
                INTEGERS += [get_integer(distances, d)]
            from_i += chunksize

INTEGERS = np.array(INTEGERS).astype(np.int8)

PATH = inchikey2webrepo(inchikey)

with h5py.File("%s/%s" % (PATH, outfile), "a") as hf:
    dname = coordinate + "_obs"
    if dname in hf.keys(): del hf[dname]
    hf.create_dataset(dname, data = INTEGERS)

# Inserting the current version in the database

cmd = "INSERT INTO distance_versions (inchikey, coord, vname, version) VALUES ('%s', '%s', '%s', '%s') ON CONFLICT (inchikey, coord, vname) DO UPDATE SET version = '%s'" % (inchikey, coordinate + "_obs", vname, VERSION, VERSION)
querylite(cmd, checkerconfig.SQLITE_DATABASE_RELEASES)
