#!/miniconda/bin/python


# Imports

import subprocess
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
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
import time

# Compare molecules

#coordinate, inchikey, infile, bgfile, outfile, vname, VERSION = sys.argv[1].split('---')
coordinate, from_j, to_j, infile, bgfile, outfile, vname, VERSION, batch_ref = sys.argv[1].split('---')

from_j, to_j = int(from_j), int(to_j)

print coordinate, vname, from_j, to_j

chunksize = int(batch_ref)


# Read pvalues

with h5py.File(bgfile, "r") as hf:
    distances = hf["distance"][:]
    max_idx = len(distances) - 1

def get_integer(d):
    return bisect.bisect_left(distances, d)

get_integer = np.vectorize(get_integer)


# Compute similarities

time0 = time.time()

with h5py.File(infile, "r") as hf:
    N        = len(hf["inchikeys"][:])
    my_V     = hf["V"][from_j:to_j]
    my_iks   = hf["inchikeys"][from_j:to_j]
    INTEGERS = np.zeros((N, my_V.shape[0]), np.int8)
    from_i = 0
    while from_i < N:
        to_i = from_i + chunksize
        INTEGERS[from_i:to_i, :] = get_integer(cdist(hf["V"][from_i:to_i], my_V, metric = "cosine"))
        from_i += chunksize

INTEGERS[INTEGERS > max_idx] = max_idx

time1 = time.time()

print "- Similarities took:", round(time1 - time0), "secs"

# Store results

VALUES = []

for j, ik in enumerate(my_iks):

    PATH = inchikey2webrepo(ik)

    try:

        with h5py.File("%s/%s" % (PATH, outfile), "a") as hf:
            dname = coordinate + "_obs"
            dversion = dname + "_vrs"
            if dname in hf.keys(): del hf[dname]
            hf.create_dataset(dname, data = INTEGERS[:,j])
            if dversion in hf.keys(): del hf[dversion]
            hf.create_dataset(dversion, data = [VERSION])



    except:

        print "Error when saving %s: %s/%s" % (ik, PATH, outfile)


# Inserting the current version in the database

#cmd = "INSERT INTO distance_versions (inchikey, coord, vname, version) VALUES ('%s', '%s', '%s', '%s') ON CONFLICT (inchikey, coord, vname) DO UPDATE SET version = '%s'" % (inchikey, coordinate + "_obs", vname, VERSION, VERSION)
#querylite(cmd, checkerconfig.SQLITE_DATABASE_RELEASES)
