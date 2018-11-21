#!/miniconda/bin/python


# Imports

from scipy.spatial.distance import cdist
import sys
import os
import h5py
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0], "../../pipeline/config"))
from checkerUtils import inchikey2webrepo
import numpy as np
import bisect
import time

# Compare molecules

coordinate, from_j, to_j, infile, bgfile, outfile, vname, VERSION, batch_ref = sys.argv[1].split('---')

from_j, to_j = int(from_j), int(to_j)

# Numerical precision

numerical_precision = 10

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
    N = len(hf["keys"][:])
    my_V = hf["V"][from_j:to_j]
    my_iks = hf["keys"][from_j:to_j]
    INTEGERS = np.zeros((N, my_V.shape[0]), np.int8)
    from_i = 0
    while from_i < N:
        to_i = from_i + chunksize
        INTEGERS[from_i:to_i, :] = get_integer(np.round(cdist(hf["V"][from_i:to_i], my_V, metric="cosine"), numerical_precision))
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
            if dname in hf.keys():
                del hf[dname]
            hf.create_dataset(dname, data=INTEGERS[:, j])
            if dversion in hf.keys():
                del hf[dversion]
            hf.create_dataset(dversion, data=[VERSION])

    except:

        print "Error when saving %s: %s/%s" % (ik, PATH, outfile)
