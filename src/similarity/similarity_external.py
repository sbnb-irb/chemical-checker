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

# Compare molecules


coordinate, from_j, to_j, infolder_infile, infile, bgfile, outfile, vname, VERSION, batch_ref = sys.argv[1].split('---')

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

with h5py.File(infolder_infile, "r") as hf:
    my_V = hf["V"][from_j:to_j]
    my_iks = hf["keys"][from_j:to_j]

with h5py.File(infile, "r") as hf:
    N = len(hf["keys"][:])
    INTEGERS = np.zeros((N, my_V.shape[0]), np.int8)
    from_i = 0
    while from_i < N:
        to_i = from_i + chunksize
        INTEGERS[from_i:to_i, :] = get_integer(np.round(cdist(hf["V"][from_i:to_i], my_V, metric="cosine"), numerical_precision))
        from_i += chunksize


INTEGERS = np.array(INTEGERS).astype(np.int8)

VALUES = []

for j, ik in enumerate(my_iks):

    PATH = inchikey2webrepo(ik)

    try:

        with h5py.File("%s/%s" % (PATH, outfile), "a") as hf:
            dname = coordinate + "_obs"
            if dname in hf.keys():
                del hf[dname]
            hf.create_dataset(dname, data=INTEGERS[:, j])

    except:

        print "Error when saving %s: %s/%s" % (ik, PATH, outfile)
