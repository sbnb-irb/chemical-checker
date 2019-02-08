#!/miniconda/bin/python

# Imports    

import sys, os
import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import random
random.seed(42)
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import Psql
import checkerconfig
from checkerUtils import logSystem,log_data,coordinate2mosaic,inchikey2webrepo_no_mkdir
from scipy.stats import sem

# Variables


B = 10000 # Number of molecules to sample for the expected distributions.
k = 5 # Number of clusters for the distributions.

pvals = checkerconfig.PVALRANGES

# These are the weights that we put to the Euclidean distances

min_pval = np.min(pvals[pvals > 0]) / 10.
pvals[pvals == 0] = min_pval
weights   = -np.log10(pvals)
weights   = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) + 1e-10

# Main

def main(coord, vname, versionpath):

    with h5py.File("%s/indices/%s.h5" % (versionpath, vname), "r") as hf:
        iks = hf[coord][:]

    C = np.zeros((B, len(pvals))).astype(np.int)

    for i, ik in enumerate(random.sample(iks, np.min([B, len(iks)]))):

        with h5py.File("%s/%s.h5" % (inchikey2webrepo_no_mkdir(ik), vname), "r") as hf:
            s = hf["%s_obs" % coord][:]
            for j in s:
                C[i,j] += 1

    C[:,0] -= 1 # Remove yourself

    means = np.mean(C, axis = 0)

    C_scaled = np.zeros(C.shape)

    for j in xrange(C.shape[1]):
        if means[j] == 0: continue
        C_scaled[:,j] = C[:,j] / means[j] * weights[j]

    clusts = MiniBatchKMeans(n_clusters = k, compute_labels = False)
    X = clusts.fit_transform(C_scaled)

    centroids = np.zeros((k, C_scaled.shape[1]))

    for j in xrange(X.shape[1]):
        centroids[j,:] = C_scaled[np.argmin(X[:,j]),:]

    # Save

    with h5py.File(coordinate2mosaic(coord) + "/models/sampled_%s_distances.h5" % vname, "w") as hf:
        hf.create_dataset("means", data = means)
        hf.create_dataset("weights", data = weights)
        hf.create_dataset("centroids", data = centroids)


if __name__ == '__main__':
	coord, vname, versionpath = sys.argv[1].split("---")
	main(coord, vname, versionpath)

