
# coding: utf-8

# # Infer similarities.
# 
# # Use probabilities to predict the similarity of small molecules in different spaces.

# Imports

from __future__ import division
import sys, os
sys.path.append(os.path.join(sys.path[0], "../dbutils/"))
import Psql
import h5py
import bisect
import numpy as np
import itertools
from tqdm import tqdm
from cuts import performance_integers as c_integers
import CONFIG
from scipy.spatial.distance import cdist
import collections

# Paths of precomputed data

PATH = os.path.dirname(os.path.realpath(__file__))

inchikeys_indices_file  = PATH + "/models/inchikeys_indices.h5"
likelihood_file         = PATH + "/models/likelihood.h5"
coordcorr_file          = PATH + "/models/kappa.h5"
cluster_membership_file = PATH + "/models/cluster_membership.h5"
cluster_weights_file    = PATH + "/models/clustw.h5"

inttype = np.int16
intmax  = np.iinfo(inttype).max

# Variables

max_l_cutoff = 100
min_l_cutoff = 20

# Read p-values

pvalues = CONFIG.PVALRANGES
min_pvalue = np.min(pvalues[pvalues > 0]) / 10.

null = bisect.bisect_left(pvalues, 0.5) # In case of non-prediction, put the integer that has a p-value of 0.5 (median).

# Coordinates

coords = CONFIG.all_coords()

# Correlations between datasets. I take the Kappa correlation.

def load_correlations():
    with h5py.File(coordcorr_file, "r") as hf:
        K = hf["K"][:]
        K[K < 0] = 0 # So that we don't fool calculations of weights.
    return K

# ### Load conditional probabilities

# ## Experimental data of the molecule of interest
# 
# Load data experimentally observed.
# 
# -99 means "target molecule not in coordinate", -1 means "target molecule in coordinate,
# but no experimental similarity because the query molecule is not in this coordinate"


def load_experimental_data(PATH):

    # Read experimentals similars data

    inchikeys_indices = {}
    with h5py.File(inchikeys_indices_file, "r") as hf:
        for coord in coords:
            inchikeys_indices[coord] = hf[coord][:]
        V = hf["V"][:]
        inchikeys = hf["inchikeys"][:]
        
    with h5py.File(PATH + "/%s.h5" % vname, "r") as hf:
        obs = set(hf.keys())
        for j in xrange(len(coords)):
            coord = coords[j]
            if coord + "_obs" not in obs: continue
            v = hf[coord + "_obs"][:]
            idxs = inchikeys_indices[coord]
            for l in xrange(len(v)):
                V[idxs[l],j] = v[l]
                
    return V, inchikeys


def load_cluster_membership():

    with h5py.File(cluster_membership_file, "r") as hf:
        Cm = hf["Cm"][:]

    return Cm


def load_cluster_weights():

    with h5py.File(cluster_weights_file, "r") as hf:
        Cw = hf["Clustw_10x"][:]

    return Cw


def load_likelihood(): # These are in natural log scale, and multiplied by 10

    with h5py.File(likelihood_file, "r") as hf:
        L = hf["likeli_10x"][:]
 
    return L


def weights(e_idxs):

    if len(e_idxs) == 1:
        return np.array([1.])

    s = len(e_idxs) - np.sum(K[e_idxs,:][:,e_idxs], axis = 1)

    return len(e_idxs)/np.sum(s)*s


def expected_distributions(V):

    # Distribution (expected)

    ExpDistros = np.zeros((len(pvalues), len(coords))).astype(np.int)

    if expected_bg_euclideans:

        # Distributions based on the clusters of distributions computed by 'expected_distributions.py'

        Centroids = np.zeros((len(pvalues), len(coords)))

        obs_idxs = collections.defaultdict(set)

        for j, coord in enumerate(coords):

            obs = np.where(V[:,j] > -1)[0]

            if len(obs) == 0: continue

            obs_idxs[j] = set(obs)

            with h5py.File(CONFIG.coordinate2mosaic(coord) + "/models/sampled_%s_distances.h5" % vname, "r") as hf:
                m = hf["means"][:]
                w = hf["weights"][:]
                centroids = hf["centroids"][:]

            # Observed distribution
            c = np.zeros(len(pvalues))
            for o in V[obs,j]: c[o] += 1
            c[0] -= 1 # Remove yourself
            C = np.zeros((1,len(c)))
            # Do the scaling
            for i in xrange(len(C)):
                if m[i] == 0: continue
                C[0,i] = c[i] / m[i] * w[i]

            centroid = centroids[cdist(C, centroids).argmin(axis = 1)[0]]
            
            centroid = centroid / w

            Centroids[:,j] = centroid

        for j, coord in enumerate(coords):

            # Consider all the places where the molecule needs to be predicted

            if minus_one_only:
                actuals = set(np.where(V[:,j] == -1)[0])
            else:
                actuals = set(np.where(V[:,j] >= -1)[0])
            
            if len(actuals) == 0: continue

            # The indices of the observed (excluding oneself)
            
            e_idxs = sorted(set(obs_idxs.keys()).difference([j]))
            if len(e_idxs) == 0: continue

            # Weights according to the Kappa
            
            wk  = weights(e_idxs)
            wks = np.sum(wk)
            if wks > 0:
                wk = wk / wks

            # Weights according to the coverage
            
            wc = np.zeros(len(wk))
            for i, e_idx in enumerate(e_idxs):
                wc[i] = len(actuals.intersection(obs_idxs[i]))
            wcs = np.sum(wc)
            if wcs > 0:
                wc = wc / wcs

            np.average(Centroids[:,e_idxs], weights = wc*wk, axis = 1)

            #print centroid
            #print centroid.shape
            #centroid = [c[i] * m[i] / w[i] for i in xrange(len(centroid))]

            #if n == 0: continue

    if np.sum(ExpDistros) == 0:

        # These are just the distributions from the expected background euclideans

        for j, coord in enumerate(coords):
   
            # Consider all the places where the molecule needs to be predicted.

            if minus_one_only:
                n = np.sum(V[:,j] == -1)
            else:
                n = np.sum(V[:,j] >= -1)

            if n == 0: continue

            with h5py.File(CONFIG.coordinate2mosaic(coord) + "/models/sampled_%s_distances.h5" % vname, "r") as hf:
                m = hf["means"][:]

            expdistro = (m / np.sum(m) * n).astype(np.int)

            expdistro[null] += (n - np.sum(expdistro))

            ExpDistros[:,j] = expdistro

    sys.exit()

    return ExpDistros


def threader(Yl, ExpDistros):
    
    # Declare the empty output

    Y = np.full(Yl.shape, -99).astype(np.int8)
    Y[Yl != -99] = -1

    # Given ranks of predictions (based on cumulative likelihood estimates), adjust (thread) it to the expected distribution.
    
    # Iterate over squares
    for j in xrange(Y.shape[1]):

        exp_distro = ExpDistros[:,j] # Expected distribution.
        idxs = np.where(Yl[:,j] >= 0)[0] # Those positions in Yl where there are predictions available.
        vals = Yl[idxs, j] # The values of the likelihoods
        argsorts = np.argsort(-vals) # Sort the values by likelihood
        sorted_idxs = idxs[argsorts] # The sorted list of indices

        previous_counts = 0
        counts = 0
        
        # Iterate over integers
        for i, local_counts in enumerate(exp_distro):
            counts += local_counts
            myidxs = sorted_idxs[previous_counts:counts]
            Y[myidxs,j] = i
            previous_counts = counts
            #ExpDistros[i,j] = ExpDistros[i,j] - len(myidxs)

    return Y

     
# Functions to make predictions

def predict_with_inchikey(V):

    # In this case, minus one only.

    # Match by connectivity layer of the inchikey string.

    connectivity   = query_inchikey.split("-")[0]
    connectivities = [ik.split("-")[0] for ik in inchikeys]

    # Define a template matrix.

    Y = np.array(V).astype(np.int8)

    # Iterate over observations looking for matches in connectivities

    for i, v in enumerate(V):

        c_idxs = np.where(v == -1)[0]
        if len(c_idxs) == 0: continue

        # Assume that molecules with the same connectivity will have the same prediction (ahem...)
        # For simplicity, this has nothing to do with the expected distributions.
        if connectivity == connectivities[i]:
            Y[i, c_idxs] = 0

    return Y


def predict(V, ExpDistros, is_last, l_cutoff = None):

    # Empty cumulative likelihood dataset

    Yl = np.full(V.shape, -99).astype(inttype)
    Yl[V != -99] = -1

    # Functions

    if is_last and not minus_one_only:
        
        # In the last step, and if not specified, 

        def sanitizer(Yl):
            Yl[Yl == -1] = 0
            return Yl

        def c_idxer(v):
            return np.where(v != -99)[0]

        def merger(V, Y):
            Y[Y == -1] = null
            return Y
   
    else:

        # In iterative steps, do minus one only

        def sanitizer(Yl):
            return Yl

        def c_idxer(v):
            return np.where(v == -1)[0]

        def merger(V, Y):
            mask = V >= 0
            Y[mask] = V[mask]
            return Y

    if is_last:

        # If is last, always assign a value

        def assigner(val):
            return val

    else:

        # In iterative steps, only assign a value if below a certain cutoff

        def assigner(val):
            if val < l_cutoff: return -1
            return val 

    # Start iterating over molecules

    for i, v in tqdm(enumerate(V)):

        # Where is it worth doing a prediction?

        c_idxs = c_idxer(v)
        if len(c_idxs) == 0: continue

        # Where do we have experimental data?

        e_idxs = np.where(v >= 0)[0]
        if len(e_idxs) == 0: continue

        # Get pre-computed weights (from the correlation matrix)

        w = weights(e_idxs)

        # Cluster memberships

        cms_e = Cm[i, e_idxs]

        # Cluster weights in log scale

        clustw = np.zeros((len(e_idxs), len(c_idxs)))
        for ii, x in enumerate(zip(e_idxs, cms_e)):
            for jj, y in enumerate(zip(c_idxs, cm_c[c_idxs])):
                clustw[ii,jj] = Cw[x[1], y[1], x[0], y[0]]

        # Likelihood ratios in log scale

        lec = np.zeros((len(e_idxs), len(c_idxs)))
        for idx, e_idx in enumerate(e_idxs):
            lec[idx] = L[v[e_idx], e_idx, c_idxs]

        # Make predictions

        for j, c_idx in enumerate(c_idxs):

            Yl[i,c_idx] = assigner(min([intmax, int(max([np.sum(lec[:,j].T*w), np.sum(clustw[:,j].T*w)]))])) # I just take the best between similars and clusters.

    # Callibrate

    Y = threader(sanitizer(Yl), ExpDistros)

    # Merge with observational data. If we are in the 'minus one only' case, then 

    Y = merger(V, Y)

    # Done

    return Y, ExpDistros


def iterative_prediction(V, max_iter = 5, epsilon = 0.001):

    print "    -- Predict with inchikey"

    Y = predict_with_inchikey(V)

    print "    -- Start iteration"

    ExpDistro = expected_distributions(Y)

    print ExpDistro
    print np.sum(ExpDistro, axis = 1)

    sys.exit()

    l_cutoff = max_l_cutoff
            
    all_min1 = np.sum(V == -1)

    bef_min1 = all_min1
    for _ in xrange(max_iter):
        Y, ExpDistro = predict(Y, ExpDistro, is_last = False, l_cutoff = l_cutoff)
        ExpDistro = expected_distributions(Y)
        now_min1 = np.sum(Y == -1)
        if (bef_min1 - now_min1) / all_min1 < epsilon: break
        bef_min1 = now_min1
        if l_cutoff > min_l_cutoff: l_cutoff = int(l_cutoff*0.9)
    
    print "    -- Final prediction"

    Y, ExpDistro = predict(Y, ExpDistro, is_last = True)

    return Y


# Performance functions

def metrics(TRUE, PRED, ALLS):

    # Contingency
    tp  = len(PRED.intersection(TRUE))
    fp  = len(PRED) - tp
    fn  = len(TRUE) - tp
    tn  = len(ALLS) - (tp + fp + fn)

    # Performance scores
    if (tp + fp) == 0:
        ppv = -99
    else: 
        ppv  = tp / (tp + fp)
    if (tp + fn) == 0:
        sens = -99
    else:
        sens = tp / (tp + fn)
    if (tn + fp) == 0:
        spec = -99
    else:
        spec = tn / (tn + fp)
    if (sens == -99 or spec == -99):
        bacc = -99
    else:
        bacc = 0.5 * (sens + spec)
    if (ppv == -99 or sens == -99 or (ppv + sens) == 0):
        f1   = -99
    else:         
        f1   = 2*(ppv * sens)/(ppv + sens)

    N = tn + tp + fn + fp
    if N == 0:
        mcc = -99
    else:
        S = (tp + fn) / N
        P = (tp + fp) / N
        sqrt = np.sqrt(P*S*(1-S)*(1-P))
        if sqrt == 0:
            mcc = -99
        else:
            mcc = (tp/N - S*P) / sqrt

    return len(TRUE), len(PRED), tp, fp, fn, tn, ppv, sens, spec, bacc, f1, mcc


def performance(inchikey, V, Y):

    S = []

    for j, coord_c in enumerate(coords):

        ALLS = np.where(np.logical_and(V[:,j] >= 0, Y[:,j] >= 0))[0]

        if len(ALLS) == 0: continue # No observations/predictions here

        for c_integer in c_integers:

            mets  = []            

            PRED  = set(np.where(Y[:,j] <= c_integer)[0]).intersection(ALLS)
            TRUE  = set(np.where(V[:,j] <= c_integer)[0]).intersection(ALLS)
      
            mets += list(metrics(TRUE, PRED, ALLS))

            S += ["('%s', '%s', '%s', %d, %.2g, %d, %d, %d, %d, %d, %d, %.2f,  %.2f, %.2f, %.2f, %.2f, %.2f)" % tuple([method, inchikey, coord_c, c_integer, pvalues[c_integer]] + mets)]
       
        # Delete if exists (overwrite in the database)
        Psql.query("DELETE FROM mapper_performance WHERE method = '%s' AND inchikey = '%s' AND coord_c = '%s'" % (method, inchikey, coord_c), dbname)

    Psql.query("INSERT INTO mapper_performance VALUES %s" % ",".join(S), dbname)


# Save function

def save(Y, PATH):
    with h5py.File(PATH + "/%s.h5" % vname, "r+") as hf:
        keys = set(hf.keys())
        for key in keys:
            if "_prd" in key: del hf[key]
        for j in xrange(Y.shape[1]):
            mask = Y[:,j] != -99
            if not np.any(mask): continue
            v = np.array([c_integers[i] for i in Y[mask,j]]).astype(np.int8)
            key = "%s_prd" % coords[j]
            hf.create_dataset(key, data = v)


if __name__ == '__main__':

    # Parse arguments

    args = sys.argv[1].split("---")
    
    dbname = args[0]
    vname = args[1]

    if args[2] == "NULL":
        do_performance = False
    else:
        do_performance = True
        method = args[2]

    if args[3] == "mone_y":
        minus_one_only = True
    else:
        minus_one_only = False

    if args[4] == "genex_n":
        expected_bg_euclideans = True
    else:
        expected_bg_euclideans = False

    query_inchikeys = args[5:]

    print "Loading weight correlations"
    K = load_correlations()

    print "Loading likelihoods"
    L = load_likelihood()

    print "Loading cluster memberships"
    Cm = load_cluster_membership()

    print "Loading cluster weights"
    Cw = load_cluster_weights()

    print "Start predictions!"
    for query_inchikey in query_inchikeys:

        print "*** %s" % query_inchikey
        PATH = Psql.inchikey2webrepo_no_mkdir(query_inchikey)

        print "    Loading experimental data"
        V, inchikeys = load_experimental_data(PATH)

        print "    Getting cluster membership of this molecule"
        cm_c = Cm[bisect.bisect_left(inchikeys, query_inchikey)]

        print "    Iterative prediction"
        Y = iterative_prediction(V)

        if do_performance:
            print "    Doing performance"
            performance(query_inchikey, V, Y)

        else:
            print "    Saving"
            save(Y, PATH)