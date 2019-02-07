#!/miniconda/bin/python


# Imports

from __future__ import division
import sys, os
import numpy as np
import bisect
import h5py
import itertools
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import Psql
import checkerconfig
from checkerUtils import logSystem,log_data,coordinate2mosaic,inchikey2webrepo_no_mkdir,all_coords
import collections
from scipy.stats import rankdata
import random
import argparse
random.seed(42)

# Variables

PATH = os.path.dirname(os.path.realpath(__file__))

inchikeys_file = "inchikeys.tsv"
inchikeys_indices_file = "inchikeys_indices.h5"
p_e_c_file = "p_e_c.h5"
p_clust_e_c_file = "p_clust_e_c.h5"
clustw_file = "clustw.h5"
kappa_file = "kappa.h5"
cca_file = "cca.h5"
rbo_file = "rbo.h5"
likelihood_file = "likelihood.h5"
handmade_file = "handmade.h5"
clust_memb_file = "cluster_membership.h5"

min_common = 3
min_boundaries = 5

equalize = True

min_handmade = 0.001
max_handmade = 0.1


# Normalization

def quantile_normalization_of_corr_matrix(H, expected_value = None):
    
    vals = []
    idxs = []
    for i in xrange(H.shape[0] - 1):
        for j in range(i+1, H.shape[1]):
            if np.isnan(H[i,j]) or np.isnan(H[j,i]): continue
            vals += [np.mean([H[i,j], H[j,i]])]
            idxs += [(i,j)]

    if expected_value is None:
        ranks = rankdata(np.array(vals))
        ranks = (ranks - np.min(ranks)) / (np.max(ranks) - np.min(ranks))
    else:
        ranks = rankdata(np.array(vals + [expected_value]))
        exp_rank = ranks[-1]
        ranks = ranks[:-1]
        ranks = (ranks - exp_rank) / (np.max(ranks) - exp_rank)
    
    Hn = np.full(H.shape, np.nan)
    for k, r in zip(idxs, ranks):
        Hn[k[0], k[1]] = r
        Hn[k[1], k[0]] = r

    return Hn


# Fetch canonical correlation analysis

def fetch_cca(dbname):
    coords = list(all_coords())
    O1 = np.full((len(coords), len(coords)), np.nan)
    O2 = np.full((len(coords), len(coords)), np.nan)
    R1 = np.full((len(coords), len(coords)), np.nan)
    R2 = np.full((len(coords), len(coords)), np.nan)
    for r in Psql.qstring("SELECT coord_a, coord_b, o1, o2, cv_r1, cv_r2 FROM coordinate_correlation ORDER BY (coord_a, coord_b)", dbname):
        i, j = coords.index(r[0]), coords.index(r[1])
        O1[i,j] = r[2]
        O1[j,i] = r[2]
        O2[i,j] = r[3]
        O2[j,i] = r[3]
        R1[i,j] = r[4]
        R1[j,i] = r[4]
        R2[i,j] = r[5]
        R2[j,i] = r[5]
    with h5py.File("%s/models/%s" % (PATH, cca_file), "w") as hf:
        hf.create_dataset("Odds_1", data = O1)
        hf.create_dataset("Odds_2", data = O2)
        hf.create_dataset("Rho_1" , data = R1)
        hf.create_dataset("Rho_2" , data = R2)

# Do Rank-Biased overlap

def fetch_rbo(dbname):
    coords = list(all_coords())
    RBO = np.full((len(coords), len(coords)), np.nan)
    for r in Psql.qstring("SELECT coord_a, coord_b, rbo FROM coordinate_ranks WHERE n_used > 5  ORDER BY (coord_a, coord_b)", dbname):
        i, j = coords.index(r[0]), coords.index(r[1])
        RBO[i,j] = r[2]
        RBO[j,i] = r[2]
    RBOn = quantile_normalization_of_corr_matrix(RBO)
    with h5py.File("%s/models/%s" % (PATH, rbo_file), "w") as hf:
        hf.create_dataset("RBO", data = RBO)
        hf.create_dataset("RBOn", data = RBOn)

# Handmade correlation

def handmade_correlation(likelis):
    logps =[]
    for p in pvalues:
        if p == 0:
            logps += [-6]
            continue
        logps   += [np.log10(p)]
    logprange    = np.arange(np.log10(min_handmade), np.log10(max_handmade), 0.1)
    likelisrange = np.interp(logprange, logps, np.log2(likelis))
    return np.trapz(likelisrange, logprange)


def fetch_handmade(dbname):
    coords = list(all_coords())
    H  = np.full((len(coords), len(coords)), np.nan)
    with h5py.File("%s/models/%s" % (PATH, handmade_file), "w") as hf:
        # Likelihood ratio
        for i, coord_e in enumerate(coords):
            for j, coord_c in enumerate(coords):
                if i == j: continue
                integers, likelis = [], []
                for r in Psql.qstring("SELECT integer, l_e_c FROM coordinate_paired_conditionals WHERE coord_e = '%s' AND coord_c = '%s' AND l_e_c IS NOT NULL AND a >= %d AND (a+b) >=  %d AND (a+c) >= %d ORDER BY (integer)" % (coord_e, coord_c, min_common, min_boundaries, min_boundaries), dbname):
                    integers += [r[0]]
                    likelis  += [r[1]]
                if not integers: continue
                H[i,j] = handmade_correlation(np.interp(all_integers, integers, likelis))

        # A quantile-normalized version

        Hn = quantile_normalization_of_corr_matrix(H, expected_value = 0.)

        # Save dataset
        hf.create_dataset("H", data = H)
        hf.create_dataset("Hn", data = Hn)


# Fetch conditional probabilities

def fetch_conditional_probabilities(dbname):
    e_integers = [r[0] for r in Psql.qstring("SELECT DISTINCT(integer_e) AS g FROM coordinate_conditionals ORDER BY (g)", dbname)]
    c_integers = [r[0] for r in Psql.qstring("SELECT DISTINCT(integer_c) AS g FROM coordinate_conditionals ORDER BY (g)", dbname)]
    P  = np.zeros((len(e_integers), len(c_integers), len(coords), len(coords)))
    nP = np.zeros((len(e_integers), len(c_integers), len(coords), len(coords)))
    with h5py.File("%s/models/%s" % (PATH, p_e_c_file), "w") as hf:
        hf.create_dataset("c_integers", data = c_integers)
        hf.create_dataset("e_integers", data = e_integers)
        p_c = np.zeros((len(c_integers), len(coords)))
        p_e = np.zeros((len(e_integers), len(coords)))
        coords_idxs = dict((c, i) for i, c in enumerate(coords))
        # P(C) differs depending on the coordinate, for now, we take the mean...
        R = list(set([r for r in Psql.qstring("SELECT integer_c, coord_c, log_p_c FROM coordinate_conditionals", dbname)]))
        d = collections.defaultdict(list)
        for r in R: d[(r[0], r[1])] += [r[2]]
        for k,v in d.iteritems(): p_c[bisect.bisect_left(c_integers, k[0]), coords_idxs[k[1]]] = np.mean(v)
        # Idem for P(E)
        R = list(set([r for r in Psql.qstring("SELECT integer_e, coord_e, log_p_e FROM coordinate_conditionals", dbname)]))
        d = collections.defaultdict(list)
        for r in R: d[(r[0], r[1])] += [r[2]]
        for k,v in d.iteritems(): p_e[bisect.bisect_left(e_integers, k[0]), coords_idxs[k[1]]] = np.mean(v)
        # Conditional probability
        for i, coord_e in enumerate(coords):
            for j, coord_c in enumerate(coords):
                if i == j: continue
                v = [r[0] for r in Psql.qstring("SELECT log_p_e_c FROM coordinate_conditionals WHERE coord_e = '%s' AND coord_c = '%s' ORDER BY (integer_e, integer_c)" % (coord_e, coord_c), dbname)]
                v = np.reshape(v, (len(e_integers), len(c_integers)))
                P[:,:,i,j] = v
        # Conditional probability of the opposite class
        for i, coord_e in enumerate(coords):
            for j, coord_c in enumerate(coords):
                if i == j: continue
                v = [r[0] for r in Psql.qstring("SELECT log_p_e_nc FROM coordinate_conditionals WHERE coord_e = '%s' AND coord_c = '%s' ORDER BY (integer_e, integer_c)" % (coord_e, coord_c), dbname)]
                v = np.reshape(v, (len(e_integers), len(c_integers)))
                nP[:,:,i,j] = v
        # Save datasets     
        hf.create_dataset("pec" , data = P  )
        hf.create_dataset("penc", data = nP )
        hf.create_dataset("pc"  , data = p_c)
        hf.create_dataset("pe"  , data = p_e)

# Fetch conditional paired probabilities
# We wildly interpolate (by integer, not by nominal value...)

def fetch_conditional_paired_probabilites(dbname):

    # Save this in natural log units, so that the guilter/mapper can compute faster.

    L   = np.full((len(all_integers), len(coords), len(coords)), 0.)
    with h5py.File("%s/models/%s" % (PATH, likelihood_file), "w") as hf:
        # Likelihood ratio
        hf.create_dataset("integers", data = all_integers)
        for i, coord_e in enumerate(coords):
            for j, coord_c in enumerate(coords):
                if i == j: continue # The relative likelihood of yourself is 0. Convenient for the predictor.
                integers, likelis = [], []
                for r in Psql.qstring("SELECT integer, log_l_e_c FROM coordinate_paired_conditionals WHERE coord_e = '%s' AND coord_c = '%s' AND log_l_e_c IS NOT NULL AND a >= %d AND (a+b) >=  %d AND (a+c) >= %d ORDER BY (integer)" % (coord_e, coord_c, min_common, min_boundaries, min_boundaries), dbname):
                    integers += [r[0]]
                    likelis  += [r[1]]
                if not integers: continue
                L[:,i,j] = np.interp(all_integers, integers, likelis)

        # Save dataset
        hf.create_dataset("likeli", data = L)
        inttype = np.int8
        intmax = np.iinfo(inttype).max
        L_10x = L*10
        L_10x[L_10x > intmax] = intmax
        hf.create_dataset("likeli_10x", data = L_10x.astype(inttype)) # This is convenient for the guilter.


# Fetch conditional clust probabilities

def get_cluster_weights(dbname):
    
    coords = all_coords()

    clusts = set()

    for coord in coords:
        with h5py.File(coordinate2mosaic(coord)+ "clust.h5", "r") as hf:
            labs = hf["labels"][:]
            clusts.update(hf["labels"][:])
    
    clusts = sorted(clusts)

    Clustw = np.zeros((clusts[-1]+1, clusts[-1]+1, len(coords), len(coords))) # The weight of the clustering.

    for i, coord_e in enumerate(coords):
        for j, coord_c in enumerate(coords):
            integers = []
            likelis  = []
            d = collections.defaultdict(list)
            for r in Psql.qstring("SELECT clust_e, clust_c, integer, l_e_c FROM coordinate_clust_paired_conditionals WHERE l_e_c IS NOT NULL AND coord_e = '%s' AND coord_c = '%s' AND a >= %d AND (a+b) >= %d AND (a+c) >= %d ORDER BY (integer)" % (coord_e, coord_c, min_common, min_boundaries, min_boundaries), dbname):
                d[(r[0], r[1])] += [(r[2], r[3])]
            for k,v in d.iteritems():
                integers = [x[0] for x in v]
                likelis  = [x[1] for x in v]
                likelis = np.interp(all_integers, integers, likelis)
                c = np.log(2**max(0, handmade_correlation(likelis))) # Handmade correlations were given in log2 scale. Here, for consistency with regular L(E|C), use natural log.
                Clustw[k[0], k[1], i, j] = c

    with h5py.File("%s/models/%s" % (PATH, clustw_file), "w") as hf:
        hf.create_dataset("Clustw", data = Clustw)
        inttype = np.int8
        intmax = np.iinfo(inttype).max
        Clustw_10x = Clustw*10
        Clustw_10x[Clustw_10x > intmax] = intmax
        hf.create_dataset("Clustw_10x", data = Clustw_10x.astype(inttype))


def cluster_membership():

    coords = all_coords()

    inchikeys = set()
    coords_inchikeys = {}
    coords_labels = {}

    for coord in coords:
        with h5py.File(coordinate2mosaic(coord)+ "clust.h5", "r") as hf:
            coords_labels[coord] = hf["labels"][:]
            iks = hf["keys"][:]
            coords_inchikeys[coord] = iks
            inchikeys.update([ik for ik in iks])
    inchikeys = sorted(inchikeys)

    with h5py.File("%s/models/%s" % (PATH, clust_memb_file), "w") as hf:
        Cm = np.full((len(inchikeys), len(coords)), -99).astype(np.int16) # Check that, some day, we don't have more than np.int16 clusters! (quite unlikely)
        for j in xrange(len(coords)):
            coord = coords[j]
            labels = coords_labels[coord]
            idxs = [bisect.bisect_left(inchikeys, ik) for ik in coords_inchikeys[coord]]
            for idx, label in zip(idxs, labels):
                Cm[idx, j] = label
        
        hf.create_dataset("Cm", data = Cm)
        hf.create_dataset("keys", data = np.array(inchikeys))

# Fetch contingency table

def contingency_fetcher(coord_e, coord_c,dbname):
    R = Psql.qstring("SELECT integer_e, integer_c, common_pairs FROM coordinate_conditionals WHERE coord_e = '%s' AND coord_c = '%s' ORDER BY (integer_e, integer_c)" % (coord_e, coord_c), dbname)
    common_integers = sorted(set([r[0] for r in R]).intersection([r[1] for r in R]))
    s = set(common_integers)
    C = np.zeros((len(common_integers), len(common_integers)))
    i, j = 0, 0
    for r in R:
        if r[0] not in s or r[1] not in s: continue
        C[bisect.bisect_left(common_integers, r[1]), bisect.bisect_left(common_integers, r[0])] = r[2]
    return C
        
# Given a very imbalanced contingency table, weight it so that rows and columns add up to a desired quantity (typically 1)

def equalizer(C, weight_by_probability = True):
    
    C += 1
    
    if weight_by_probability:
        S = np.sum(C)
        row_w = [np.log10(np.sum(C[i,:]) / S) for i in xrange(C.shape[0])]
        col_w = [np.log10(np.sum(C[:,j]) / S) for j in xrange(C.shape[1])]
    else:
        row_w = [1. for _ in xrange(C.shape[0])]
        col_w = [1. for _ in xrange(C.shape[1])]
    N = np.array(C)
    for _ in xrange(1000):
        # Rows
        for i in xrange(N.shape[0]):
            N[i,:] = row_w[i]*N[i,:]/np.linalg.norm(N[i,:], ord = 1)
        # Columns
        for j in xrange(N.shape[1]):
            N[:,j] = col_w[j]*N[:,j]/np.linalg.norm(N[:,j], ord = 1)
    return N

# Kappa score, with a typical quadratic weighting

def kappa(C, weights = "quadratic"):

    if equalize: C = equalizer(C)
    
    n_classes = C.shape[0]
    sum0 = np.sum(C, axis=0)
    sum1 = np.sum(C, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
        
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * C) / np.sum(w_mat * expected)
    return 1 - k


def kappa_matrix(dbname):
    with h5py.File("%s/models/%s" % (PATH, kappa_file), "w") as hf:
        d = collections.defaultdict(list)
        for coord_e in coords:
            for coord_c in coords:
                k = kappa(contingency_fetcher(coord_e, coord_c,dbname))
                d[tuple(sorted([coord_e, coord_c]))] += [k]
        K = np.zeros((len(coords), len(coords)))
        for k, v in d.iteritems():
            v = np.mean(v)
            i = list(coords).index(k[0])
            j = list(coords).index(k[1])
            K[i,j] = v
            K[j,i] = v
        Kn = quantile_normalization_of_corr_matrix(K)
        hf.create_dataset("coords", data = coords)
        hf.create_dataset("K", data = K)
        hf.create_dataset("Kn", data = Kn)


# V matrix

def placeholder_matrix(vname):
    inchikeys = set()
    coords_inchikeys = {}
    for coord in coords:
        with h5py.File("%s/%s.h5" % (coordinate2mosaic(coord), vname), "r") as hf:
            iks = hf["keys"][:]
            coords_inchikeys[coord] = iks
            inchikeys.update([ik for ik in iks])
    inchikeys = sorted(inchikeys)
    with h5py.File("%s/models/%s" % (PATH, inchikeys_indices_file), "w") as hf:
        V = np.full((len(inchikeys), len(coords)), -99).astype(np.int8)
        hf.create_dataset("keys", data = np.array(inchikeys))
        for j in xrange(len(coords)):
            coord = coords[j]
            idxs = np.array([bisect.bisect_left(inchikeys, ik) for ik in coords_inchikeys[coord]]).astype(np.int32)
            hf.create_dataset(coord, data = idxs)
            for i in idxs: V[i, j] = -1
        hf.create_dataset("V", data = V)
    with open("%s/models/%s" % (PATH, inchikeys_file), "w") as f:
        for ik in inchikeys:
            f.write("%s\n" % ik)


# Main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    global PATH

    parser.add_argument("--vname", type = str, default = "sig", help = "Name of the vectors / similarities datasets.")
    parser.add_argument("--dbname", type = str, default = "mosaic", help = "Name of the database.")
    parser.add_argument("--path", type = str, default = PATH, help = "Path where data models will be saved")
    args = parser.parse_args()

    pvalues = checkerconfig.PVALRANGES
    PATH = args.path
    all_integers = np.array([i for i in xrange(len(pvalues))])
    
    modelsDir = os.path.join(PATH,"models")
    if not os.path.exists(modelsDir):
        os.makedirs(modelsDir)

    coords = all_coords()

    print "Calculating handmade correlations"
    fetch_handmade(args.dbname)

    print "Fetching conditional probabilities"
    fetch_conditional_probabilities(args.dbname)

    print "Fetching paired conditional probabilities"
    fetch_conditional_paired_probabilites(args.dbname)

    print "Fetching cluster paired conditional probabilities"
    get_cluster_weights(args.dbname)

    print "Calculating Kappa correlations"
    kappa_matrix(args.dbname)

    print "Fetching CCA"
    fetch_cca(args.dbname)

    print "Fetching RBO"
    fetch_rbo(args.dbname)

    print "V file"
    placeholder_matrix(args.vname)

    print "Cluster membership file"
    cluster_membership()
