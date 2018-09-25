#!/miniconda/bin/python


# Conditional probabilities and more.

# Imports

from __future__ import division
from sklearn.metrics.cluster.supervised import mutual_info_score
import sys, os
import argparse
import h5py
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import Psql
import checkerconfig
from checkerUtils import logSystem,log_data,coordinate2mosaic,inchikey2webrepo_no_mkdir
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import math
from scipy.stats import chi2_contingency
from intbitset import intbitset
import bisect
import random

# Variables

def f2s(x):
    if np.isnan(x): return 'NULL'
    return x

if __name__ == '__main__':


    coordinate_e, coordinate_c,dbname = sys.argv[1].split("---")
    
    B = 10000
    
    cumulative = False # The cumulative version is better applied in the coordinate_paired_conditionals.py
    
    print coordinate_e, coordinate_c
    
    # Get molecules
    
    with h5py.File(coordinate2mosaic(coordinate_e)+ "sig.h5", "r") as hf:
        iks_e = hf['keys'][:]
    
    with h5py.File(coordinate2mosaic(coordinate_c)+ "sig.h5", "r") as hf:
        iks_c = hf['keys'][:]
    
    if len(iks_e) < len(iks_c):
        do_e = True
    else:
        do_e = False
    
    # Convert inchikeys to integers
    
    inchikeys = sorted(set(iks_e).union(iks_c))
    iks_idxs  = dict((inchikeys[i], i) for i in xrange(len(inchikeys)))
    
    inchikeys_str  = sorted(set(iks_e).intersection(set(iks_c)))
    
    iks_e = [iks_idxs[ik] for ik in iks_e]
    iks_c = [iks_idxs[ik] for ik in iks_c]
    inchikeys = [iks_idxs[ik] for ik in inchikeys_str]
    
    inchikeys_e = dict((iks_e[i], i) for i in xrange(len(iks_e)))
    inchikeys_c = dict((iks_c[i], i) for i in xrange(len(iks_c)))
    
    all_idxs_e = intbitset([inchikeys_e[ik] for ik in inchikeys])
    all_idxs_c = intbitset([inchikeys_c[ik] for ik in inchikeys])
    
    # Integers
    
    if cumulative:
        def fetch_similars(iks, integers, cut, prev_cut, idx, all_idxs):
            I = intbitset([int(i) + idx + 1 for i in np.where(integers <= cut)[0]]) & all_idxs
            return intbitset([iks[i] for i in list(I)])
    else:
        def fetch_similars(iks, integers, cut, prev_cut, idx, all_idxs):
            I = intbitset([int(i) + idx + 1 for i in np.where(np.logical_and(integers <= cut, integers > prev_cut))[0]]) & all_idxs
            return intbitset([iks[i] for i in list(I)])
    
    def count_all(integers, idx, all_idxs):
        return len(intbitset([int(i) + idx + 1 for i in xrange(len(integers))]) & all_idxs)
    
    N = 0
    
    cutpairs = len(checkerconfig.cuts_e)*len(checkerconfig.cuts_c)
    pairs_e  = np.zeros(cutpairs, dtype = np.int)
    pairs_c  = np.zeros(cutpairs, dtype = np.int)
    pairs_ec = np.zeros(cutpairs, dtype = np.int)
    
    idxs = sorted(random.sample([i for i in xrange(len(inchikeys))], np.min([len(inchikeys), B])))
    
    for idx in idxs:
        inchikey = inchikeys[idx]
        inchikey_idx_e = bisect.bisect_left(iks_e, inchikey) # Be careful! this means that iks is sorted!
        inchikey_idx_c = bisect.bisect_left(iks_c, inchikey) # These idxs are used to only explore one triangle of the similarity matrix.
        PATH   = inchikey2webrepo_no_mkdir(inchikeys_str[idx]) + "/sig.h5"
        with h5py.File(PATH, "r") as hf:
            integers_e = hf["%s_obs" % coordinate_e][inchikey_idx_e + 1:]
            integers_c = hf["%s_obs" % coordinate_c][inchikey_idx_c + 1:]
    
        if do_e:
            N += count_all(integers_e, inchikey_idx_e, all_idxs_e)
        else:
            N += count_all(integers_c, inchikey_idx_c, all_idxs_c)
    
        k = 0
        prev_cut_e = -1
        for cut_e in checkerconfig.cuts_e:
            prev_cut_c = -1
            for cut_c in checkerconfig.cuts_c:
                sim_e = fetch_similars(iks_e, integers_e, cut_e, prev_cut_e, inchikey_idx_e, all_idxs_e)        
                sim_c = fetch_similars(iks_c, integers_c, cut_c, prev_cut_c, inchikey_idx_c, all_idxs_c)
                pairs_e[k]  += len(sim_e)
                pairs_c[k]  += len(sim_c)
                pairs_ec[k] += len(sim_e & sim_c)
                k += 1
                prev_cut_c = cut_c
            prev_cut_e = cut_e
    
        if N > 1e10:
            break
    
    
    ### Contingency table
    #     e Y  e N
    # c Y  A    B
    # c N  C    D
    
    k = 0
    
    R = []
    
    for cut_e in checkerconfig.cuts_e:
    
        for cut_c in checkerconfig.cuts_c:
    
            A = pairs_ec[k]
            B = pairs_c[k] - A
            C = pairs_e[k] - A
            D = N - (A + B + C)
    
            # Laplace smoothing
    
            A += 1
            B += 1
            C += 1
            D += 1
    
            # Probabilities
    
            p_e = (A + C) / (A + B + C + D)
            p_c = (A + B) / (A + B + C + D)
    
            # Conditional probability
    
            p_e_c  = A / (A + B)
            p_e_nc = C / (C + D)
    
            # Likelihood ratio
    
            l_e_c = p_e_c / p_e_nc
    
            # Save results
    
            R += [(cut_e, cut_c, p_e, p_c, p_e_c, p_e_nc, l_e_c, A-1)]
    
            k += 1
    
    
    # Insert to database
    
    
    
    S = ["('%s', '%s', %d, %d, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %d)" % (coordinate_e, coordinate_c, r[0], r[1],
                                                                               f2s(r[2]), f2s(r[3]),
                                                                               f2s(r[4]), f2s(r[5]),
                                                                               f2s(r[6]), f2s(np.log(r[2])),
                                                                               f2s(np.log(r[3])), f2s(np.log(r[4])),
                                                                               f2s(np.log(r[5])), f2s(np.log(r[6])), r[7]) for r in R]
    
    Psql.query("DELETE FROM coordinate_conditionals WHERE coord_e = '%s' AND coord_c = '%s'" % (coordinate_e, coordinate_c), dbname)
    Psql.query("INSERT INTO coordinate_conditionals VALUES %s" % ",".join(S), dbname)
    
    print "Done!"
