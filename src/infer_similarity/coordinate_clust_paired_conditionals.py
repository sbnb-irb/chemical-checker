#!/miniconda/bin/python


# Fill the coordinate_clust_conditionals table.

# Imports

from __future__ import division
import sys, os
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import Psql
import checkerconfig
from checkerUtils import logSystem,log_data,coordinate2mosaic,inchikey2webrepo_no_mkdir
import numpy as np
import h5py
import collections
import bisect
from scipy.spatial.distance import cdist
import random
random.seed(42)

# Variables

def f2s(x):
    if np.isnan(x): return 'NULL'
    return x

def log(x):
    if np.isnan(x): return np.nan
    if x == 0: return np.nan
    return np.log(x)



B = 1000 # Number of samples
R = 5    # Repetitions

if __name__ == '__main__':


    coord_e, coord_c,dbname = sys.argv[1].split('---')
    
    path_e = coordinate2mosaic(coord_e)
    path_c = coordinate2mosaic(coord_c)
    
    with h5py.File(coordinate2mosaic("A1") + "models/bg_euclideans.h5", "r") as hf:
        pvals = hf["pvalue"][:]
    
    pvals = [pvals[c] for c in checkerconfig.cuts]
    
    # Load cluster annotations
    
    with h5py.File(path_e + "/clust.h5", "r") as hf:
        clust_e = hf["labels"][:]
        iks_e   = hf["keys"][:]
        
    with h5py.File(path_c + "/clust.h5", "r") as hf:
        clust_c = hf["labels"][:]
        iks_c   = hf["keys"][:]
        V       = hf["V_pqcode"][:]
    
    # Get the PQ-code of coord_c
    
    all_iks = set(iks_c).intersection(iks_e) # All inchikeys
    iks = iks_c[[x in all_iks for x in iks_c]]
    V   = V[[x in all_iks for x in iks_c]]
    
    # Get, for each cluster, indices of the inchikeys in the V matrix
    
    clust_e_idx = collections.defaultdict(set)
    for ik, clu in zip(iks_e, clust_e):
        if ik not in all_iks: continue
        clust_e_idx[clu].update([bisect.bisect_left(iks, ik)])
    clust_e_idx = dict((k, sorted(v)) for k,v in clust_e_idx.iteritems())
    
    clust_c_idx = collections.defaultdict(set)
    for ik, clu in zip(iks_c, clust_c):
        if ik not in all_iks: continue
        clust_c_idx[clu].update([bisect.bisect_left(iks, ik)])
    clust_c_idx = dict((k, sorted(v)) for k,v in clust_c_idx.iteritems())
    
    c_e = sorted(set(clust_e))
    c_c = sorted(set(clust_c))
     
    # Clean a bit
    
    del all_iks
    del iks_e
    del iks_c
    
    # Load PQ model, and background symmetric distances (euclidean approximations).
    
    with h5py.File(path_e + "/models/clustencoder.h5", "r") as hf:
        A = hf["A"][:]
    
    with h5py.File(path_c + "/models/bg_pq_euclideans.h5", "r") as hf:
        integers = hf["integer"][:]
        dists    = hf["distance"][:]
    
    # Distances of the chosen cutpoints (in thq PQ space)
    
    cuts_d = dists[checkerconfig.cuts]
        
    def symmetric_distance(x,y):
        d = 0
        for m in xrange(A.shape[0]):
            d += A[m][x[m],y[m]]
        return d
    
    # Do the macro contingency
    
    # Number of samples for the computation
    
    n_c = int(np.sqrt(B))
    n_e = n_c
    
    macroC = np.zeros((R, len(c_c), len(c_e), len(checkerconfig.cuts)))
    for r in xrange(R):
        for i, c in enumerate(c_c):
            if c not in clust_c_idx: continue
            idx_c = clust_c_idx[c]
            N_c = len(idx_c)
            idx_c = random.sample(idx_c, np.min([n_c, N_c]))
            for j, e in enumerate(c_e):
                if e not in clust_e_idx: continue
                idx_e = clust_e_idx[e]
                N_e = len(idx_e)
                idx_e = random.sample(idx_e, np.min([n_e, N_e]))
                d = cdist(V[idx_c], V[idx_e], metric = symmetric_distance).ravel()
                Ratio = (N_c * N_e) / len(d) # This ratio is done to account for the subsampling
                for k, cut_d in enumerate(cuts_d):
                    s = np.sum(d <= cut_d) * Ratio
                    macroC[r,i,j,k] = s
    
    # Do the average
    macroC = np.round(np.sum(macroC, axis = 0) / R, 0).astype(np.int)
    
            
    # Do all the contingencies
    
    ### Contingency table
    #     e Y  e N
    # c Y  A    B
    # c N  C    D
    
    # C = integers
    # E = clusters
    
    N = np.sum(macroC)
    
    R = []
    
    for i, c in enumerate(c_c):
        
        for j, e in enumerate(c_e):
            
            for k, integer in enumerate(checkerconfig.cuts):
                
                # Contingency table
                
                A = macroC[i,j,k]
                B = np.sum(macroC[:,:,k]) - A
                C = np.sum(macroC[i,j,:]) - A
                D = N - (A + B + C)
                
                # Probabilities
    
                p_e = (A + C) / (A + B + C + D)
                p_c = (A + B) / (A + B + C + D)
    
                # Conditional probability
    
                if (A+B) > 0:
                    p_e_c  = A / (A + B)
                else:
                    p_e_c = np.nan
                if (C + D) > 0:
                    p_e_nc = C / (C + D)
                else:
                    p_e_nc = np.nan
    
                # Likelihood ratio
    
                if np.isnan(p_e_c) or np.isnan(p_e_nc) or p_e_nc == 0:
                    l_e_c = np.nan
                else:
                    l_e_c = p_e_c / p_e_nc
               
     
                # Save results
                
                R += [[e, c, integer, pvals[k], p_e, p_c, p_e_c, p_e_nc, l_e_c, A, B, C, D]]
                
    
    # Insert to database
    
    
    
    
    S = ["('%s', '%s', %d, %d, %d, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %d, %d, %d, %d)" % (coord_e, coord_c, r[0], r[1], r[2], f2s(r[3]),
                                                                                                   f2s(r[4]), f2s(r[5]),
                                                                                                   f2s(r[6]), f2s(r[7]), f2s(r[8]),
                                                                                                   f2s(log(r[4])), f2s(log(r[5])),
                                                                                                   f2s(log(r[6])), f2s(log(r[7])),
                                                                                                   f2s(log(r[8])), r[9], r[10], r[11], r[12]) for r in R]
    
    Psql.query("DELETE FROM coordinate_clust_paired_conditionals WHERE coord_e = '%s' AND coord_c = '%s'" % (coord_e, coord_c), dbname)
    for s in Psql.chunker(S, 10000): Psql.query("INSERT INTO coordinate_clust_paired_conditionals VALUES %s" % ",".join(s), dbname)
    
    print "Done!"
