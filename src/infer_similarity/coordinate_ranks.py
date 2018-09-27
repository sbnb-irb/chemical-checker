#!/miniconda/bin/python


# Ranks metrics.

# Imports

from __future__ import division
import sys, os
import argparse
import h5py
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import Psql
import checkerconfig
from checkerUtils import logSystem,log_data,coordinate2mosaic,inchikey2webrepo_no_mkdir
import numpy as np
import math
from intbitset import intbitset
import bisect
import random
from rbo import rbo
random.seed(42)

# Variables
if __name__ == '__main__':


    coordinate_a, coordinate_b,dbname = sorted(sys.argv[1].split("---"))
    
    B = 1000
    obj_pvalue = 0.05 # When adjusting patience, look up to this p-value.
    prop_weight = 0.1 # Proportion of weight relative to the first rank.
    
    print coordinate_a, coordinate_b
    
    # Get molecules of each coordinate
    
    with h5py.File(coordinate2mosaic(coordinate_a)+ "sig.h5", "r") as hf:
        iks_a = hf['keys'][:]
    
    with h5py.File(coordinate2mosaic(coordinate_b)+ "sig.h5", "r") as hf:
        iks_b = hf['keys'][:]
    
    # Simply get the integers (typically 0 to 103)
    
    with h5py.File(coordinate2mosaic(coordinate_a) + "models/bg_euclideans.h5", "r") as hf:
        integers = hf["integer"][:]
        pvalue   = hf["pvalue"][:]
        obj_idx  = bisect.bisect_left(pvalue, obj_pvalue)
    
    # All iks
    
    iks = sorted(set(iks_a).intersection(iks_b))
    idxs_a = [bisect.bisect_left(iks_a, ik) for ik in iks]
    idxs_b = [bisect.bisect_left(iks_b, ik) for ik in iks]
    
    # Calculate RBO
    
    B = np.min([B, len(iks)])
    
    RBOs = []
    
    for idx in random.sample([i for i in xrange(len(iks))], B):
        
        rnk_a = [set() for _ in xrange(len(integers))]
        rnk_b = [set() for _ in xrange(len(integers))]
        PATH = inchikey2webrepo_no_mkdir(iks[idx]) + "/sig.h5"
        with h5py.File(PATH, "r") as hf:
            sim_a = hf["%s_obs" % coordinate_a][:][idxs_a[:idx]+idxs_a[(idx+1):]]
            sim_b = hf["%s_obs" % coordinate_b][:][idxs_b[:idx]+idxs_b[(idx+1):]]
        for i, j in enumerate(sim_a): rnk_a[j].update([i])
        for i, j in enumerate(sim_b): rnk_b[j].update([i])
    
        my_obj_idx = obj_idx
        for ab in zip(rnk_a,rnk_b):
            if not ab[0] or not ab[1]:
                ra = rnk_a[0]
                rnk_a = rnk_a[1:]
                rnk_a[0].update(ra)
                rb = rnk_b[0]
                rnk_b = rnk_b[1:]
                rnk_b[0].update(rb)
                my_obj_idx -= 1
            else:
                break
    
    #    random.shuffle(rnk_b)
    
        if my_obj_idx < 0:
            continue
        else:
            found = False
            step = 0.01
            for p in np.arange(1-step, 0, -step):
                w = (1-p)*(p**my_obj_idx) / (1-p)
                if w < prop_weight:
                    found = True
                    break
            if not found: continue
    
        RBOs += [rbo(rnk_a, rnk_b, p)]
    
    if RBOs:
        perc_05 = np.percentile(RBOs, 5 )
        perc_25 = np.percentile(RBOs, 25)
        perc_50 = np.percentile(RBOs, 50)
        perc_75 = np.percentile(RBOs, 75)
        perc_95 = np.percentile(RBOs, 95)
        mean = np.mean(RBOs)
    else:
        perc_05 = 0.
        perc_25 = 0.
        perc_50 = 0.
        perc_75 = 0.
        perc_95 = 0.
        mean    = 0.
    
    Psql.query("DELETE FROM coordinate_ranks WHERE coord_a = '%s' and coord_b = '%s'" % (coordinate_a, coordinate_b), dbname)
    Psql.query("INSERT INTO coordinate_ranks VALUES ('%s', '%s', %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f)" % (coordinate_a, coordinate_b, len(RBOs), mean, perc_05, perc_25, perc_50, perc_75, perc_95), dbname)
    
