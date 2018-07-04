'''

Morphology from LINCS.

I haven't tested it yet.

UNFINISHED!

'''

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
sys.path.append(os.path.join(sys.path[0], "../../mlutils/"))
from gaussian_scale_impute import scaleimpute
import Psql
import numpy as np
import collections
import csv
import random
import matplotlib.pyplot as plt
from cmapPy.pandasGEXpress import parse

# Variables

db = "XXX" # Psql.mosaic

morphlincs_molrepo = "XXX" # morphlincs.tsv
lds_1195 = "XXX" # data/LDS-1195/cdrp.img.profiles_n30440x812.gctx

# Functions

def parse_morphology():

    with open(morphlincs_molrepo, "r") as f:
        pertid_inchikey = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            pertid_inchikey[l[0]] = l[2]

    pd = parse(lds_1195)
    rownames = []
    for c in list(pd.col_metadata_df["pert_id"]):
        pertid = c.split("-")
        pertid = "%s-%s" % (pertid[0], pertid[1])
        rownames += [pertid]
    X = np.array(pd.data_df).T
    X = scaleimpute(X)
    rownames = np.array(rownames)

    sigs = collections.defaultdict(list)
    for i in xrange(len(rownames)):
        if rownames[i] not in pertid_inchikey: continue
        ik = pertid_inchikey[rownames[i]]
        sigs[ik] += [X[i]]

    return sigs


def find_strongest_signature(v):
    if len(v) == 1: return v[0]
    my_i = 0
    Sum = 0
    for i in xrange(len(v)):
        x = np.sum(np.abs(v[i]))
        if x > Sum:
            my_i = i
            Sum = x
    return v[my_i]

def filter_data(sigs):

    A = np.array([np.sum(np.abs(x)) for x in Xr])
    D = np.array([np.sum(np.abs(x)) for x in Xd])
    T = np.array([np.sum(np.abs(x)) for x in X])
    cutoff = int(0.5*np.percentile(A, 95) + 0.5*np.percentile(D, 95))

    rownames_f = np.array(rownames)[T > cutoff]
    X_f = X[T > cutoff]


    sigs = collections.defaultdict(list)
    for i in tqdm(xrange(len(rownames_f))):
        if rownames_f[i] not in pertid_inchikey: continue
        ik = pertid_inchikey[rownames_f[i]]
        sigs[ik] += [X_f[i]]
        
    print len(sigs)

    sigs = dict((k, find_strongest_signature(v)) for k,v in sigs.iteritems())

    inchikey_raw = dict((k, ",".join(["%.3f" % x for x in v])) for k,v in sigs.iteritems())

    Psql.insert_raw("morphology", inchikey_raw)
