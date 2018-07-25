#!/miniconda/bin/python


# Imports

import sys, os
import collections
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
from gaussian_scale_impute import scaleimpute
import numpy as np
import collections
import csv
import random
import matplotlib.pyplot as plt
from cmapPy.pandasGEXpress import parse

import checkerconfig


# Variables

dbname = "XXX" # Psql.mosaic

morphlincs_molrepo = "XXX" # morphlincs.tsv
lds_1195 = "LDS-1195/Data/cdrp.img.profiles_n30440x812.gctx"

perc = 95
Br   = 5
B    = 100000


# Functions


def parse_morphology(lds_file):

    with open(morphlincs_molrepo, "r") as f:
        pertid_inchikey = {}
        inchikey_inchi = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            pertid_inchikey[l[0]] = l[2]
            inchikey_inchi[l[2]] = l[3]

    pd = parse(lds_file)
    rownames = []
    for c in list(pd.col_metadata_df["pert_id"]):
        pertid = c.split("-")
        pertid = "%s-%s" % (pertid[0], pertid[1])
        rownames += [pertid]
    X = np.array(pd.data_df).T
    X = scaleimpute(X)
    rownames = np.array(rownames)

    return X, rownames, pertid_inchikey,inchikey_inchi


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


def colshuffled_matrix(X):
    Xr = np.array(X)
    for j in tqdm(xrange(Xr.shape[1])):
        shuffled = sorted(Xr[:,j], key=lambda k: random.random())
        Xr[:,j] = shuffled
    return Xr


def filter_data(X, rownames, pertid_inchikey):

    # Xd
    Xd = np.random.sample(size = (B, X.shape[1]))

    # Xr
    Arrs = []
    for _ in xrange(Br):
        Arrs += [colshuffled_matrix(X)]
    Xr = np.vstack(tuple(Arrs))

    # Cuts
    A = np.array([np.sum(np.abs(x)) for x in Xr])
    D = np.array([np.sum(np.abs(x)) for x in Xd])
    T = np.array([np.sum(np.abs(x)) for x in X])
    cutoff = int(0.5*np.percentile(A, perc) + 0.5*np.percentile(D, perc))

    rownames_f = np.array(rownames)[T > cutoff]
    X_f = X[T > cutoff]

    # To signatures
    sigs = collections.defaultdict(list)
    for i in tqdm(xrange(len(rownames_f))):
        if rownames_f[i] not in pertid_inchikey: continue
        ik = pertid_inchikey[rownames_f[i]]
        sigs[ik] += [X_f[i]]

    sigs = dict((k, find_strongest_signature(v)) for k,v in sigs.iteritems())

    return sigs


def insert_to_database(sigs,inchikey_inchi):

    inchikey_raw = dict((k, ",".join(["%.3f" % x for x in v])) for k,v in sigs.iteritems())
    
    todos = Psql.insert_structures(inchikey_inchi, dbname)
    for ik in todos:
        draw(ik,inchikey_inchi[ik])

    Psql.insert_raw("morphology", inchikey_raw,dbname)



# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    global morphlincs_molrepo
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    morphlincs_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"morphlincs.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    log.info(  "Parsing morphology data...")

    X, rownames, pertid_inchikey,inchikey_inchi = parse_morphology(os.path.join(downloadsdir,lds_1195))

    log.info(  "Filtering...")

    sigs = filter_data(X, rownames, pertid_inchikey)

    log.info(  "Inserting to database...")

    insert_to_database(sigs,inchikey_inchi)    


if __name__ == '__main__':
    main()