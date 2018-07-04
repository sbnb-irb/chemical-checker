'''
# LINCS raw data

We follow recommendations by http://www.biorxiv.org/content/biorxiv/early/2017/05/10/136168.full.pdf

Data are pre-processed in `data/new/` using the `_connectivity.py` script.

The idea is to obtain an NCS score for each perturbation in the `Touchstone` dataset.

We compare everything against the Touchstone dataset.

'''

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
sys.path.append(os.path.join(sys.path[0], "../../mlutils/"))
from gaussian_scale_impute import scaleimpute
import Psql
import numpy as np
import collections
import subprocess
import h5py
import time
from multiprocessing import Pool

db = Psql.mosaic

# Variables

lincs_molrepo = "XXXX" # LINCS molrepo file
mini_sig_info_file = "XXXX" # data/new/mini_sig_info.tsv
connectivity_folder= "XXXX" # data/new/connectivity/
ik_matrices = "XXXX" # data/new/ik_matrices/
consensus   = "XXXX" # consensus.h5 - Let's talk about where to save it!

table = "transcript"

# Functions

def read_l1000():
    
    with open(lincs_molrepo, "r") as f:
        pertid_inchikey = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            pertid_inchikey[l[0]] = l[2]

    # Read signature data

    touchstones = set()
    siginfo = {}
    with open(mini_sig_info_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if int(l[4]) == 1:
                touchstones.update([l[1]])
            siginfo[l[0]] = l[1]
            
    inchikey_sigid = collections.defaultdict(list)

    PATH = connectivity_folder
    for r in os.listdir(PATH):
        if ".h5" not in r: continue
        sig_id = r.split(".h5")[0]
        pert_id = siginfo[sig_id]
        if pert_id in pertid_inchikey:
            ik = pertid_inchikey[pert_id]
            inchikey_sigid[ik] += [sig_id]

    return inchikey_sigid


def do_ik_matrices(inchikey_sigid):

    # Be careful!!! As it is now, it is a multiprocess.

    def get_summary(v):
        Qhi = np.percentile(v, 66)
        Qlo = np.percentile(v, 33)
        if np.abs(Qhi) > np.abs(Qlo):
            return Qhi
        else:
            return Qlo

    # New version, across CORE cell lines and TOUCHSTONE signatures.

    # This will take a while...

    with open("%s/signatures.tsv" % PATH, "r") as f:
        signatures = [l.rstrip("\n") for l in f]

    if not os.path.exists(ik_matrices): os.mkdir(ik_matrices)
        
    cols   = sorted(set(siginfo[s] for s in signatures))
    cols_d = dict((cols[i], i) for i in xrange(len(cols)))

    p = Pool()

    pbar = total = len(inchikey_sigid) / p._processes

    def parse_results(ik):
        v = inchikey_sigid[ik]
        neses = collections.defaultdict(list)
        for sigid in v:
            with h5py.File("%s/%s.h5" % (PATH, sigid), "r") as hf:
                nes = hf["nes"][:]
            for i in xrange(len(signatures)):
                neses[(sigid, siginfo[signatures[i]])] += [nes[i]]
        neses  = dict((x, get_summary(y)) for x,y in neses.iteritems())
        rows   = sorted(set([k[0] for k in neses.keys()]))
        rows_d = dict((rows[i], i) for i in xrange(len(rows)))
        X = np.zeros((len(rows), len(cols))).astype(np.int16)
        for x,y in neses.iteritems():
            i = rows_d[x[0]]
            j = cols_d[x[1]]
            X[i,j] = y
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "w") as hf:
            hf.create_dataset("X", data = X)
            hf.create_dataset("rows", data = rows)
        pbar.update(1)

    p.map(parse_results, inchikey_sigid.keys())


def do_consensus():

    inchikeys = [ik.split(".h5")[0] for ik in os.listdir(ik_matrices)]

    def consensus_signature(ik):
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "r") as hf:
            X = hf["X"][:]
        return [np.int16(get_summary(X[:,j])) for j in xrange(X.shape[1])] # It could be max, min...

    X = np.array([consensus_signature(ik) for ik in inchikeys])

    with h5py.File(consensus, "w") as hf:
        hf.create_dataset("inchikeys", data = inchikeys)
        hf.create_dataset("X", data = X)

    return X, inchikeys


def process():

    def whiten(X):
        
        Xw = np.zeros(X.shape)
        
        import gaussianize as g
        from scipy.stats import rankdata
        
        for j in xrange(X.shape[1]):
            V = X[:,j]
            V = rankdata(V, "ordinal")
            gauss = g.Gaussianize(strategy = "brute")
            gauss.fit(V)
            V = gauss.transform(V)
            Xw[:,j] = np.ravel(V)
        
        return Xw

    Xw = whiten(X)

    def cutoffs(X):
        return [np.percentile(X[:,j], 99) for j in xrange(X.shape[1])]
            
    cuts = cutoffs(X)

    Xcut = []
    for j in xrange(len(cuts)):
        c = cuts[j]
        v = np.zeros(X.shape[0])
        v[X[:,j] > c] = 1
        Xcut += [v]
        
    Xcut = np.array(Xcut).T

    return Xcut


def insert_to_database(Xcut, inchikeys):

    inchikey_raw = {}
    for i in xrange(len(inchikeys)):
        ik = inchikeys[i]
        if np.sum(Xcut[i,:]) < 5: continue
        idxs = np.where(Xcut[i,:] == 1)[0]
        inchikey_raw[ik] = ",".join(["%d(1)" % x for x in idxs])

    Psql.insert_raw(transcript, inchikey_raw)


# Main

def main():

    print "Reading L1000"
    inchikey_sigid = read_l1000()

    print "Doing ik_matrices"
    do_ik_matrices(inchikey_sigid)

    print "Doing consensus"
    X, inchikeys = do_consensus()

    print "Process output"
    Xcut = process()

    print "Insert to database"
    insert_to_database(Xcut)


if __name__ == '__main__':
    main()