
# Imports

import h5py
import numpy as np
import sys, os
import collections

# Functions


def read_l1000(connectivitydir,mini_sig_info_file,lincs_molrepo):
    
    with open(lincs_molrepo, "r") as f:
        pertid_inchikey = {}
        inchikey_inchi = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            pertid_inchikey[l[0]] = l[2]
            inchikey_inchi[l[2]] = l[3]   

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

    PATH = connectivitydir
    for r in os.listdir(PATH):
        if ".h5" not in r: continue
        sig_id = r.split(".h5")[0]
        pert_id = siginfo[sig_id]
        if pert_id in pertid_inchikey:
            ik = pertid_inchikey[pert_id]
            inchikey_sigid[ik] += [sig_id]

    return inchikey_sigid,inchikey_inchi,siginfo

def get_summary(v):
        Qhi = np.percentile(v, 66)
        Qlo = np.percentile(v, 33)
        if np.abs(Qhi) > np.abs(Qlo):
            return Qhi
        else:
            return Qlo
        

# Main

if __name__ == '__main__':
    
    ik = sys.argv[1]
    mini_sig_info_file = sys.argv[2]
    PATH = sys.argv[3]
    ik_matrices = sys.argv[4]
    lincs_molrepo = sys.argv[5]

    inchikey_sigid,inchikey_inchi,siginfo = read_l1000(PATH,mini_sig_info_file,lincs_molrepo)
    
    with open("%s/signatures.tsv" % PATH, "r") as f:
        signatures = [l.rstrip("\n") for l in f]

        
    cols   = sorted(set(siginfo[s] for s in signatures))
    cols_d = dict((cols[i], i) for i in xrange(len(cols)))
    
    #parse_results(key,inchikey_sigid,siginfo,signatures,ik_matrices)
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
    
    
