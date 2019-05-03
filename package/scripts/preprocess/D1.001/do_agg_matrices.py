
# Imports

import h5py
import numpy as np
import sys
import os
import collections
import pickle

from chemicalchecker.database import Molrepo

# Functions


def read_l1000(connectivitydir, mini_sig_info_file):

    molrepos = Molrepo.get_by_molrepo_name("lincs")
    pertid_inchikey = {}
    inchikey_inchi = {}
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        pertid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

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
        if ".h5" not in r:
            continue
        sig_id = r.split(".h5")[0]
        pert_id = siginfo[sig_id]
        if pert_id in pertid_inchikey:
            ik = pertid_inchikey[pert_id]
            inchikey_sigid[ik] += [sig_id]

    return inchikey_sigid, siginfo


def read_l1000_predict(connectivitydir, mini_sig_info_file):

    # Read signature data

    touchstones = set()
    siginfo = {}
    with open(mini_sig_info_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if int(l[4]) == 1:
                touchstones.update([l[1]])
            siginfo[l[0]] = l[1]

    pertid_sigid = collections.defaultdict(list)

    PATH = connectivitydir
    for r in os.listdir(PATH):
        if ".h5" not in r:
            continue
        sig_id = r.split(".h5")[0]
        sig, pert_id = sig_id.split("---")
        pertid_sigid[pert_id] += [sig]

    return pertid_sigid, siginfo


def get_summary(v):
    Qhi = np.percentile(v, 66)
    Qlo = np.percentile(v, 33)
    if np.abs(Qhi) > np.abs(Qlo):
        return Qhi
    else:
        return Qlo


# Main

if __name__ == '__main__':

    task_id = sys.argv[1]
    filename = sys.argv[2]
    mini_sig_info_file = sys.argv[3]
    connectivitydir = sys.argv[4]
    agg_matrices = sys.argv[5]
    method = sys.argv[6]

    inputs = pickle.load(open(filename, 'rb'))
    iks = inputs[task_id]

    if method == "fit":

        inchikey_sigid, siginfo = read_l1000(
            connectivitydir, mini_sig_info_file)

    else:

        inchikey_sigid, siginfo = read_l1000_predict(
            connectivitydir, mini_sig_info_file)

    with open("%s/signatures.tsv" % connectivitydir, "r") as f:
        signatures = [l.rstrip("\n") for l in f]

    cols = sorted(set(siginfo[s] for s in signatures))
    cols_d = dict((cols[i], i) for i in range(len(cols)))

    for ik in iks:
        v = inchikey_sigid[ik]
        neses = collections.defaultdict(list)
        for sigid in v:
            filename = sigid
            if method == "predict":
                filename = sigid + "---" + ik
            with h5py.File("%s/%s.h5" % (connectivitydir, filename), "r") as hf:
                nes = hf["nes"][:]
            for i in xrange(len(signatures)):
                neses[(sigid, siginfo[signatures[i]])] += [nes[i]]
        neses = dict((x, get_summary(y)) for x, y in neses.items())
        rows = sorted(set([k[0] for k in neses.keys()]))
        rows_d = dict((rows[i], i) for i in xrange(len(rows)))
        X = np.zeros((len(rows), len(cols))).astype(np.int16)
        for x, y in neses.items():
            i = rows_d[x[0]]
            j = cols_d[x[1]]
            X[i, j] = y
        with h5py.File("%s/%s.h5" % (agg_matrices, ik), "w") as hf:
            hf.create_dataset("X", data=X)
            hf.create_dataset("rows", data=rows)
