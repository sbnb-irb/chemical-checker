
# Imports

import h5py
import numpy as np
import sys
import os
import collections
import pickle

# Functions

core_cells = set(['A375', 'A549', 'HA1E', 'HCC515',
                  'HEPG2', 'MCF7', 'PC3', 'VCAP', 'HT29'])

only_touchstone = True


def es_score(idxs, ref_expr, min_idxs, p=1):
    if len(idxs) < min_idxs:
        return 0.
    N = len(ref_expr)
    Nh = len(idxs)
    norm = 1. / (N - Nh)
    miss = np.empty(N)
    miss[:] = norm
    miss[idxs] = 0.
    hit_nums = np.zeros(N)
    for i in idxs:
        hit_nums[i] = np.abs(ref_expr[i])**p
    hit = hit_nums / np.sum(hit_nums)
    P_hit = hit
    P_miss = miss
    P_hit = np.cumsum(hit)
    P_miss = np.cumsum(miss)
    ES = P_hit - P_miss
    return ES[np.argmax(np.abs(ES))]


def connectivity_score(up, dw, signature_file, signatures_dir, min_idxs):
    with h5py.File("%s/%s" % (signatures_dir, signature_file), "r") as hf:
        expr = hf["expr"][:]
        gene = hf["gene"][:]
        up_idxs, dw_idxs = [], []
        i = 0
        for g in gene:
            if g in up:
                up_idxs += [i]
            elif g in dw:
                dw_idxs += [i]
            i += 1
    es_up = es_score(up_idxs, expr, min_idxs)
    es_dw = es_score(dw_idxs, expr, min_idxs)
    if np.sign(es_up) == np.sign(es_dw):
        return 0
    else:
        return (es_up - es_dw) / 2


def signature_info(mini_sig_info_file):
    d = {}
    with open(mini_sig_info_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            d[l[0]] = (l[1], l[2], l[3])
    return d

# Main


def main(SIG, up, dw, mini_sig_info_file, signatures_dir, connectivity_dir, touch, min_idxs):

    sig_info = signature_info(mini_sig_info_file)

    CTp = collections.defaultdict(list)
    CTm = collections.defaultdict(list)
    R = []
    for f in os.listdir(signatures_dir):
        if ".h5" not in f:
            continue
        sig = f.split("/")[-1].split(".h5")[0]
        sinfo = sig_info[sig]

        if only_touchstone:
            if sinfo[0] not in touch or sinfo[2] not in core_cells:
                continue
        cs = connectivity_score(up, dw, f, signatures_dir, min_idxs)
        R += [(sig, cs)]
        if cs > 0:
            CTp[(sinfo[1], sinfo[2])] += [cs]
        elif cs < 0:
            CTm[(sinfo[1], sinfo[2])] += [cs]
        else:
            continue
    CTp = dict((k, np.median(v)) for k, v in CTp.items())
    CTm = dict((k, np.median(v)) for k, v in CTm.items())
    S = []
    for r in R:
        cs = r[1]
        sig = r[0]
        sinfo = sig_info[sig]
        if cs > 0:
            mu = CTp[(sinfo[1], sinfo[2])]
            ncs = cs / mu
        elif cs < 0:
            mu = -CTm[(sinfo[1], sinfo[2])]
            ncs = cs / mu
        else:
            ncs = 0.
        S += [(r[0], cs, ncs)]
    S = sorted(S, key=lambda tup: tup[0])
    if not os.path.exists("%s/signatures.tsv" % connectivity_dir):
        with open("%s/signatures.tsv" % connectivity_dir, "w") as f:
            for s in S:
                f.write("%s\n" % s[0])

    with h5py.File("%s/%s.h5" % (connectivity_dir, SIG), "w") as hf:
        es = np.array([s[1] * 1000 for s in S]).astype(np.int16)
        nes = np.array([s[2] * 1000 for s in S]).astype(np.int16)
        hf.create_dataset("es", data=es)
        hf.create_dataset("nes", data=nes)

if __name__ == '__main__':

    task_id = sys.argv[1]
    filename = sys.argv[2]
    mini_sig_info_file = sys.argv[3]
    signatures_dir = sys.argv[4]
    connectivity_dir = sys.argv[5]
    gene_info = sys.argv[6]
    min_idxs = int(sys.argv[7])

    inputs = pickle.load(open(filename, 'rb'))
    sigs = inputs[task_id]

    touch = set()
    with open(gene_info, "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            trt = l[2]
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                continue
            if l[3] == '0':
                continue
            touch.add(l[0])

    for k, v in sigs.items():

        if "up" in v:

            main(k, v["up"], v["down"], mini_sig_info_file,
                 signatures_dir, connectivity_dir, touch, min_idxs)
        else:
            with h5py.File(v["file"], "r") as hf:
                expr = hf["expr"][:]
                gene = hf["gene"][:]
                R = np.array(sorted(zip(gene, expr), key=lambda tup: -
                                    tup[1]), dtype=np.dtype([('gene', '|S300'), ('expr', np.float)]))
                up = R[:250]
                dw = R[-250:]
                up = set(up['gene'][up['expr'] > 2])
                dw = set(dw['gene'][dw['expr'] < -2])
                main(k, up, dw, mini_sig_info_file,
                     signatures_dir, connectivity_dir, touch, min_idxs)
