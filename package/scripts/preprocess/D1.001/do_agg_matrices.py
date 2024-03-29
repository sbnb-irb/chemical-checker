# Imports

import h5py
import numpy as np
import sys
import os
import collections
import pickle

from chemicalchecker.database import Molrepo
from chemicalchecker.core.signature_data import DataSignature
# Functions


def read_l1000(connectivitydir, mini_sig_info_file):
    
######## use when new mapping on molrepo ########

    molrepos = Molrepo.get_by_molrepo_name("lincs")
    pertid_inchikey = {}
    inchikey_inchi = {}
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        pertid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

######## In the meantime, we can map passing as argument the mapping file ########
    # pertid_inchikey = {}
    # inchikey_inchi = {}

    # with open(LINCS_2020_cp_info, "r") as f:
    #     f.readline()
    #     for l in f:
    #         l = l.rstrip("\n").split("\t")
    #         if l[-1] == '':
    #             continue

    #         pertid_inchikey[l[1]] =l[-1]  # pert_id ->inchikey 
    #         inchikey_inchi[l[-1]] = l[5]  # inchikey --> smile

    # Read signature data

    touchstones = set()
    siginfo = {}
    with open(mini_sig_info_file, "r") as f: # Ns REMINDER mini_sig_info_file contains:
                                             # sig_id, pert_id, pert_type, cell_id, istouchstone)
        for l in f:
            l = l.rstrip("\n").split("\t")
            if int(l[4]) == 1:               # select pert_id from Touchstone dataset
                touchstones.update([l[1]])
            siginfo[l[0]] = l[1]             # siginfo is {sig_id: pert_id, ..}

    inchikey_sigid = collections.defaultdict(list)  # creates an empty list if the key doesn't exists

    PATH = connectivitydir                          # List all individual h5 connectivity matrices
    for r in os.listdir(PATH):                       
        if ".h5" not in r:
            continue
        sig_id = r.split(".h5")[0]                 # sign id = h5 file name
        pert_id = siginfo[sig_id]                  # recovers pert_id of this signature 
        if pert_id in pertid_inchikey:             # if this pert_id happens to have an inchikey
            ik = pertid_inchikey[pert_id]
            inchikey_sigid[ik] += [sig_id]         # add it to the defaultdict of pert inchikeys {inchickey : [sigid]}

    return inchikey_sigid, siginfo                 # {inchickey : [sigid]}, {sig_id: pert_id}


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
    Qhi = np.percentile(v, 66)   # value before which 2/3 of the data are contained
    Qlo = np.percentile(v, 33)   # value before which 1/3 of the data are contained


    if np.abs(Qhi) > np.abs(Qlo):
        return Qhi
    else:
        # If both scores are negative then the 33-percentile is returned which is in fact the 66-percentile
        return Qlo


# Main

if __name__ == '__main__':

    task_id = sys.argv[1]
    filename = sys.argv[2]
    mini_sig_info_file = sys.argv[3]
    #LINCS_2020_cp_info = sys.argv[4]
    connectivitydir = sys.argv[4]
    agg_matrices = sys.argv[5]
    method = sys.argv[6]

    inputs = pickle.load(open(filename, 'rb'))
    iks = inputs[task_id]    # array of U27 str inchickeys

    # NS: example of input (task id is 998 here)
     # '998': array(['RWDQACDHOIJVRL-OXXUSNJHSA-N', 'WZQJIBXYFKEECF-WCZJQEMASA-N',
     #    'WOLKNTYXBIJULP-PJIZGREPSA-N', 'JEEWDARUSPNTGW-ZGQJUUMOSA-N',
     #    'BFDAKIJRIQBPOO-BPHNFWMXSA-N', 'LMYACBAJSFEPOG-SQQWPVGBSA-N',
     #    'RWDQACDHOIJVRL-WCZJQEMASA-N', 'KPHBLEYIQPKXAE-NTSGDEEZSA-N',
     #    'GBDHSNMZLIRBLK-INVAMZEASA-N', 'YGSIRKPQRSDRAD-UVMVBKAHSA-N'],
     #   dtype='<U27'),

    if method == "fit":

        inchikey_sigid, siginfo = read_l1000(connectivitydir, mini_sig_info_file)  # {inchickey : [sigid]}, {sig_id: pert_id}

    else:

        inchikey_sigid, siginfo = read_l1000_predict(connectivitydir, mini_sig_info_file)

    with open("%s/signatures.tsv" % connectivitydir, "r") as f:   # contains a list of sign ids from Touchstone
        signatures = [l.rstrip("\n") for l in f]                  # list of sign_id 

    cols = sorted(set(siginfo[s] for s in signatures))         # list of sorted unique pert_id corresponding to entries in signatures.tsv
    cols_d = dict((cols[i], i) for i in range(len(cols)))         # {pert_id:index}

    for ik in iks:                                           # going through the chunk of 10 iks(perturbagens) from input file (see above)
        v = inchikey_sigid[ik]                                    # [several sigid]
        neses = collections.defaultdict(list)                     # creates an empty list if the key doesn't exists

        for sigid in v:                                           # For every sig_id corresponding TO THIS PERTURBAGEN (ik)
            filename = sigid
            if method == "predict":
                filename = sigid + "---" + ik
            with h5py.File("%s/%s.h5" % (connectivitydir, filename), "r") as hf:  # recover the corresponding connectivity h5 file
                nes = hf["nes"][:]                                                # get the normalized connectivity scores for this sign
            for i in range(len(signatures)):                 # For all Touchstone signatures (many sign can refer to the same perturbagen)
                neses[(sigid, siginfo[signatures[i]])] += [nes[i]]   # create neses dict as {(current sigid, pert_id):[norm_conn_scores of all signatures refering to pert_id]}
        
        # Here neses correspond to all the signatures refering to our perturbagen's ik
        neses = dict((x, get_summary(y)) for x, y in neses.items())    
        # convert it into {(sigid, pert_id): 66-percentile-nes} 
        # (current sigid, pert_id)--> 'Median' connectivity score of all signatures referring to pert_id with our current signature  

        rows = sorted(set([k[0] for k in neses.keys()]))   # neses keys are (sigid, pert_id), take signid and sort-> sorted list of sigids
        rows_d = dict((rows[i], i) for i in range(len(rows)))     # {sigid:i}
        X = np.zeros((len(rows), len(cols))).astype(int)     #  all sigid for this pertb x all sorted pert_id in signatures.tsv

        for x, y in neses.items():                                # for all  (sigid, pert_id): 66-percentile-nes
            i = rows_d[x[0]]                                      # rows_d[sigid] is the index of sorted sigid
            j = cols_d[x[1]]                                      # cols_d[pert_id] is the index of sorted pertid 
            X[i, j] = y                                           # X[signid_idx, pertid_idx] = 66-percentile-nes (normalized conn score)

        if len(rows) == 0:                                        # If there are no signatures for this perturbagen
            continue

        # Create a matrix containing the 'median' connectivity score between
        # rows: a couple of signatures (usually one) corresponding to the current inchikey
        # Each perturbagen from Touchstone (7841 for the 2020 update)
        with h5py.File("%s/%s.h5" % (agg_matrices, ik), "w") as hf:
            hf.create_dataset("X", data=X)
            # getting strings instead of bytes from the h5 file
            hf.create_dataset("rows", data=np.array(rows, DataSignature.string_dtype()))
