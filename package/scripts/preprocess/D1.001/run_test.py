# NS: TEST VERSION for the aggregation matrices
import random

import os
import sys
import argparse
import h5py
import tempfile
import numpy as np
from scipy.stats import rankdata
import collections
import shutil
from cmapPy.pandasGEXpress import parse
import glob
import logging

from chemicalchecker.util import logged, Config, profile
from chemicalchecker.util.hpc import HPC
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.util.performance import gaussianize as g
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.core.signature_data import DataSignature

# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:] #NS D1.001
features_file = "features.h5"


def parse_level(mini_sig_info_file, map_files, signaturesdir):

    readyfile = "sigs.ready"
    if os.path.exists(os.path.join(signaturesdir, readyfile)):
        return

    GSE92742_Broad_LINCS_pert_info = os.path.join(
        map_files["GSE92742_Broad_LINCS_pert_info"], "GSE92742_Broad_LINCS_pert_info.txt")
    touchstone = set()
    with open(GSE92742_Broad_LINCS_pert_info, "r") as f:
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            trt = l[2]
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                continue
            if l[3] == '0':
                continue
            touchstone.add(l[0])

    # Gene symbols (same for LINCS I and II)
    GSE92742_Broad_LINCS_gene_info = os.path.join(
        map_files["GSE92742_Broad_LINCS_gene_info"], "GSE92742_Broad_LINCS_gene_info.txt")
    genes = {}
    with open(GSE92742_Broad_LINCS_gene_info, "r") as f:
        f.readline()
        for l in f:
            l = l.split("\t")
            genes[l[0]] = l[1]

    sig_info_ii = {}

    for file_name in glob.glob(os.path.join(map_files["GSE70138_Broad_LINCS_sig_info"], "*.txt")):
        with open(file_name, "r") as f:
            f.readline()
            for l in f:
                l = l.rstrip("\n").split("\t")
                if l[1] in touchstone:
                    v = 1
                else:
                    v = 0
                sig_info_ii[l[0]] = (l[1], l[3], l[4], v, 2)

    # Signature metrics

    sigs = collections.defaultdict(list)
    for file_name in glob.glob(os.path.join(map_files["GSE70138_Broad_LINCS_sig_metrics"], "*.txt")):
        with open(file_name, "r") as f:
            f.readline()
            for l in f:
                l = l.rstrip("\n").split("\t")[1:]
                if float(l[1]) < 0.2:
                    continue
                trt = l[5]
                sig_id = l[0]
                if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                    continue
                if sig_id not in sig_info_ii:
                    continue
                v = sig_info_ii[sig_id]
                tas = float(l[6])
                nsamp = int(l[-1])
                phase = 2
                sigs[(v[0], v[2])] += [(sig_id, trt, tas, nsamp, phase)]

    sig_info_i = {}
    GSE92742_Broad_LINCS_sig_info = os.path.join(
        map_files["GSE92742_Broad_LINCS_sig_info"], "GSE92742_Broad_LINCS_sig_info.txt")
    with open(GSE92742_Broad_LINCS_sig_info, "r") as f:
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[1] in touchstone:
                v = 1
            else:
                v = 0
            sig_info_i[l[0]] = (l[1], l[3], l[4], v, 1)

    GSE92742_Broad_LINCS_sig_metrics = os.path.join(
        map_files["GSE92742_Broad_LINCS_sig_metrics"], "GSE92742_Broad_LINCS_sig_metrics.txt")
    with open(GSE92742_Broad_LINCS_sig_metrics, "r") as f:
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if float(l[4]) < 0.2:
                continue
            trt = l[3]
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                continue
            sig_id = l[0]
            if sig_id not in sig_info_i:
                continue
            v = sig_info_i[sig_id]
            tas = float(l[8])
            nsamp = int(l[-1])
            phase = 1
            sigs[(v[0], v[2])] += [(sig_id, trt, tas, nsamp, phase)]

    def get_exemplar(v):
        s = [x for x in v if x[3] >= 2 and x[3] <= 6]
        if not s:
            s = v
        sel = None
        max_tas = 0.
        for x in s:
            if not sel:
                sel = (x[0], x[-1])
                max_tas = x[2]
            else:
                if x[2] > max_tas:
                    sel = (x[0], x[-1])
                    max_tas = x[2]
        return sel

    sigs = dict((k, get_exemplar(v)) for k, v in sigs.items())

    cids = []
    with open(mini_sig_info_file, "w") as f:
        for k, v in sigs.items():
            if v[1] == 1:
                x = sig_info_i[v[0]]
            else:
                x = sig_info_ii[v[0]]
            f.write("%s\t%s\t%s\t%s\t%d\t%d\n" %
                    (v[0], x[0], x[1], x[2], x[3], v[1]))
            cids += [(v[0], v[1])]

    gtcx_i = os.path.join(
        map_files["GSE92742_Broad_LINCS_Level5_COMPZ"], "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx")
    for file_name in glob.glob(os.path.join(map_files["GSE70138_Broad_LINCS_Level5_COMPZ"], "*.gctx")):
        gtcx_ii = file_name

    genes_i = [genes[r[0]] for r in parse.parse(
        gtcx_i, cid=[[x[0] for x in cids if x[1] == 1][0]]).data_df.iterrows()]
    genes_ii = [genes[r[0]] for r in parse.parse(gtcx_ii, cid=[
                                                 [x[0] for x in cids if x[1] == 2][0]]).data_df.iterrows()]  # Just to make sure.

    for cid in cids:
        if cid[1] == 1:
            expr = np.array(parse.parse(gtcx_i, cid=[cid[0]]).data_df).ravel()
            genes = genes_i
        elif cid[1] == 2:
            expr = np.array(parse.parse(gtcx_ii, cid=[cid[0]]).data_df).ravel()
            genes = genes_ii
        else:
            continue
        R = zip(genes, expr)
        R = sorted(R, key=lambda tup: -tup[1])
        with h5py.File(os.path.join(signaturesdir, "%s.h5" % cid[0]), "w") as hf:
            hf.create_dataset("expr", data=[float(r[1]) for r in R])
            hf.create_dataset("gene", data=DataSignature.h5_str([r[0] for r in R]))

    with open(os.path.join(signaturesdir, readyfile), "w") as f:
        f.write("")


def read_l1000(mini_sig_info_file, connectivitydir):

    inchikey_inchi = {}
    pertid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("lincs")  # return all molecules from the lincs repository

    # NS, for each lincs molecule, record inchikey inchi in mappings
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue

        pertid_inchikey[molrepo.src_id] = molrepo.inchikey  # mol_id ->inchikey 
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi    # mol_inchikey -> mol_inchi

    # Read signature data

    touchstones = set()
    siginfo = {}

    with open(mini_sig_info_file, "r") as f:  # read the minisign file
        for l in f:
            l = l.rstrip("\n").split("\t")    # keeps 
            if int(l[4]) == 1:
                touchstones.update([l[1]])    # put records like BRD-K25943794 in the set (second record of mini_sig_info_file = perturbagen id)
            siginfo[l[0]] = l[1]              # REP.A001_A375_24H:K17 : BRD-A29289453

    inchikey_sigid = collections.defaultdict(list)

    PATH = connectivitydir                  # NS signature0full_path/raw/models/connectivity_fit
    for r in os.listdir(PATH):              # ls without fullpath (only filenames)
        if ".h5" not in r:                  # select h5 files in the connectivity dir
            continue
        sig_id = r.split(".h5")[0]          # recovers the filename without extension
        pert_id = siginfo[sig_id]           # maps the name of the h5 file to the second record of mini_sig_info_file (perturbagen id)

        if pert_id in pertid_inchikey:
            ik = pertid_inchikey[pert_id]
            inchikey_sigid[ik] += [sig_id]  # inchikey -> [list of perturbagen ids]

    #returns the following mappings from L1000:
    # inchikey_sigid:   mol_inchikey -> mol_id (h5 filename without extension)
    # inchikey_inchi:   mol_inchikey -> mol_inchi
    # siginfo:          REP.A001_A375_24H:K17 -> mol_id

    return inchikey_sigid, inchikey_inchi, siginfo


def get_summary(v):
    Qhi = np.percentile(v, 66)
    Qlo = np.percentile(v, 33)
    if np.abs(Qhi) > np.abs(Qlo):
        return Qhi
    else:
        return Qlo


def do_consensus(ik_matrices, consensus):

    inchikeys = [ik.split(".h5")[0]
                 for ik in os.listdir(ik_matrices) if ik.endswith(".h5")]

    def consensus_signature(ik):
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "r") as hf:
            X = hf["X"][:]
        # It could be max, min...
        return [np.int16(get_summary(X[:, j])) for j in range(X.shape[1])]

    X = np.array([consensus_signature(ik) for ik in inchikeys])

    with h5py.File(consensus, "w") as hf:
        hf.create_dataset("inchikeys", data=np.array(inchikeys, dtype=h5py.special_dtype(vlen=str)))
        hf.create_dataset("X", data=X)

    return X, inchikeys


def process(X):

    def whiten(X):

        Xw = np.zeros(X.shape)

        for j in range(X.shape[1]):
            V = X[:, j]
            V = rankdata(V, "ordinal")
            gauss = g.Gaussianize(strategy="brute")
            gauss.fit(V)
            V = gauss.transform(V)
            Xw[:, j] = np.ravel(V)

        return Xw

    def cutoffs(X):
        return [np.percentile(X[:, j], 99) for j in range(X.shape[1])]

    cuts = cutoffs(X)

    Xcut = []
    for j in range(len(cuts)):
        c = cuts[j]
        v = np.zeros(X.shape[0])
        v[X[:, j] > c] = 1
        Xcut += [v]

    Xcut = np.array(Xcut).T

    return Xcut


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
#@profile
def main(args):

    args = Preprocess.get_parser().parse_args(args)

    # NS ex: fir D1.001, args will be a mapping:
    # -o : signature0full_path/raw/preprocess.h5    # --output_file
    # -mp: signature0full_path/raw/models           # --model_path
    # -m: 'fit'                                     # --method

    mini_sig_info_file = os.path.join(args.models_path, 'mini_sig_info.tsv')  # NS file with 83637 records from L1000?

    dataset = Dataset.get(dataset_code) #NS D1.001 dataset object built to queryour sql database

    map_files = {}  # NS: will store datasource names of D1.00X and path to the corresponding files

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.

    # NS: 13 datasources in dataset_had_datasource table for the 2020 update
    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path

    main._log.debug("Running preprocess fit method for dataset " + dataset_code + ". Saving output in " + args.output_file)

    signaturesdir = os.path.join(args.models_path, "signatures") #signature0full_path/raw/models/signatures
                                                                 # contains 83 639 h5 files and a sigs.ready file

    if os.path.exists(signaturesdir) is False:
        os.makedirs(signaturesdir)

    if args.method == 'fit':  # True

        mpath = args.models_path  # signature0full_path/raw/models/

        main._log.info("Parsing")
        parse_level(mini_sig_info_file, map_files, signaturesdir)  #--> creates all these h5 files by parsing datasource, just returns if sigs.ready is present

        # Creates subdirs in signature0full_path/raw/models/
        ik_matrices = os.path.join(mpath, 'ik_matrices_fit')

        if os.path.exists(ik_matrices) is False:
            os.makedirs(ik_matrices, 0o775)

        connectivitydir = os.path.join(mpath, 'connectivity_fit')

        if os.path.exists(connectivitydir) is False:
            os.makedirs(connectivitydir, 0o775)

        # signature0full_path/raw/models/consensus_fit.h5, not present after the crash
        consensus = os.path.join(mpath, "consensus_fit.h5")

        min_idxs = 10

        cp_sigs = set()

        # Parsing the record file
        with open(mini_sig_info_file) as f:
            for l in f:
                l = l.rstrip("\n").split("\t")
                if l[2] == "trt_cp":
                    # update the set cp_sigs with records such as 'REP.A001_A375_24H:E14'
                    cp_sigs.update([l[0]])


        sig_map = {}
        # Populate sig_map dict with i.e 'REP.A001_A375_24H:E14' : {"file": "signature0full_path/raw/models/signatures/REP.A001_A375_24H:E14.h5"}
        for SIG in cp_sigs:
            sig_map[SIG] = {"file": "%s/%s.h5" % (signaturesdir, SIG)}

    if args.method == 'predict':  # False here

        mpath = tempfile.mkdtemp(
            prefix='predict_', dir=args.models_path)

        ik_matrices = tempfile.mkdtemp(prefix='ik_matrices_', dir=mpath)

        connectivitydir = tempfile.mkdtemp(
            prefix='connectivity_', dir=mpath)

        consensus = os.path.join(mpath, "consensus_predict.h5")

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        min_idxs = 1

        sig_map = {}
        inchikey_inchi = {}

        inchikey_sigid = collections.defaultdict(list)

        with open(args.input_file) as f:
            for l in f:
                items = l.split('\t')
                if len(items) != 4:
                    raise Exception("Input data not in the right format. Expected 4 columns only " + str(len(items)))
                if items[1] == '':
                    ik = items[0]
                else:
                    ik = items[1]
                inchikey_sigid[ik] += [items[0]]
                sig_map[items[0] + "---" + ik] = {"up": items[2].split(","), "down": items[3].split(",")}

    WD = os.path.dirname(os.path.realpath(__file__))  # directory from which run.py is launched

    connectivity_script = WD + "/connectivity.py"     # scripts called by run.py in the same directory

    ikmatrices_script = WD + "/do_agg_matrices.py"

    readyfile = "conn.ready"

    config = Config()                                # reads os.environ["CC_CONFIG"]

    # CONNECTIVITY JOB 
    # Note, connectivitydir is signature0full_path/raw/models/connectivity_fit
    if not os.path.exists(os.path.join(connectivitydir, readyfile)):   # contains 65 151 h5 files and conn.ready so False here

        main._log.info("Getting signature files...")

        job_path = os.path.join(mpath, "job_conn")  # contains a script file job-CC_D1_conn.sh and an input file which is a pickle from sig_map
                                                    # launches the 6514 jobs of connectivity.py with the input file, mini_sig_info.tsv, connectivity_fit
                                                    # GSE92742_Broad_LINCS_pert_info.txt as arguments

        if os.path.isdir(job_path):
            shutil.rmtree(job_path)
        os.mkdir(job_path)

        GSE92742_Broad_LINCS_pert_info = os.path.join(
            map_files["GSE92742_Broad_LINCS_pert_info"], "GSE92742_Broad_LINCS_pert_info.txt")

        params = {}

        params["num_jobs"] = len(sig_map.keys()) / 10
        params["jobdir"] = job_path
        params["job_name"] = "CC_D1_conn"
        params["elements"] = sig_map
        # reminder: sigmap is a dict with the following entry type:
        #'REP.A001_A375_24H:E14' : {"file": "signature0full_path/raw/models/signatures/REP.A001_A375_24H:E14.h5"}
        params["memory"] = 10
        # job command
        singularity_image = config.PATH.SINGULARITY_IMAGE
        command = "MKL_NUM_THREADS=1 singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {} {}".format(
            singularity_image, connectivity_script, mini_sig_info_file, signaturesdir,
            connectivitydir, GSE92742_Broad_LINCS_pert_info, min_idxs)
        # submit jobs
        cluster = HPC.from_config(config)
        cluster.submitMultiJob(command, **params)

        if cluster.status() == 'error':
            main._log.error(
                "Connectivity job produced some errors. The preprocess script can't continue")
            sys.exit(1)

        if args.method == 'fit':
            with open(os.path.join(connectivitydir, readyfile), "w") as f:
                f.write("")

    readyfile = "agg_matrices.ready_test"

    # MATRIX AGGREGATION JOB
    if not os.path.exists(os.path.join(ik_matrices, readyfile)):

        if args.method == 'fit':
            main._log.info("Reading L1000")

            # NS Get molecules from Lincs
            inchikey_sigid, inchikey_inchi, siginfo = read_l1000(mini_sig_info_file, connectivitydir)
            # inchikey_sigid:   mol_inchikey -> mol_id (h5 filename without extension)
            # inchikey_inchi:   mol_inchikey -> mol_inchi
            # siginfo:          REP.A001_A375_24H:K17 -> mol_id

            #NS reducing the number of entries for test

            print("Choosing a random sample")
            random_keys= random.sample(list(inchikey_sigid.keys()), 10)
            inchikey_sigid = {k:v for (k,v) in inchikey_sigid.items() if k in random_keys}
            inchikey_inchi = {k:v for (k,v) in inchikey_inchi.items() if k in random_keys}

            print("inchikey_sigid-->\n", inchikey_sigid)
            print("inchikey_inchi-->\n", inchikey_inchi)
            print("siginfo-->\n", siginfo)
            if len(inchikey_sigid) == 0 or len(inchikey_inchi) ==0:
                print("Empty sample dict!")
                sys.exit(1)
            print("Done")

        main._log.info("Doing aggregation matrices")

        #/aloy/web_checker/package_cc/2020_01/full/D/D1/D1.001/sign0/raw/models/job_agg_matrices
        job_path = os.path.join(mpath, "job_agg_matrices")

        if os.path.isdir(job_path):
            shutil.rmtree(job_path)
        os.mkdir(job_path)

        params = {}

        params["num_jobs"] = len(inchikey_sigid.keys()) / 10
        params["jobdir"] = job_path
        params["job_name"] = "CC_D1_agg_mat"
        params["elements"] = list(inchikey_sigid.keys())  # dict_key view of mol_ids
        params["memory"] = 10
        cc_package = os.path.join(config.PATH.CC_REPO, 'package')

        # job command
        singularity_image = config.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {}".format(
            cc_package, os.environ['CC_CONFIG'], singularity_image, ikmatrices_script, mini_sig_info_file, connectivitydir, ik_matrices, args.method)

        # submit jobs
        cluster = HPC.from_config(config)
        cluster.submitMultiJob(command, **params)  #--> THE GUILTY LINE

        if cluster.status() == 'error':
            main._log.error(
                "Agg_matrices job produced some errors. The preprocess script can't continue")
            sys.exit(1)

        if args.method == 'fit':
            with open(os.path.join(ik_matrices, readyfile), "w") as f:
                f.write("")

    main._log.info("Doing consensus")
    X, inchikeys = do_consensus(ik_matrices, consensus)

    main._log.info("Process output")
    Xcut = process(X)

    main._log.info("Saving raws")
    inchikey_raw = {}
    for i in range(len(inchikeys)):
        ik = inchikeys[i]
        if np.sum(Xcut[i, :]) < 5:
            continue
        idxs = np.where(Xcut[i, :] == 1)[0]
        inchikey_raw[ik] = [(str(x), 1) for x in idxs]

    keys = []
    words = set()
    for k in sorted(inchikey_raw.keys()):
        keys.append(str(k))
        words.update([x[0] for x in inchikey_raw[k]])

    if features is not None:
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_raw[k]:
            raws[i][wordspos[word[0]]] += word[1]

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys, DataSignature.string_dtype()))
        hf.create_dataset("X", data=raws)
        hf.create_dataset("features", data=np.array(orderwords, DataSignature.string_dtype()))

    if args.method == "fit":
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=np.array(orderwords, h5py.special_dtype(vlen=str)))

        print("TEST PASSED OK! Quitting.")
        sys.exit(0)

    if args.method == 'predict':
        shutil.rmtree(mpath)
        # shutil.rmtree(connectivitydir)

if __name__ == '__main__':
    main(sys.argv[1:])
