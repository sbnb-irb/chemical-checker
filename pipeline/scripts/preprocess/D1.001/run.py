import os
import sys
import argparse
import h5py
import numpy as np
from scipy.stats import rankdata
import collections
import shutil
from cmapPy.pandasGEXpress import parse
import glob


from chemicalchecker.util import logged, Config, profile
from chemicalchecker.util import HPC
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.util import gaussianize as g
# Variables


def parse_level(mini_sig_info_file, map_files, signaturesdir):

    readyfile = "sigs.ready"
    if os.path.exists(os.path.join(signaturesdir, readyfile)):
        return

    GSE92742_Broad_LINCS_pert_info = os.path.join(
        map_files["GSE92742_Broad_LINCS_pert_info"], "GSE92742_Broad_LINCS_pert_info.txt")
    touchstone = set()
    with open(GSE92742_Broad_LINCS_pert_info, "r") as f:
        f.next()
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
        f.next()
        for l in f:
            l = l.split("\t")
            genes[l[0]] = l[1]

    sig_info_ii = {}

    for file_name in glob.glob(os.path.join(map_files["GSE70138_Broad_LINCS_sig_info"], "*.txt")):
        with open(file_name, "r") as f:
            f.next()
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
            f.next()
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
        f.next()
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
        f.next()
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

    sigs = dict((k, get_exemplar(v)) for k, v in sigs.iteritems())

    cids = []
    with open(mini_sig_info_file, "w") as f:
        for k, v in sigs.iteritems():
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
            hf.create_dataset("gene", data=[r[0] for r in R])

    with open(os.path.join(signaturesdir, readyfile), "w") as f:
        f.write("")


def read_l1000(mini_sig_info_file, connectivitydir):

    inchikey_inchi = {}
    pertid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("lincs")
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
        return [np.int16(get_summary(X[:, j])) for j in xrange(X.shape[1])]

    X = np.array([consensus_signature(ik) for ik in inchikeys])

    with h5py.File(consensus, "w") as hf:
        hf.create_dataset("inchikeys", data=inchikeys)
        hf.create_dataset("X", data=X)

    return X, inchikeys


def process(X):

    def whiten(X):

        Xw = np.zeros(X.shape)

        for j in xrange(X.shape[1]):
            V = X[:, j]
            V = rankdata(V, "ordinal")
            gauss = g.Gaussianize(strategy="brute")
            gauss.fit(V)
            V = gauss.transform(V)
            Xw[:, j] = np.ravel(V)

        return Xw

    def cutoffs(X):
        return [np.percentile(X[:, j], 99) for j in xrange(X.shape[1])]

    cuts = cutoffs(X)

    Xcut = []
    for j in xrange(len(cuts)):
        c = cuts[j]
        v = np.zeros(X.shape[0])
        v[X[:, j] > c] = 1
        Xcut += [v]

    Xcut = np.array(Xcut).T

    return Xcut


# Parse arguments


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_file', type=str,
                        required=False, default='.', help='Input file only for predict method')
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
    parser.add_argument('-m', '--method', type=str,
                        required=False, default='fit', help='Method: fit or predict')
    parser.add_argument('-mp', '--models_path', type=str,
                        required=False, default='', help='The models path')
    return parser


@logged
@profile
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'D1.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    mini_sig_info_file = os.path.join(
        args.models_path, 'mini_sig_info.tsv')

    if args.method == 'fit':

        dataset = Dataset.get(dataset_code)

        map_files = {}

        for ds in dataset.datasources:
            map_files[ds.name] = ds.data_path

        main._log.debug(
            "Running preprocess fit method for dataset " + dataset_code + ". Saving output in " + args.output_file)

        signaturesdir = os.path.join(args.models_path, "signatures")

        if os.path.exists(signaturesdir) is False:
            os.makedirs(signaturesdir)

        main._log.info("Parsing")
        parse_level(mini_sig_info_file, map_files, signaturesdir)

        ik_matrices = os.path.join(args.models_path, 'ik_matrices_fit')

        if os.path.exists(ik_matrices) is False:
            os.makedirs(ik_matrices, 0o775)

        connectivitydir = os.path.join(args.models_path, 'connectivity_fit')

        if os.path.exists(connectivitydir) is False:
            os.makedirs(connectivitydir, 0o775)

        consensus = os.path.join(args.models_path, "consensus_fit.h5")

        min_idxs = 10

        cp_sigs = set()
        with open(mini_sig_info_file) as f:
            for l in f:
                l = l.rstrip("\n").split("\t")
                if l[2] == "trt_cp":
                    cp_sigs.update([l[0]])

        sig_map = {}

        for SIG in cp_sigs:
            sig_map[SIG] = {"file": "%s/%s.h5" % (signaturesdir, SIG)}

    if args.method == 'predict':

        ik_matrices = os.path.join(args.models_path, 'ik_matrices_predict')

        if os.path.exists(ik_matrices) is False:
            os.makedirs(ik_matrices, 0o775)

        connectivitydir = os.path.join(
            args.models_path, 'connectivity_predict')

        if os.path.exists(connectivitydir) is False:
            os.makedirs(connectivitydir, 0o775)

        consensus = os.path.join(args.models_path, "consensus_predict.h5")

        min_idxs = 1

        sig_map = {}

        with open(args.input_file) as f:
            for l in f:
                items = l.split('\t')
                sig_map[items[0]] = {"up": items[2].split(
                    ","), "down": items[3].split(",")}

    WD = os.path.dirname(os.path.realpath(__file__))

    connectivity_script = WD + "/connectivity.py"

    ikmatrices_script = WD + "/do_agg_matrices.py"

    main._log.info("Getting signature files...")

    job_path = os.path.join(Config().PATH.CC_TMP, "job_conn")

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
    params["memory"] = 10
    # job command
    singularity_image = Config().PATH.SINGULARITY_IMAGE
    command = "singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {} {}".format(
        singularity_image, connectivity_script, mini_sig_info_file, signaturesdir,
        connectivitydir, GSE92742_Broad_LINCS_pert_info, min_idxs)
    # submit jobs
    cluster = HPC(Config())
    cluster.submitMultiJob(command, **params)

    if cluster.status() == 'error':
        main._log.error(
            "Connectivity job produced some errors. The preprocess script can't continue")
        sys.exit(1)

    main._log.info("Reading L1000")
    inchikey_sigid, inchikey_inchi, siginfo = read_l1000(
        mini_sig_info_file, connectivitydir)

    main._log.info("Doing aggregation matrices")

    job_path = os.path.join(Config().PATH.CC_TMP, "job_agg_matrices")

    if os.path.isdir(job_path):
        shutil.rmtree(job_path)
    os.mkdir(job_path)

    params = {}

    params["num_jobs"] = len(inchikey_sigid.keys()) / 10
    params["jobdir"] = job_path
    params["job_name"] = "CC_D1_agg_mat"
    params["elements"] = inchikey_sigid.keys()
    params["memory"] = 10
    # job command
    singularity_image = Config().PATH.SINGULARITY_IMAGE
    command = "singularity exec {} python {} <TASK_ID> <FILE> {} {} {}".format(
        singularity_image, ikmatrices_script, mini_sig_info_file, connectivitydir, ik_matrices)
    # submit jobs
    cluster = HPC(Config())
    cluster.submitMultiJob(command, **params)

    if cluster.status() == 'error':
        main._log.error(
            "Agg_matrices job produced some errors. The preprocess script can't continue")
        sys.exit(1)

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

    orderwords = list(words)
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_raw[k]:
            raws[i][wordspos[word[0]]] += word[1]

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))


if __name__ == '__main__':
    main()
