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


def es_score(idxs, ref_expr, min_idxs, p=1):
    """
    Enrichment score
    idx: list of gene indices
    ref_expr: expression levels of all genes in the signature which we want to
    do connectivity

    """
    # if not enough genes matching in the up/down regulated list (10 minimum)
    if len(idxs) < min_idxs:
        return 0.
    # number of genes in the expression profile
    N = len(ref_expr)
    # number of matches found
    Nh = len(idxs)
    # normalise by 1/the gene number difference between expr profile and
    # query list of up/down regulated genes
    norm = 1. / (N - Nh)

    # Return a new array of given shape and type filled
    # with the normalization factor
    miss = np.full(N, norm)
    # initialize the matching gene positions of gene expression profile to 0
    miss[idxs] = 0.

    # array of size (number of genes in expression profile) initialized with 0
    hit_nums = np.zeros(N)

    # Where a match occurs, replace zero by the expression level of that gene
    hit_nums[idxs] = np.abs(ref_expr[idxs])**p

    # Normalize this array dividing by the total number of matches
    hit = hit_nums / np.sum(hit_nums)
    # P_hit = hit
    P_miss = miss
    # now transform hit into its cumulative sum array.
    P_hit = np.cumsum(hit)
    P_miss = np.cumsum(miss)
    # element-wise difference between the two cumulative sum arrays
    ES = P_hit - P_miss
    # Return the max value of this difference array (in absolute value)
    return ES[np.argmax(np.abs(ES))]


def connectivity_score(up, dw, signature_file, signatures_dir, min_idxs):

    with h5py.File("%s/%s" % (signatures_dir, signature_file), "r") as hf:
        # i.e [ 5.02756786,  4.73850965,  4.49766302 ..]
        expr = hf["expr"][:]
        # i.e ['AGR2', 'RBKS', 'HERC6', ..., 'SLC25A46', 'ATP6V0B', 'SRGN']
        gene = hf["gene"][:].astype(str)

    up_idxs = np.where(np.in1d(gene, up))[0]
    dw_idxs = np.where(np.in1d(gene, dw))[0]

    es_up = es_score(up_idxs, expr, min_idxs)
    es_dw = es_score(dw_idxs, expr, min_idxs)

    if np.sign(es_up) == np.sign(es_dw):
        return 0
    else:
        return (es_up - es_dw) / 2


def signature_info(mini_sig_info_file):
    d = {}
    with open(mini_sig_info_file, "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            d[line[0]] = (line[1], line[2], line[3])
    return d


# Main

def main(SIG, up, dw, mini_sig_info_file, signatures_dir, connectivity_dir,
         touch, min_idxs, output_h5):
    """
    Args:
    SIG:(str): diff gene expression signature id
    up(list): list of string of up-regulated genes
    dw(list): list of string of down-regulated genes
    mini_sig_info_file(str: Path to mini_sig_info_file.tsv, which contains info
        about each signature (string)
    signatures_dir(str): Path to the directory that contains the gene diff
        expression signature h5 files (string)
    touch(set): Perturbagen ids belonging to the touchstone dataset
    min_idxs(int): minimum number of genes matching in the up/down regulated
        list (10 by default)
    output_h5(str): Path to output H5 connectivity file

    """

    # dict version of mini_sig_info_file.tsv
    sig_info = signature_info(mini_sig_info_file)
    # sign_id: (pert_id, treatment, cell_line, is_touchstone)

    # Find signature in touchstone
    touch_files = dict()
    for f in os.listdir(signatures_dir):
        if ".h5" not in f:
            continue
        # print("match against-->", f)
        # file name without extension, ex: REP.A001_A375_24H:A19.h5
        sig = f.split("/")[-1].split(".h5")[0]
        # (pert_id, treatment, cell_line, is_touchstone)
        sinfo = sig_info[sig]

        if sinfo[0] not in touch or sinfo[2] not in core_cells:
            # print(f,"not in touchstone or in core_cells, skipping")
            continue
        touch_files[f] = (sinfo[0], sinfo[2])

    # Going through all h5 files of gene expression data to match our list of
    # up/down-regulated genes. Each signature will be compared with all the
    # others, and connectivity scores are calculated
    R = []
    # These dicts will never throw a KeyError, if the key doesn't exist
    CTp = collections.defaultdict(list)
    # it creates it and puts an empty list as the default value
    CTm = collections.defaultdict(list)
    for f, sinfo in list(touch_files.items()):
        cs = connectivity_score(up, dw, f, signatures_dir, min_idxs)
        R.append((sig, cs))     # signature:id, connectivity score
        if cs > 0:
            CTp[sinfo] += [cs]  # treatment, cell_line
        elif cs < 0:
            CTm[sinfo] += [cs]
        else:
            continue

    # median of positive connec. scores found for each treatment and cell_line
    CTp = dict((k, np.median(v)) for k, v in CTp.items())
    # median of positive connec. scores found for each treatment and cell_line
    CTm = dict((k, np.median(v)) for k, v in CTm.items())

    # S Will contain all connectivity score for this query signature to all
    # others (sign_id, connect.score, normalized connect. score)
    S = []
    # for each signature:id, connectivity score (preselected for belonging
    # to the Touchstone dataset)
    for r in R:
        cs = r[1]
        sig = r[0]
        # # (pert_id, treatment, cell_line, is_touchstone)
        sinfo = sig_info[sig]

        if cs > 0:
            mu = CTp[(sinfo[1], sinfo[2])]
            # divide the connectivity score by the median of connec.
            # scores found for this treatment and cell_line
            ncs = cs / mu
        elif cs < 0:
            mu = -CTm[(sinfo[1], sinfo[2])]
            ncs = cs / mu
        else:
            ncs = 0.
        # add (sign_id, connect.score, normalized connect. score)
        S += [(r[0], cs, ncs)]

    # sort by sign_id (preselected for belonging to the Touchstone dataset)
    S = sorted(S, key=lambda tup: tup[0])
    # Write signatures id only to refer to the h5 file
    if not os.path.exists("%s/signatures.tsv" % connectivity_dir):
        with open("%s/signatures.tsv" % connectivity_dir, "w") as f:
            for s in S:
                f.write("%s\n" % s[0])

    with h5py.File(output_h5, "w") as hf:
        # connectivity score
        es = np.array([s[1] * 1000 for s in S]).astype(np.int16)
        # normalized connectivity score
        nes = np.array([s[2] * 1000 for s in S]).astype(np.int16)
        hf.create_dataset("es", data=es)
        hf.create_dataset("nes", data=nes)


if __name__ == '__main__':

    task_id = sys.argv[1]
    filename = sys.argv[2]
    mini_sig_info_file = sys.argv[3]
    signatures_dir = sys.argv[4]
    connectivity_dir = sys.argv[5]
    pert_info = sys.argv[6]        # NSex: GSE92742_Broad_LINCS_pert_info.txt
    sig_info = sys.argv[7]  # GSE92742_Broad_LINCS_sig_info.txt
    min_idxs = int(sys.argv[8])    # was 10 in our case

    # contains signid: path_to_the sign h5 file
    inputs = pickle.load(open(filename, 'rb'))
    sigs = inputs[task_id]

    # Obtain touchstone, name of the trt_oe has change so we need to mapp
    touch = set()
    tou_oe = set()

    with open(pert_info, "r") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n").split("\t")
            trt = line[2]
            if (trt == "trt_oe") and (line[3] == '1'):
                tou_oe.add(line[0])
            # checks the treatment record (cp, sh)
            if trt not in ["trt_cp", "trt_sh.cgs"]:
                continue
            if line[3] == '0':
                continue
            # Keep perturbagen ids belonging to the touchstone dataset
            touch.add(line[0])

    # Obtain new id of trt_oe touchstone
    with open(sig_info, "r") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n").split("\t")
            if line[1] in tou_oe:
                touch.add(line[0].split(':')[1])

    # k=signid1, v={'file': pathtosignature1.h5}
    for k, v in sigs.items():
        output_h5 = os.path.join(connectivity_dir, k + '.h5')
        if not os.path.exists(output_h5):
            # If up/downregulated genes have already been selected
            if "up" in v:
                main(k, v["up"], v["down"], mini_sig_info_file, signatures_dir,
                     connectivity_dir, touch, min_idxs, output_h5)
            else:
                # select up / down regulated genes from the GEX profile
                with h5py.File(v["file"], "r") as hf:
                    # i.e [ 5.02756786,  4.73850965,  4.49766302 ..]
                    expr = hf["expr"][:]
                    # i.e ['AGR2', 'RBKS', 'HERC6', ..., 'SRGN']
                    gene = hf["gene"][:]

                # Make a np array of (gene, diff expression),
                # sorted by epr level
                R = np.array(sorted(zip(gene, expr), key=lambda tup: -tup[1]),
                             dtype=np.dtype([('gene', '|S300'),
                                             ('expr', np.float)]))
                # R contains 12328 genes
                #       array([(b'AGR2',  5.02756786), (b'RBKS',  4.73850965),
                #  (b'HERC6',  4.49766302), ..., (b'SLC25A46', -6.47712374),
                #  (b'ATP6V0B', -6.93565464), (b'SRGN', -8.43125248)],

                # the first 250 genes are considered up-regulated
                up = R[:250]
                # the last 250 genes are considered down-regulated
                dw = R[-250:]
                # then keep up/down-regulated genes whose expression is a least
                # 2 units absolute value
                up = set(up['gene'][up['expr'] > 2])
                # Then it is only an array of gene names
                dw = set(dw['gene'][dw['expr'] < -2])

                # decode the bytes into Py3 strings
                up = list({s.decode() for s in up})
                dw = list({s.decode() for s in dw})

                # main will take the list of up/down regulated-genes for this
                # sign_id(k) and compare match it to all others
                main(k, up, dw, mini_sig_info_file, signatures_dir,
                     connectivity_dir, touch, min_idxs, output_h5)
        else:
            print("{} already exists, skipping".format(output_h5))
