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

only_touchstone = True # Doing the connectivity only with touchstone signatures

def es_score(idxs, ref_expr, min_idxs, p=1):
    """
    Enrichment score
    idx: list of gene indices
    ref_expr: expression levels of all genes in the signature which we want to do connectivity

    """
    if len(idxs) < min_idxs:  # if not enough genes matching in the up/down regulated list (10 minimum)
        return 0.
    N = len(ref_expr)         # number of genes in the expression profile
    Nh = len(idxs)            # number of matches found 
    norm = 1. / (N - Nh)      # normalise by 1/the gene number difference between expr profile and query list of up/down regulated genes

    miss = np.full(N, norm)       # Return a new array of given shape and type filled with the normalization factor
    miss[idxs] = 0.           # initialize the matching gene positions of the gene expression profile to zero

    hit_nums = np.zeros(N)    # array of size (number of genes in expression profile) initialized with zeros

    hit_nums[idxs] = np.abs(ref_expr[idxs])**p # Where a match occurs, replace zero by the expression level of that gene
     
    hit = hit_nums / np.sum(hit_nums)  # Normalize this array dividing by the total number of matches
    #P_hit = hit
    P_miss = miss
    P_hit = np.cumsum(hit)             # now transform hit into its cumulative sum array.
    P_miss = np.cumsum(miss)            
    ES = P_hit - P_miss                # element-wise difference between the two cumulative sum arrays
    return ES[np.argmax(np.abs(ES))]   # Return the max value of this difference array (in absolute value)

def connectivity_score(up, dw, signature_file, signatures_dir, min_idxs):

    with h5py.File("%s/%s" % (signatures_dir, signature_file), "r") as hf:
        expr = hf["expr"][:]  # i.e [ 5.02756786,  4.73850965,  4.49766302 ..]
        gene = hf["gene"][:].astype(str)  #i.e ['AGR2', 'RBKS', 'HERC6', ..., 'SLC25A46', 'ATP6V0B', 'SRGN']

    up_idxs = np.where(np.in1d(gene,up))[0]
    dw_idxs = np.where(np.in1d(gene,dw))[0]

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

def main(SIG, up, dw, mini_sig_info_file, signatures_dir, connectivity_dir, touch, min_idxs, output_h5):
    """ NS, main:
    SIG:                diff gene expression signature id (string)
    up:                 set of up-regulated genes in this signature (set of strings b'')
    dw:                 set of down-regulated genes in this signature (set of strings)
    mini_sig_info_file: Path to mini_sig_info_file.tsv, which contains info about each signature (string)
    signatures_dir:     Path to the directory that contains the gene diff expression signature h5 files (string)
    touch:              Set of perturbagen ids belonging to the touchstone dataset (set of strings)
    min_idxs:           integer (10 in the update)
    output_h5:          str, name of the output h5 connectivity file

    """

    sig_info = signature_info(mini_sig_info_file)  # dict version of mini_sig_info_file.tsv
                                                   # sign_id: (pert_id, treatment, cell_line, is_touchstone)

    CTp = collections.defaultdict(list)            # These dicts will never throw a KeyError, if the key doesn't exist
    CTm = collections.defaultdict(list)            # it creates it and puts an empty list as the default value

    R = []
    for f in os.listdir(signatures_dir):          # Going through all h5 files of gene expression data to match our list of up/down-regulated genes
        if ".h5" not in f:
            continue
        #print("match against-->", f)
        sig = f.split("/")[-1].split(".h5")[0]   # file name without extension, ex: REP.A001_A375_24H:A19.h5
        sinfo = sig_info[sig]                    # (pert_id, treatment, cell_line, is_touchstone)

        if sinfo[0] not in touch or sinfo[2] not in core_cells:
            #print(f,"not in touchstone or in core_cells, skipping")
            continue

        # Each signature will be compared with all the others, and connectivity scores are calculated
        cs = connectivity_score(up, dw, f, signatures_dir, min_idxs)

        R += [(sig, cs)]     # signature:id, connectivity score

        if cs > 0:
            CTp[(sinfo[1], sinfo[2])] += [cs]  # treatment, cell_line
        elif cs < 0:
            CTm[(sinfo[1], sinfo[2])] += [cs]
        else:
            continue

    CTp = dict((k, np.median(v)) for k, v in CTp.items()) # median of positive connec. scores found for each treatment and cell_line
    CTm = dict((k, np.median(v)) for k, v in CTm.items()) # median of positive connec. scores found for each treatment and cell_line

    # S Will contain all connectivity score for this query signature to all others (sign_id, connect.score, normalized connect. score)
    S = []
    for r in R:   #for each signature:id, connectivity score (preselected for belonging to the Touchstone dataset)
        cs = r[1]
        sig = r[0]
        sinfo = sig_info[sig]  # # (pert_id, treatment, cell_line, is_touchstone)

        if cs > 0:
            mu = CTp[(sinfo[1], sinfo[2])]
            ncs = cs / mu      # divide the connectivity score by the median of connec. scores found for this treatment and cell_line
        elif cs < 0:
            mu = -CTm[(sinfo[1], sinfo[2])]
            ncs = cs / mu
        else:
            ncs = 0.
        S += [(r[0], cs, ncs)]    # add (sign_id, connect.score, normalized connect. score)

    S = sorted(S, key=lambda tup: tup[0])  # sort by sign_id (preselected for belonging to the Touchstone dataset)
    if not os.path.exists("%s/signatures.tsv" % connectivity_dir): #Write signatures id only to refer to the h5 file
        with open("%s/signatures.tsv" % connectivity_dir, "w") as f:
            for s in S:
                f.write("%s\n" % s[0])

    # Each signature will show its connectivity to all other signatures in this h5 file
    with h5py.File(output_h5, "w") as hf:
        es = np.array([s[1] * 1000 for s in S]).astype(int)   # connectivity score
        nes = np.array([s[2] * 1000 for s in S]).astype(int)  # normalized connectivity score
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

    inputs = pickle.load(open(filename, 'rb'))  # contains signid: path_to_the sign h5 file
    sigs = inputs[task_id]                      # value for a particular task id, is a dict which values are themselves dict

    ### EP: Obtain touchstone ### - Name of the trt_oe has change so we need to mapp
    touch = set()
    tou_oe = set()

    with open(pert_info, "r") as f:
        f.readline() # skip header
        for l in f:
            l = l.rstrip("\n").split("\t")
            trt = l[2]
            if (trt == "trt_oe") and (l[3] == '1'):
                tou_oe.add(l[0])
            # checks the treatment record (cp, sh)
            if trt not in ["trt_cp", "trt_sh.cgs"]:
                continue
            if l[3] == '0':
                continue
            touch.add(l[0])                              # Keep perturbagen ids belonging to the touchstone dataset

    ### EP: Obtain new id of trt_oe touchstone ###
    with open(sig_info, "r") as f:
        f.readline() # skip header
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[1] in tou_oe: 
                touch.add(l[0].split(':')[1])
        
    for k, v in sigs.items():               # k=signid1, v={'file': pathtosignature1.h5}

        output_h5 = os.path.join(connectivity_dir,k+'.h5')
        if not os.path.exists(output_h5):

            #print("Recovering up/down-regulated genes in signature {}".format(k))
            if "up" in v:                       # If up/downregulated genes have already been selected (not our case)

                main(k, v["up"], v["down"], mini_sig_info_file, signatures_dir, connectivity_dir, touch, min_idxs, output_h5)
            else:
                # NS: select up / down regulated genes from the gene expression profile
                with h5py.File(v["file"], "r") as hf:      # pathtosignature1.h5
                    expr = hf["expr"][:]                   # i.e [ 5.02756786,  4.73850965,  4.49766302 ..]
                    gene = hf["gene"][:]                   # i.e ['AGR2', 'RBKS', 'HERC6', ..., 'SLC25A46', 'ATP6V0B', 'SRGN']

                # Make a np array of (gene, diff expression), sorted by epr level
                R = np.array(sorted(zip(gene, expr), key=lambda tup: -tup[1]), dtype=np.dtype([('gene', '|S300'), ('expr', float)]))
                # R contains 12328 genes
                  #       array([(b'AGR2',  5.02756786), (b'RBKS',  4.73850965),
                  #  (b'HERC6',  4.49766302), ..., (b'SLC25A46', -6.47712374),
                  #  (b'ATP6V0B', -6.93565464), (b'SRGN', -8.43125248)],
                  # dtype=[('gene', 'S300'), ('expr', '<f8')])--> these are bytes, need to decode the output!!!

                up = R[:250]       # the first 250 genes are considered up-regulated
                dw = R[-250:]      # the last 250 genes are considered down-regulated
                up = set(up['gene'][up['expr'] > 2]) # then keep up/down-regulated genes whose expression is a least 2 units absolute value
                dw = set(dw['gene'][dw['expr'] < -2]) # Then it is only an array of gene names

                # decode the bytes into Py3 strings
                up = list({s.decode() for s in up})
                dw = list({s.decode() for s in dw})

                #  main will take the list of up/down regulated-genes for this sign_id(k) and compare match it to all others
                main(k, up, dw, mini_sig_info_file, signatures_dir, connectivity_dir, touch, min_idxs, output_h5)
        else:
            print("{} already exists, skipping".format(output_h5))
