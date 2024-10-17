import os
import sys
import h5py
import glob
import shutil
import logging
import tempfile
import numpy as np
import collections
from scipy.stats import rankdata
from chemicalchecker.util import logged, Config
from chemicalchecker.util.hpc import HPC
from chemicalchecker.util.transform import Gaussianize
from chemicalchecker.database import Dataset, Molrepo
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.core.signature_data import DataSignature

# Variables
TEST = False  # once the 'signatures' dir exists with h5 inside, you can copy a few of them to 'signatures_test' and check if it's working
CHUNK_SIZE=10  # number of tasks per single job sent to sge
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:] #NS D1.001
features_file = "features.h5"

# Functions
def parse_level(mini_sig_info_file, map_files, signaturesdir):
    '''
    mini_sig_info_file = name / directory to create mini_sig_info_file
    map_files = dictionary with the paths of the files needed to parse
    signaturesdir = directory to save the signatures in the needed format
    '''
    try:
        from cmapPy.pandasGEXpress import parse
    except ImportError:
        raise ImportError("requires cmapPy " + "https://github.com/cmap/cmapPy")
    
    readyfile = "sigs.ready"
    if os.path.exists(os.path.join(signaturesdir, readyfile)):
        print("sigs.ready is present in {}, nothing to parse!".format(signaturesdir))
        return
    
    ### Obtain touchstone ### - Name of the trt_oe has change so we need to mapp
    
    touchstone = set()
    tou_oe = set()

    GSE92742_Broad_LINCS_pert_info = os.path.join(map_files["GSE92742_Broad_LINCS_pert_info"], "GSE92742_Broad_LINCS_pert_info.txt") 
    # header of this file:
    # pert_id>pert_iname>-----pert_type>------is_touchstone>--inchi_key_prefix>-------inchi_key>------canonical_smiles>-------pubchem_cid
    # 56582>--AKT2>---trt_oe>-0>-------666>----666>----666>----666
    # We need to use the file of the last version of LINCS (2017) because in t2020 version don't give information about Touchstone

    with open(GSE92742_Broad_LINCS_pert_info, "r") as f:
        f.readline() # skip header
        for l in f:
            l = l.rstrip("\n").split("\t")
            trt = l[2]
            if (trt == "trt_oe") and (l[3] == '1'):
                tou_oe.add(l[0])
            # checks the treatment record (cp, sh or oe)
            if trt not in ["trt_cp", "trt_sh.cgs"]:
                continue
            if l[3] == '0':
                continue
            touchstone.add(l[0])                              # Keep perturbagen ids belonging to the touchstone dataset

    ### Obtain new id of trt_oe touchstone ###

    GSE92742_Broad_LINCS_sig_info = os.path.join(map_files["GSE92742_Broad_LINCS_sig_info"], "GSE92742_Broad_LINCS_sig_info.txt") 
    with open(GSE92742_Broad_LINCS_sig_info, "r") as f:
        f.readline() # skip header
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[1] in tou_oe: 
                touchstone.add(l[0].split(':')[1])

    ## Obtain Gene symbols ###

    LINCS_2020_gene_info = os.path.join(map_files["geneinfo_beta"], "geneinfo_beta.txt") 
    genes = {}

    # gene_id>-----gene_symbol>-ensembl_id>--gene_title>-------gene_type>----src>---feature_space>
    # 750>----GAS8-AS1>---ENSG00000221819>---GAS8 antisense RNA 1>----ncRNA>------NCBI>---inferred

    with open(LINCS_2020_gene_info, "r") as f:
        f.readline()
        for l in f:
            l = l.split("\t")
            genes[l[0]] = l[1]     # keep gene id and map it to its symbol   
                
    ### Signature info and signature metrics (gene expr signatures - LEVEL V metadata) ###
   
    LINCS_2020_sig_info = os.path.join(map_files["siginfo_beta"], "siginfo_beta.txt") 
    # LINCS_2020_sig_info = os.path.join(path_metadata, "siginfo_beta.txt") 
    sig_info = {}
    sigs = collections.defaultdict(list)

    #'bead_batch'>---'nearest_dose'>---'pert_dose'>---'pert_dose_unit','pert_idose'>---'pert_itime'>---'pert_time'>---'pert_time_unit'>---cell_mfc_name'>---'pert_mfc_id'>---'nsample'>---
    #'cc_q75'>---'ss_ngene'>---'tas'>---'pct_self_rank_q25'>---'wt'>---'median_recall_rank_spearman'>---'median_recall_rank_wtcs_50'>---'median_recall_score_spearman'>---
    #'median_recall_score_wtcs_50'>---'batch_effect_tstat'>---'batch_effect_tstat_pct'>---'is_hiq'>---'qc_pass'>---'pert_id'>---'sig_id'>---'pert_type'>---'cell_iname'>---
    #'det_wells'>---'det_plates'>---'distil_ids'>---'build_name'>---'project_code'>---'cmap_name'>---'is_ncs_exemplar'

    #'b17>---NAN>---100>---ug/ml>---100 ug/ml>---336 h>---336>---h>---N8>---BRD-U44432129>---4>---
    #0.6164>---446>---0.530187>---0>---0.26,0.26,0.22,0.26>---0.925926>---1.15741>---0.548655>---
    #0.705263>----2.31>---0.488085>---1>---1>---BRD-U44432129>---MET001_N8_XH:BRD-U44432129:100:336>---trt_cp>---NAMEC8>---
    #H05|H06|H07|H08>---MET001_N8_XH_X1_B17>---MET001_N8_XH_X1_B17:H05|MET001_N8_XH_X1_B17:H06|MET001_N8_XH_X1_B17:H07|MET001_N8_XH_X1_B17:H08>--->---MET>---BRD-U44432129>---0'

    with open(LINCS_2020_sig_info, "r") as f:
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[24] in touchstone:          # only touchstone perturbagens --> select touchstone
                v = 1
            else:
                v = 0
            sig_info[l[25]] = (l[24], l[26], l[27], v)  # map signature_id with pert_id , pert_type, cell_id, is_touchstone

            if float(l[11]) >= 0.2:               # if cc_q75 <0.2
                trt = l[26]                        # pert_type
                sig_id = l[25]                     # sig_id
                if trt in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                    t = sig_info[sig_id]
                    tas = float(l[13]) # tas - measure of strong of the expression
                    nsamp = int(l[10]) # number of inst for creating the signature
                    # Mapping (pert_id, cell_id) : list of tuples of (sign_id, treat_type, tas, nsample, phase)
                    sigs[(t[0], t[2])] += [(sig_id, trt, tas, nsamp)] 

    def get_exemplar(v):  # ex of v: [(sig_id, trt, tas, nsamp, phase), (sig_id, trt, tas, nsamp, phase), ...]
        s = [t for t in v if t[3] >= 2 and t[3] <= 6]   # keep tuples of v for which nsample between 2 and 6
        if not s:                                       # if no tuple is selected (the list is empty)
            s = v                                       # keep the original list of tuples
        sel = None
        max_tas = 0.
        for x in s:                                     # for each (filtered) tuple of v (sig_id, trt, tas, nsamp, phase)
            if not sel:                                 # First tuple 
                sel = (x[0])                     # sel = (sig_id, phase) ; max_tas=tas
                max_tas = x[2]
            else:                                       # Other tuples
                if x[2] > max_tas:                      # Keep the tuple with BIGGEST tas
                    sel = (x[0])
                    max_tas = x[2]
        return sel                                     # return (sig_id, phase) fo the tuple with biggest tas

    # Transform sigs in a dict (pert_id, cell_id) --> (sig_id, phase) fo the tuple with biggest tas
    sigs = dict((k, get_exemplar(v)) for k, v in sigs.items())

    cids = []

    # mini_sig_info_file = 'mini_sig_info_file.tsv' --> we already have this variable as argument of the method
    with open(mini_sig_info_file, "w") as f:
        for k, v in sigs.items():             #(pert_id, cell_id) --> (sig_id, phase) fo the tuple with biggest tas
            x = sig_info[v]          # (pert_id, pert_type, cell_id, istouchstone, 1)
    #             # Writes :
    #             # sig_id, pert_id, pert_type, cell_id, istouchstone) 
            f.write("%s\t%s\t%s\t%s\t%d\n" %(v, x[0], x[1], x[2], x[3]))
            cids += [(v, x[1])]           # (sig_id, pert_type)

    # The expression profiles (file of several GB)

    gtcx_cp = os.path.join(map_files["level5_beta_trt_cp_n720216x12328"], "level5_beta_trt_cp_n720216x12328.gctx") 
    gtcx_sh = os.path.join(map_files["level5_beta_trt_sh_n238351x12328"], "level5_beta_trt_sh_n238351x12328.gctx") 
    gtcx_oe = os.path.join(map_files["level5_beta_trt_oe_n34171x12328"], "level5_beta_trt_oe_n34171x12328.gctx") 

    genes_i = [genes[r[0]] for r in parse.parse(gtcx_cp, cid=[[x[0] for x in cids if x[1] == 'trt_cp'][0]]).data_df.iterrows()]
    genes_ii = [genes[r[0]] for r in parse.parse(gtcx_sh, cid=[[x[0] for x in cids if x[1] == 'trt_sh.cgs'][0]]).data_df.iterrows()]  # Just to make sure.
    genes_iii = [genes[r[0]] for r in parse.parse(gtcx_oe, cid=[[x[0] for x in cids if x[1] == 'trt_oe'][0]]).data_df.iterrows()]  # Just to make sure.

    main._log.info("Parsing GCTX files for each sign id")
    for cid in cids: # for each sign id  
        main._log.info("Sign id: {}".format(cid[0]))
        if cid[1] == 'trt_cp':
            expr = np.array(parse.parse(gtcx_cp, cid=[cid[0]]).data_df).ravel()
            genes = genes_i
        elif cid[1] == 'trt_sh.cgs':
            expr = np.array(parse.parse(gtcx_sh, cid=[cid[0]]).data_df).ravel()
            genes = genes_ii
        elif cid[1] == 'trt_oe':
            expr = np.array(parse.parse(gtcx_oe, cid=[cid[0]]).data_df).ravel()
            genes = genes_iii
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
    
    # LINCS_2020_cp_info = os.path.join(map_files["LINCS_2020"], "cp_info_inchikey_standard.txt") 

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

    with open(mini_sig_info_file, "r") as f:  # read the minisign file
        for l in f:
            l = l.rstrip("\n").split("\t")    # keeps perturbagens from touchdown
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
            inchikey_sigid[ik] += [sig_id]  # inchikey -> filename without extension

    #returns the following mappings from L1000:
    # inchikey_sigid:   mol_inchikey -> mol_id (h5 filename without extension)
    # inchikey_inchi:   mol_inchikey -> mol_inchi
    # siginfo:          REP.A001_A375_24H:K17 -> mol_id

    return inchikey_sigid, inchikey_inchi, siginfo

def get_summary(v):
    Qhi = np.percentile(v, 66)
    Qlo = np.percentile(v, 33)

    # If both scores are negative then the 33-percentile is returned which is in fact the 66-percentile
    if np.abs(Qhi) > np.abs(Qlo):
        return Qhi
    else:
        return Qlo

def do_consensus(ik_matrices, consensus):

    # NS Takes all individual matrices of sign-ik vs all perturbagens (each matrix is for a given ik)
    inchikeys = [ik.split(".h5")[0] for ik in os.listdir(ik_matrices) if ik.endswith(".h5")]

    def consensus_signature(ik):
        """
        Opens an ik matrix 
        takes all 66pc conn scores between signatures (rows) and perturbagens (cols)

        """
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "r") as hf:
            X = hf["X"][:]
        # It could be max, min...
        # NS takes the 66 percentile of columns (like what was done for rows in )
        # i.e 'averages' over the different signatures of the ik corresponding to that matrix
        # in the end we have a VECTOR of perc normalized conn scores of 1 perturbagen x all perturbagens from Touchstone
        return [ int(get_summary(X[:, j])) for j in range(X.shape[1])]

    # Stack all these vectors so that X is all perturbagens x all perturbagens and dump it as consensus_fit/predict.h5
    X = np.array([consensus_signature(ik) for ik in inchikeys])

    with h5py.File(consensus, "w") as hf:
        hf.create_dataset("inchikeys", data=np.array(inchikeys, DataSignature.string_dtype()))
        hf.create_dataset("X", data=X)

    return X, inchikeys

def process(X):
    """NS:  Looks like a function to remove noise in the output X"""
    def whiten(X):

        Xw = np.zeros(X.shape)

        for j in range(X.shape[1]):
            V = X[:, j]
            V = rankdata(V, "ordinal")
            gauss = Gaussianize(strategy="brute")
            gauss.fit(V)
            V = gauss.transform(V)
            Xw[:, j] = np.ravel(V)

        return Xw

    def cutoffs(X):
        # NS: cutoff 99-percentile of each column of X (over all the row perturbagens)
        return [np.percentile(X[:, j], 99) for j in range(X.shape[1])]

    # Vector of cutoffs values 1x n perturbagens
    cuts = cutoffs(X)

    Xcut = []
    # For each cutoff values (i.e each col of x)
    for j in range(len(cuts)):
        c = cuts[j]              # 99-percentile of column j
        v = np.zeros(X.shape[0]) # vector spannning all elements of 1 column
        v[X[:, j] > c] = 1       # The elements of this col of X which are bigger than the 99-percentile of the col will be assigned 1
        Xcut += [v]              # else 0 # add the binary vector as a line so we have to transpose evrything at the end

    Xcut = np.array(Xcut).T

    return Xcut                  # Binary matrix of connectivities


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
#@profile


def main(args):

    args = Preprocess.get_parser().parse_args(args)

    # NS ex: fir D1.001, args will be a mapping:
    # -o : signature0full_path/raw/preprocess.h5    # --output_file
    # -mp: signature0full_path/raw/models           # --model_path
    # -m: 'fit'                                     # --method

    # NS added: make skip everything if the final outputfile is already present
    if os.path.exists(args.output_file) and args.method == "fit":
        main._log.info("Preprocessed file {} is already present for {}, skipping the preprocessing step.".format(args.output_file,dataset_code))
        return


    mini_sig_info_file = os.path.join(args.models_path, 'mini_sig_info.tsv') # file with information about the exemplary signatures

    dataset = Dataset.get(dataset_code) #NS D1.001 dataset object built to queryour sql database
    #path_metadata = ''
    map_files = {}  # NS: will store datasource names of D1.00X and path to the corresponding files

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.

    # NS: 13 datasources in dataset_had_datasource table for the 2020 update
    # access to db only in fit mode since we want people to be able to use it in 
    try:
        for ds in dataset.datasources:
            map_files[ds.datasource_name] = ds.data_path       #Ns path is /aloy/scratch/sbnb-adm/CC/download/<datasource_name>

    except Exception as e:
        if args.method == 'fit':
            main._log.error("{}".format(e))
            sys.exit(1)
        else:
            main._log.info("Database 'dataset' cannot be accessed, will use local copies of required files if present")

    main._log.debug("Running preprocess fit method for dataset " + dataset_code + ". Saving output in " + args.output_file)

    signaturesdir = os.path.join(args.models_path, "signatures") #signature0full_path/raw/models/signatures
#                                                                  # contains 83 639 h5 files and a sigs.ready file
    if TEST:
        signaturesdir = os.path.join(args.models_path, "signatures_test") # You can copy a few h5 from signatures into  signatures_test

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

        # NS only testing a few signatures
        if TEST:
            dir_to_search=os.path.join(signaturesdir,"*.h5")
            print("Looking for h5 signatures in ",dir_to_search)
            signs_to_test= glob.glob(os.path.join(signaturesdir,"*.h5"))
            signs_to_test= [s.split('/')[-1][:-3] for s in signs_to_test]
            print("signs_to_test is", signs_to_test)

        # Parsing the record file
        ok=False
        with open(mini_sig_info_file) as f:
            for l in f:
                l = l.rstrip("\n").split("\t")

                if TEST:
                    print("l[0]",l[0])
                    if l[0] in signs_to_test and l[2] == "trt_cp":
                        # update the set cp_sigs with records such as 'REP.A001_A375_24H:E14'
                        print("selecting ", l[0])
                        cp_sigs.update([l[0]])
                        ok = True

                elif not TEST and l[2] == "trt_cp":
                    # update the set cp_sigs with records such as 'REP.A001_A375_24H:E14'
                    cp_sigs.update([l[0]])
                    ok = True

        if not ok:
            print("Problem with selecting signatures, cp_sigs is", cp_sigs)
            sys.exit(1)



        sig_map = {}
        # Populate sig_map dict with i.e 'REP.A001_A375_24H:E14' : {"file": "signature0full_path/raw/models/signatures/REP.A001_A375_24H:E14.h5"}
        for SIG in cp_sigs:
            sig_map[SIG] = {"file": "%s/%s.h5" % (signaturesdir, SIG)}
        if TEST: print("sig_map", sig_map)

    elif args.method == 'predict':

        mpath = tempfile.mkdtemp(prefix='predict_', dir=args.models_path)

        ik_matrices = tempfile.mkdtemp(prefix='ik_matrices_', dir=mpath)

        connectivitydir = tempfile.mkdtemp(prefix='connectivity_', dir=mpath)

        consensus = os.path.join(mpath, "consensus_predict.h5")

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]               # list of Xcut column indices with at least one 1
            features = set(features_list)                   # Just in case they some indices are duplicated

        min_idxs = 1

        sig_map = {}
        inchikey_inchi = {}

        inchikey_sigid = collections.defaultdict(list)

        # input data file from preprocess.call_preprocess
        # Need 4 columns: inchikey 'anything' comma-separated_list of_up_regulated_genes coma_separated_list_of_down_regulated_genes
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

                # Put the list of up/down regulated genes in a dictionary
                # Whose values are dictionaries
                # The presence of the 'up' and 'down' keys is recognized by do_agg_matrices.py
                sig_map[items[0] + "---" + ik] = {"up": items[2].split(","), "down": items[3].split(",")}

    WD = os.path.dirname(os.path.realpath(__file__))  # directory from which run.py is launched

    connectivity_script = WD + "/connectivity.py"     # scripts called by run.py in the same directory

    ikmatrices_script = WD + "/do_agg_matrices.py"

    readyfile = "conn.ready"

    config = Config()                                # reads os.environ["CC_CONFIG"]


    # NS CONNECTIVITY JOB 
    # OBTAIN -> one matrix per signature (1xnsign) 
    # rows: one signature, cols: all other signatures. 
    # Contains connectivity scores (raw and normalized) 
    # Note, connectivitydir is signature0full_path/raw/models/connectivity_fit
    if not os.path.exists(os.path.join(connectivitydir, readyfile)):   # contains 65 151 h5 files and conn.ready so False here

        main._log.info("Getting signature files...")

        job_path = os.path.join(mpath, "job_conn")  # contains a script file job-CC_D1_conn.sh and an input file which is a pickle from sig_map
                                                    # launches the 6514 jobs of connectivity.py with the input file, mini_sig_info.tsv, connectivity_fit
                                                    # GSE92742_Broad_LINCS_pert_info.txt as arguments

        if os.path.isdir(job_path):
            shutil.rmtree(job_path)
        os.mkdir(job_path)

        # NS: When running predict on standalone version of the checker, we won't have the postgres db
        # Have a copy of this file locally in this case
        try:
            GSE92742_Broad_LINCS_pert_info = os.path.join(map_files["GSE92742_Broad_LINCS_pert_info"], "GSE92742_Broad_LINCS_pert_info.txt")
            main._log.info("GSE92742_Broad_LINCS_pert_info.txt found from database")

        except Exception as e:
            GSE92742_Broad_LINCS_pert_info= os.path.join(args.models_path, "GSE92742_Broad_LINCS_pert_info.txt")
            if os.path.exists(GSE92742_Broad_LINCS_pert_info):
                main._log.info("Found: {}".format(GSE92742_Broad_LINCS_pert_info))
            else:
                main._log.error("Cannot find SE92742_Broad_LINCS_pert_info.txt, stopping!")
                sys.exit(1)
        try:
            GSE92742_Broad_LINCS_sig_info = os.path.join(map_files["GSE92742_Broad_LINCS_sig_info"], "GSE92742_Broad_LINCS_sig_info.txt")
            main._log.info("GSE92742_Broad_LINCS_sig_info.txt found from database")

        except Exception as e:
            GSE92742_Broad_LINCS_sig_info= os.path.join(args.models_path, "GSE92742_Broad_LINCS_sig_info.txt")
            if os.path.exists(GSE92742_Broad_LINCS_sig_info):
                main._log.info("Found: {}".format(GSE92742_Broad_LINCS_sig_info))
            else:
                main._log.error("Cannot find GSE92742_Broad_LINCS_sig_info.txt, stopping!")
                sys.exit(1)

        params = {}
        num_entries = len(sig_map.keys())
        # If there are less tasks to send than the numb of tasks per job then num_jobs is just num_entries (otherwise bug since dividing by CHUNK_SIZE tells it to send 0 jobs)
        params["num_jobs"] = int(num_entries / CHUNK_SIZE) if num_entries > CHUNK_SIZE else num_entries
        params["jobdir"] = job_path
        params["job_name"] = "CC_D1_conn"
        params["elements"] = sig_map  
        # reminder: sigmap is a dict with the following entry type in 'fit' mode:
        #'REP.A001_A375_24H:E14' : {"file": "signature0full_path/raw/models/signatures/REP.A001_A375_24H:E14.h5"}
        params["memory"] = 10
    
        # job command
        cc_package = os.path.join(config.PATH.CC_REPO, 'package')
        singularity_image = config.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} MKL_NUM_THREADS=1 singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {} {} {}".format(
            cc_package, os.environ['CC_CONFIG'],
            singularity_image, connectivity_script, mini_sig_info_file, signaturesdir,
            connectivitydir, GSE92742_Broad_LINCS_pert_info, GSE92742_Broad_LINCS_sig_info, min_idxs)

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

    readyfile = "agg_matrices.ready"


    # MATRIX AGGREGATION JOB
    # OBTAIN-> 1 matrix per perturbagen (inchikey).
    # rows: a couple of signatures (usually one) corresponding to the current inchikey
    # Each perturbagen from Touchstone (7841 for the 2020 update)--> take the 66percentile (or33) conn score among all signatures of a given perturbagen with our current sign (row)
    if not os.path.exists(os.path.join(ik_matrices, readyfile)):

        if args.method == 'fit':
            main._log.info("Reading L1000")

            # NS Get molecules from Lincs
            inchikey_sigid, inchikey_inchi, siginfo = read_l1000(mini_sig_info_file, connectivitydir)
            # inchikey_sigid:   mol_inchikey -> mol_id (h5 filename without extension)
            # inchikey_inchi:   mol_inchikey -> mol_inchi
            # siginfo:          REP.A001_A375_24H:K17 -> mol_id
    
        #LINCS_2020_cp_info = os.path.join(path_metadata, "cp_info_inchikey_standard.txt") 
#         try:
#             LINCS_2020_cp_info = os.path.join(map_files['LINCS_2020'], "cp_info_inchikey_standard.txt") 
#             main._log.info("cp_info_inchikey_standard.txt found from database")

#         except Exception as e:
#             LINCS_2020_cp_info= os.path.join(args.models_path, "cp_info_inchikey_standard.txt")
#             if os.path.exists(LINCS_2020_cp_info):
#                 main._log.info("Found: {}".format(LINCS_2020_cp_info))
#             else:
#                 main._log.error("Cannot find cp_info_inchikey_standard.txt, stopping!")
#                 sys.exit(1)
            
        main._log.info("Doing aggregation matrices")

        #/aloy/web_checker/package_cc/2020_01/full/D/D1/D1.001/sign0/raw/models/job_agg_matrices
        job_path = os.path.join(mpath, "job_agg_matrices")

        if os.path.isdir(job_path):
            shutil.rmtree(job_path)
        os.mkdir(job_path)

        params = {}

        num_entries = len(inchikey_sigid.keys())
        # If there are less tasks to send than the numb of tasks per job then num_jobs is just num_entries (otherwise bug since dividing by CHUNK_SIZE tells it to send 0 jobs sent)
        params["num_jobs"] = num_entries / CHUNK_SIZE if num_entries > CHUNK_SIZE else num_entries
        params["jobdir"] = job_path
        params["job_name"] = "CC_D1_agg_mat"
        params["elements"] = list(inchikey_sigid.keys())  # dict_key view of mol_ids
        params["memory"] = 10

        # job command
        singularity_image = config.PATH.SINGULARITY_IMAGE     
        cc_package = os.path.join(config.PATH.CC_REPO, 'package')
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

    # Stack all these vectors so that X is consensus connectivity score (66-percentile score over all signatures of a given perturbagen)
    # shape: all perturbagens x all perturbagens 
    # and dump it as consensus_predict.h5
    main._log.info("Doing consensus")
    X, inchikeys = do_consensus(ik_matrices, consensus)

    if TEST:
        print("X",X)
        print("inchikeys",inchikeys)

    # Binarization of X (1: good connectivity, 0 otherwise)
    main._log.info("Process output")


    Xcut = process(X)

    if TEST:
        print("Xcut",Xcut)

    main._log.info("Saving raws")
    inchikey_raw = {}

    # NS Going through the list of inchikeys processed
    for i in range(len(inchikeys)):
        ik = inchikeys[i]
        if np.sum(Xcut[i, :]) < 5:    # skip this inchikey If there are less than 5 good connectivities
            continue

        idxs = np.where(Xcut[i, :] == 1)[0]  # Indices of the Xcut cols where we have good connections for this ik
        inchikey_raw[ik] = [(str(x), 1) for x in idxs]   # store these indices in a dict  ik: [(idx1, 1), (idx2,1) etc]

    keys = []
    words = set()

    # NS added
    if len(inchikey_raw.keys()) == 0:
        main._log.info("FILTERING ELIMINATED EVERYTHING, SORRY!")
        sys.exit(1)

    # Going through the good connections of every inchikey
    for k in sorted(inchikey_raw.keys()):
        keys.append(str(k))                              # inchikey
        words.update([x[0] for x in inchikey_raw[k]])    # Update the set with col indices where good connections are found {idx1, idx2,}

    if args.method == 'predict' and features is not None: # NS: added the first cond, otherwise 'features' was unreferenced when not using 'predict'
        orderwords = features_list                        # prerecorded list of column indices of Xcut containing at least one 1
    else:
        orderwords = list(words)                         
        orderwords.sort()                                 # list of column indices of Xcut containing at least one 1

    if TEST: print("orderwords", orderwords)

    raws = np.zeros((len(keys), len(orderwords)), dtype=int)  # Matrix n_inchikeys x n_Xcut_col_indices
    wordspos = {k: v for v, k in enumerate(orderwords)}           # dict Xcut_index : i  (i being the index of the sorted Xcutindex)

    for i, k in enumerate(keys):                                  # Going through inchikeys again
        for word in inchikey_raw[k]:                              # ik: [(idx1, 1), (idx2,1) etc]--> going through this list
            raws[i][wordspos[word[0]]] += word[1]                 # raws[ik_index][X_cut_index_index] += 1

    # So our final matrix preprocess.h5
    # rows: inchikeys
    # columns: col indices of Xcut where at least one 1 was present (features)

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys, DataSignature.string_dtype()))
        hf.create_dataset("X", data=raws)
        hf.create_dataset("features", data=np.array(orderwords, DataSignature.string_dtype()))

    if args.method == "fit":
        # features.h5
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            # Keep the list of indices of Xcut where at least one 1 was present in a separate file
            # getting strings instead of bytes from the h5 file
            hf.create_dataset("features", data=np.array(orderwords, DataSignature.string_dtype()))

    if args.method == 'predict':
        #shutil.rmtree(mpath)
        #shutil.rmtree(connectivitydir)
        pass

if __name__ == '__main__':
    main(sys.argv[1:])

