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
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:] #NS D1.001
features_file = "features.h5"


def parse_level(mini_sig_info_file, map_files, signaturesdir):
    try:
        from cmapPy.pandasGEXpress import parse
    except ImportError:
        raise ImportError("requires cmapPy " + "https://github.com/cmap/cmapPy")

    readyfile = "sigs.ready"
    if os.path.exists(os.path.join(signaturesdir, readyfile)):
        print("sigs.ready is present in {}, nothing to parse!".format(signaturesdir))
        return

    touchstone = set()

    GSE92742_Broad_LINCS_pert_info = os.path.join(map_files["GSE92742_Broad_LINCS_pert_info"], "GSE92742_Broad_LINCS_pert_info.txt") 
    # header of this file:
    # pert_id>pert_iname>-----pert_type>------is_touchstone>--inchi_key_prefix>-------inchi_key>------canonical_smiles>-------pubchem_cid
    # 56582>--AKT2>---trt_oe>-0>-------666>----666>----666>----666


    with open(GSE92742_Broad_LINCS_pert_info, "r") as f:
        f.readline() # skip header
        for l in f:
            l = l.rstrip("\n").split("\t")
            trt = l[2]                                        # checks the trt (treatment?) record 
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                continue
            if l[3] == '0':
                continue
            touchstone.add(l[0])                              # Keep perturbagen ids belonging to the touchstone dataset

    # Gene symbols (same for LINCS I and II)
    GSE92742_Broad_LINCS_gene_info = os.path.join(map_files["GSE92742_Broad_LINCS_gene_info"], "GSE92742_Broad_LINCS_gene_info.txt")
    genes = {}

    # pr_gene_id>-----pr_gene_symbol>-pr_gene_title>--pr_is_lm>-------pr_is_bing
    # 780>----DDR1>---discoidin domain receptor tyrosine kinase 1>----1>------1
    # 7849>---PAX8>---paired box 8>---1>------1

    with open(GSE92742_Broad_LINCS_gene_info, "r") as f:
        f.readline()
        for l in f:
            l = l.split("\t")
            genes[l[0]] = l[1]     # keep gene id and map it to its symbol

    sig_info_ii = {}

    # signature infos ii (gene expr profiles)
    for file_name in glob.glob(os.path.join(map_files["GSE70138_Broad_LINCS_sig_info"], "*.txt")):
    # sig_id>-pert_id>pert_iname>-----pert_type>------cell_id>pert_idose>-----pert_itime>-----distil_id
    # LJP005_A375_24H:A03>----DMSO>---DMSO>---ctl_vehicle>----A375>----666>---24h>---LJP005_A375_24H_X1_B19:A03|LJP005_A375_24H_X2_B19:A03|LJP005_A375_24H_X3_B19:A03

        with open(file_name, "r") as f:
            f.readline()
            for l in f:
                l = l.rstrip("\n").split("\t")
                if l[1] in touchstone:          # only touchstone perturbagens
                    v = 1
                else:
                    v = 0
                sig_info_ii[l[0]] = (l[1], l[3], l[4], v, 2)  # map signature_id (with pert_id , pert_type, cell_id, is_touchstone, 2)

    # Signature metrics
    sigs = collections.defaultdict(list)
    for file_name in glob.glob(os.path.join(map_files["GSE70138_Broad_LINCS_sig_metrics"], "*.txt")):
        # linenum----->sig_id>-distil_cc_q75>--distil_ss>------pert_id pert_iname>-----pert_type>------tas>----ngenes_modulated_up_lm>-ngenes_modulated_dn_lm>-pct_self_rank_q25>------distil_nsample
        #0>------REP.A001_A375_24H:A03>--0.1>----4.11955>DMSO>---DMSO>---ctl_vehicle>----0.131454>-------36>-----28>-----18.4375>3
        #1>------REP.A001_A375_24H:A04>--0.1>----3.47207>DMSO>---DMSO>---ctl_vehicle>----0.105571>-------26>-----20>-----12.9241071429>--3
        with open(file_name, "r") as f:
            f.readline()
            for l in f:
                l = l.rstrip("\n").split("\t")[1:]  # here the first field is excluded
                if float(l[1]) < 0.2:               # if distil_ss <0.2
                    continue
                trt = l[5]                          # Treatment type (ctl_vehicle or trt_cp etc)
                sig_id = l[0]                       # signature_id
                if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                    continue
                if sig_id not in sig_info_ii:       # must include only the touchstone perturbagen dataset 
                    continue
                v = sig_info_ii[sig_id]             # info previously recorded about the signature
                tas = float(l[6])                   # some number
                nsamp = int(l[-1])                  # distil_nsample
                phase = 2                           # identifies the sign ii dict

                # Mapping (pert_id, cell_id) : list of tuples of (sign_id, treat_type, tas, nsample, phase)
                sigs[(v[0], v[2])] += [(sig_id, trt, tas, nsamp, phase)]  

    sig_info_i = {}

    GSE92742_Broad_LINCS_sig_info = os.path.join(map_files["GSE92742_Broad_LINCS_sig_info"], "GSE92742_Broad_LINCS_sig_info.txt")
    # sig_id>-pert_id>pert_iname>-----pert_type>------cell_id>pert_dose>------pert_dose_unit>-pert_idose>-----pert_time>------pert_time_unit>-pert_itime>-----distil_id
    #AML001_CD34_24H:A05>----DMSO>---DMSO>---ctl_vehicle>----CD34>---0.1>----%>------0.1 %>--24>-----h>------24 h>---AML001_CD34_24H_X1_F1B10:A05

    with open(GSE92742_Broad_LINCS_sig_info, "r") as f:
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[1] in touchstone:
                v = 1
            else:
                v = 0
            # Mapping sign_id : (pert_id, pert_type, cell_id, istouchstone, 1)  
            sig_info_i[l[0]] = (l[1], l[3], l[4], v, 1)  


    GSE92742_Broad_LINCS_sig_metrics = os.path.join(map_files["GSE92742_Broad_LINCS_sig_metrics"], "GSE92742_Broad_LINCS_sig_metrics.txt")
    # sig_id>-pert_id>pert_iname>-----pert_type>------distil_cc_q75>--distil_ss>------ngenes_modulated_up_lm>-ngenes_modulated_dn_lm>-tas>----pct_self_rank_q25>------is_exemplar>----distil_nsample
    # AML001_CD34_24H:BRD-A03772856:0.37037>--BRD-A03772856>--BRD-A03772856>--trt_cp>-0.4>----3.02055>8>------15>-----0.166769>-------4.97076>0>------2

    with open(GSE92742_Broad_LINCS_sig_metrics, "r") as f:
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if float(l[4]) < 0.2:             #distil_cc_q75
                continue
            trt = l[3]                        # treatment type
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]:
                continue
            sig_id = l[0]                     # signature_id
            if sig_id not in sig_info_i:      # Ensure that it is present in the correspnding info file
                continue
            v = sig_info_i[sig_id]            # (pert_id, pert_type, cell_id, istouchstone, 1)
            tas = float(l[8])                 # ex 0.166769
            nsamp = int(l[-1])                # nsample, ex 2
            phase = 1                         # identifies the sign i dict

            # Mapping (pert_id, cell_id) to list of tuples (sig_id, treatment_type, tas, nsample, phase)
            sigs[(v[0], v[2])] += [(sig_id, trt, tas, nsamp, phase)]

    def get_exemplar(v):  # ex of v: [(sig_id, trt, tas, nsamp, phase), (sig_id, trt, tas, nsamp, phase), ...]
        s = [t for t in v if t[3] >= 2 and t[3] <= 6]   # keep tuples of v for which nsample between 2 and 6
        if not s:                                       # if no tuple is selected (the list is empty)
            s = v                                       # keep the original list of tuples
        sel = None
        max_tas = 0.
        for x in s:                                     # for each (filtered) tuple of v (sig_id, trt, tas, nsamp, phase)
            if not sel:                                 # First tuple 
                sel = (x[0], x[-1])                     # sel = (sig_id, phase) ; max_tas=tas
                max_tas = x[2]
            else:                                       # Other tuples
                if x[2] > max_tas:                      # Keep the tuple with BIGGEST tas
                    sel = (x[0], x[-1])
                    max_tas = x[2]
        return sel                                     # return (sig_id, phase) fo the tuple with biggest tas

    # Transform sigs in a dict (pert_id, cell_id) --> (sig_id, phase) fo the tuple with biggest tas
    sigs = dict((k, get_exemplar(v)) for k, v in sigs.items())

    cids = []

    # NS: Create  mini_sig_info_file.tsv
    with open(mini_sig_info_file, "w") as f:
        for k, v in sigs.items():             #(pert_id, cell_id) --> (sig_id, phase) fo the tuple with biggest tas
            if v[1] == 1:                     # from sig_info_i
                x = sig_info_i[v[0]]          # (pert_id, pert_type, cell_id, istouchstone, 1)
            else:
                x = sig_info_ii[v[0]]         # (pert_id, pert_type, cell_id, istouchstone, 2)

            # Writes :
            # sig_id, pert_id, pert_type, cell_id, istouchstone, phase) 
            f.write("%s\t%s\t%s\t%s\t%d\t%d\n" %(v[0], x[0], x[1], x[2], x[3], v[1]))
            cids += [(v[0], v[1])]             # (sig_id, phase)

    # The expression profiles (file of several GB)
    gtcx_i = os.path.join(map_files["GSE92742_Broad_LINCS_Level5_COMPZ"], "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx")

    for file_name in glob.glob(os.path.join(map_files["GSE70138_Broad_LINCS_Level5_COMPZ"], "*.gctx")):
        gtcx_ii = file_name

    genes_i = [genes[r[0]] for r in parse.parse(gtcx_i, cid=[[x[0] for x in cids if x[1] == 1][0]]).data_df.iterrows()]
    genes_ii = [genes[r[0]] for r in parse.parse(gtcx_ii, cid=[[x[0] for x in cids if x[1] == 2][0]]).data_df.iterrows()]  # Just to make sure.

    for cid in cids: # for each sign id  
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
    if np.abs(Qhi) > np.abs(Qlo):
        return Qhi
    else:
        return Qlo


def do_consensus(ik_matrices, consensus):

    inchikeys = [ik.split(".h5")[0] for ik in os.listdir(ik_matrices) if ik.endswith(".h5")]

    def consensus_signature(ik):
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "r") as hf:
            X = hf["X"][:]
        # It could be max, min...
        return [ int(get_summary(X[:, j])) for j in range(X.shape[1])]

    X = np.array([consensus_signature(ik) for ik in inchikeys])

    with h5py.File(consensus, "w") as hf:
        hf.create_dataset("inchikeys", data=np.array(inchikeys, DataSignature.string_dtype()))
        hf.create_dataset("X", data=X)

    return X, inchikeys


def process(X):

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

    # NS added: make skip everything if the final outputfile is already present
    if os.path.exists(args.output_file) and args.method == "fit":
        main._log.info("Preprocessed file {} is already present for {}, skipping the preprocessing step.".format(args.output_file,dataset_code))
        return


    mini_sig_info_file = os.path.join(args.models_path, 'mini_sig_info.tsv')  # NS file with 83637 records from L1000?

    dataset = Dataset.get(dataset_code) #NS D1.001 dataset object built to queryour sql database

    map_files = {}  # NS: will store datasource names of D1.00X and path to the corresponding files

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.

    # NS: 13 datasources in dataset_had_datasource table for the 2020 update
    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path       #Ns path is /aloy/scratch/sbnb-adm/CC/download/<datasource_name>

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

    readyfile = "agg_matrices.ready"

    # MATRIX AGGREGATION JOB
    if not os.path.exists(os.path.join(ik_matrices, readyfile)):

        if args.method == 'fit':
            main._log.info("Reading L1000")

            # NS Get molecules from Lincs
            inchikey_sigid, inchikey_inchi, siginfo = read_l1000(mini_sig_info_file, connectivitydir)
            # inchikey_sigid:   mol_inchikey -> mol_id (h5 filename without extension)
            # inchikey_inchi:   mol_inchikey -> mol_inchi
            # siginfo:          REP.A001_A375_24H:K17 -> mol_id

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

    if args.method == 'predict' and features is not None: # NS: added the first cond, other 'features' was unreferenced when not using 'predict'
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=int )
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
            # getting strings instead of bytes from the h5 file
            hf.create_dataset("features", data=np.array(orderwords, DataSignature.string_dtype()))

    if args.method == 'predict':
        #shutil.rmtree(mpath) # NS comment
        # shutil.rmtree(connectivitydir)
        pass

if __name__ == '__main__':
    main(sys.argv[1:])
