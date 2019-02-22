import os
import sys
import argparse
import networkx as nx
import collections
import h5py
import math
import numpy as np
import shutil
from sklearn.preprocessing import normalize

from chemicalchecker.util import logged, Config
from chemicalchecker.util import HPC
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo

chembl_dbname = 'chembl'
features_file = "features.h5"
entry_point_full = "proteins"
entry_point_neigh = "protein_neighbors"
entry_point_neigh_networks = "protein_neighbors_network"

networks = ["string_net", "inbiomap", "ppidb", "pathwaycommons", "recon"]

WD = os.path.dirname(os.path.realpath(__file__))
cc_config = os.environ['CC_CONFIG']

cpu = 10

script_lines = [
    "import sys, os",
    "import pickle",
    "sys.path.append('" + WD + "')",
    "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
    "from ppidb import ppidb",
    "from inbiomap import inbiomap",
    "from pathwaycommons import pathwaycommons",
    "from string_net import string_net",
    "from recon import recon",
    "task_id = sys.argv[1]",  # <TASK_ID>
    "filename = sys.argv[2]",  # <FILE>
    "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
    "net = str(inputs[task_id][0][0])",  # elements for current job
    "path = inputs[task_id][0][1]",  # elements for current job
    "dbname = inputs[task_id][0][2]",  # elements for current job
    'net_class = eval(net)(path,dbname,' + str(cpu) + ')',
    'net_class.run()',
    "print('JOB DONE')"
]


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
    parser.add_argument('-ep', '--entry_point', type=str,
                        required=False, default=None, help='The predict entry point')
    return parser
# Variables


def parse_chembl(ACTS=None):

    if ACTS is None:
        ACTS = collections.defaultdict(list)

    # Read molrepo file

    chemblid_inchikey = {}
    inchikey_inchi = {}
    molrepos = Molrepo.get_by_molrepo_name("chembl")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        chemblid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    # Query ChEMBL

    R = psql.qstring('''
    SELECT md.chembl_id, cseq.accession, act.pchembl_value

    FROM molecule_dictionary md, activities act, assays ass, component_sequences cseq, target_components t, target_dictionary td

    WHERE

    act.molregno = md.molregno AND
    act.assay_id = ass.assay_id AND
    act.standard_relation = '=' AND
    act.standard_flag = 1 AND
    ass.assay_type = 'B' AND
    ass.tid = t.tid AND
    ass.tid = td.tid AND
    td.target_type = 'SINGLE PROTEIN' AND
    t.component_id = cseq.component_id AND
    cseq.accession IS NOT NULL AND
    act.pchembl_value >= 5''', chembl_dbname)
    ACTS = collections.defaultdict(list)
    for r in R:
        chemblid = r[0]
        if chemblid not in chemblid_inchikey:
            continue
        ACTS[(chemblid_inchikey[chemblid], r[1], inchikey_inchi[
              chemblid_inchikey[chemblid]])] += [r[2]]

    return ACTS


def parse_bindingdb(ACTS=None, bindingdb_file=None):

    if ACTS is None:
        ACTS = collections.defaultdict(list)

    def pchemblize(act):
        try:
            act = act / 1e9
            return -math.log10(act)
        except:
            return None

    def activity(ki, ic50, kd, ec50):
        def to_float(s):
            s = s.replace("<", "")
            if s == '' or ">" in s or ">" in s:
                return []
            else:
                return [float(s)]
        acts = []
        acts += to_float(ki)
        acts += to_float(ic50)
        acts += to_float(kd)
        acts += to_float(ec50)
        if acts:
            pchembl = pchemblize(np.min(acts))
            if pchembl < 5:
                return None
            return pchembl
        return None

    # Molrepo

    bdlig_inchikey = {}
    inchikey_inchi = {}
    molrepos = Molrepo.get_by_molrepo_name("bindingdb")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        bdlig_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    # Read header of BindingDB

    f = open(bindingdb_file, "r")

    header = f.next()
    header = header.rstrip("\n").split("\t")
    bdlig_idx = header.index("Ligand InChI Key")
    smiles_idx = header.index("Ligand SMILES")
    ki_idx = header.index("Ki (nM)")
    ic50_idx = header.index("IC50 (nM)")
    kd_idx = header.index("Kd (nM)")
    ec50_idx = header.index("EC50 (nM)")
    uniprot_ac_idx = header.index(
        "UniProt (SwissProt) Primary ID of Target Chain")
    nchains_idx = header.index(
        "Number of Protein Chains in Target (>1 implies a multichain complex)")

    f.close()

    # Now read activity

    # Now get the activity.

    f = open(bindingdb_file, "r")
    f.next()
    for l in f:

        l = l.rstrip("\n").split("\t")
        if len(l) < nchains_idx:
            continue
        if l[nchains_idx] == '':
            continue
        nchains = int(l[nchains_idx])
        if nchains != 1:
            continue
        bdlig = l[bdlig_idx]
        if bdlig not in bdlig_inchikey:
            continue
        inchikey = bdlig_inchikey[bdlig]
        ki = l[ki_idx]
        ic50 = l[ic50_idx]
        kd = l[kd_idx]
        ec50 = l[ec50_idx]
        act = activity(ki, ic50, kd, ec50)
        if not act:
            continue
        uniprot_ac = l[uniprot_ac_idx]
        if not uniprot_ac:
            continue
        for p in uniprot_ac.split(","):
            ACTS[(inchikey, p, inchikey_inchi[inchikey])] += [act]
    f.close()

    return ACTS


def process_activity_according_to_pharos(ACTS):

    R = psql.qstring(
        "SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    kinase_idx = [r[0] for r in R if r[2] == 'Kinase']
    gpcr_idx = [r[0] for r in R if r[2] == 'Family A G protein-coupled receptor'
                or r[2] == 'Family B G protein-coupled receptor'
                or r[2] == 'Family C G protein-coupled receptor'
                or r[2] == 'Frizzled family G protein-coupled receptor'
                or r[2] == 'Taste family G protein-coupled receptor']
    nuclear_idx = [r[0] for r in R if r[2] == 'Nuclear receptor']
    ionchannel_idx = [r[0] for r in R if r[2] == 'Ion channel']

    G = nx.DiGraph()

    for r in R:
        G.add_edge(r[1], r[0])  # The tree

    kinase_idx = set([x for w in kinase_idx for k, v in nx.dfs_successors(
        G, w).iteritems() for x in v] + kinase_idx)
    gpcr_idx = set([x for w in gpcr_idx for k, v in nx.dfs_successors(
        G, w).iteritems() for x in v] + gpcr_idx)
    nuclear_idx = set([x for w in nuclear_idx for k, v in nx.dfs_successors(
        G, w).iteritems() for x in v] + nuclear_idx)
    ionchannel_idx = set([x for w in ionchannel_idx for k, v in nx.dfs_successors(
        G, w).iteritems() for x in v] + ionchannel_idx)

    R = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R:
        class_prot[r[0]] += [r[1]]

    # According to Pharos

    cuts = {
        'kinase': -math.log10(30e-9),
        'gpcr': -math.log10(100e-9),
        'nuclear': -math.log10(100e-9),
        'ionchannel': -math.log10(10e-6),
        'other': -math.log10(1e-6)
    }

    protein_cutoffs = collections.defaultdict(list)

    for k, v in class_prot.iteritems():
        for idx in v:
            if idx in ionchannel_idx:
                protein_cutoffs[k] += [cuts['ionchannel']]
            elif idx in nuclear_idx:
                protein_cutoffs[k] += [cuts['nuclear']]
            elif idx in gpcr_idx:
                protein_cutoffs[k] += [cuts['gpcr']]
            elif idx in kinase_idx:
                protein_cutoffs[k] += [cuts['kinase']]
            else:
                protein_cutoffs[k] += [cuts['other']]

    protein_cutoffs = dict((k, np.min(v))
                           for k, v in protein_cutoffs.iteritems())

    ACTS = dict((k, np.max(v)) for k, v in ACTS.iteritems())

    R = psql.qstring(
        "SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for r in R:
        G.add_edge(r[1], r[0])  # The tree

    R = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R:
        class_prot[r[0]] += [r[1]]

    classes = set([c for k, v in class_prot.iteritems() for c in v])
    class_path = collections.defaultdict(set)
    for c in classes:
        path = set()
        for sp in nx.all_simple_paths(G, 0, c):
            path.update(sp)
        class_path[c] = path

    classACTS = collections.defaultdict(list)

    for k, v in ACTS.iteritems():
        if k[1] in protein_cutoffs:
            cut = protein_cutoffs[k[1]]
        else:
            cut = cuts['other']
        if v < cut:
            if v < (cut - 1):
                continue
            V = 1
        else:
            V = 2
        classACTS[k] += [V]
        if k[1] not in class_prot:
            continue
        for c in class_prot[k[1]]:
            for p in class_path[c]:
                classACTS[(k[0], "Class:%d" % p, k[2])] += [V]

    classACTS = dict((k, np.max(v)) for k, v in classACTS.iteritems())

    return classACTS


def human_metaphors(id_conversion, file_9606, human_proteome):

    metaphorsid_uniprot = collections.defaultdict(set)
    f = open(id_conversion, "r")
    f.next()
    for l in f:
        l = l.rstrip("\n").split("\t")
        if l[1] == "SwissProt" or l[1] == "TrEMBL":
            metaphorsid_uniprot[l[2]].update([l[0]])
    f.close()

    any_human = collections.defaultdict(set)
    f = open(file_9606, "r")
    f.next()
    for l in f:
        l = l.rstrip("\n").split("\t")
        if l[3] not in metaphorsid_uniprot:
            continue
        if l[1] not in metaphorsid_uniprot:
            continue
        for po in metaphorsid_uniprot[l[3]]:
            for ph in metaphorsid_uniprot[l[1]]:
                any_human[po].update([ph])
                any_human[ph].update([ph])
    f.close()

    f = open(human_proteome, "r")
    f.next()
    for l in f:
        p = l.split("\t")[0]
        any_human[p].update([p])
    f.close()

    return any_human


def fetch_binding(any_human, ACTS):

    ACTS_new = collections.defaultdict(list)

    for k, v in ACTS.items():
        uniprot_ac = k[1]
        act = v

        if uniprot_ac not in any_human:
            continue
        hps = any_human[uniprot_ac]

        for hp in hps:
            ACTS_new[(k[0], hp)] += [act]
    ACTS_new = dict((k, np.max(v)) for k, v in ACTS_new.items())

    return ACTS_new

# Hotnet-specific functions


class Sm:

    def __init__(self, A, names):
        self.A = A
        self.names = names

    def get_profile(self, n):
        if n not in self.names:
            return None
        return self.A[:, self.names.index(n)]


def load_matrix(net_folder):
    f = open(net_folder + "/idx2node.tsv", )
    names = [l.rstrip("\n").split("\t")[1] for l in f]
    f.close()
    f = h5py.File(net_folder + "/similarity_matrix.h5")
    A = f['PPR'].value
    f.close()
    return Sm(A, names)


def network_impact(sm, node_scores):
    P, S = [], []
    for n, s in node_scores.iteritems():
        prof = sm.get_profile(sm, n)
        if not prof:
            continue
        P += [prof]
        S += [s]
    P = np.array(P)


def scale_by_non_diagonal_max(A):
    S = A[:]
    np.fill_diagonal(S, 0.)
    for j in range(A.shape[1]):
        S[j, j] = np.max(S[:, j])
    S = normalize(S, norm="max", axis=0)
    return S


def read_hotnet_output(outdir, ACTS):

    prots = set([k[1] for k, v in ACTS.items()])

    protein_profiles = collections.defaultdict(list)

    for network in networks:

        # Loading network
        sm = load_matrix("%s/%s" % (outdir, network))
        sm.A = scale_by_non_diagonal_max(sm.A)
        myprots = prots.intersection(sm.names)

        # We use 'string_net' because the class needs to be called like that
        # But when creating the data we change to just 'string'
        if network == 'string_net':
            network = 'string'

        # Get the protein profiles
        for prot in myprots:
            P = []
            prof = sm.get_profile(prot)
            for i in xrange(len(prof)):
                w = int(prof[i] * 10)  # Convert weights to integers
                if w == 0:
                    continue
                P += [(network + "_" + sm.names[i], w)]
            protein_profiles[prot] += P

    D = collections.defaultdict(list)
    for k, v in ACTS.items():
        if k[1] not in protein_profiles:
            continue
        for p in protein_profiles[k[1]]:
            D[k[0]] += [(p[0], p[1] * v)]

    return D


# Parse arguments


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    features = None

    if args.entry_point is None:
        args.entry_point = entry_point_full

    if args.method == "fit":

        job_path = os.path.join(Config().PATH.CC_TMP, "job_nets")

        if os.path.isdir(job_path):
            shutil.rmtree(job_path)
        os.mkdir(job_path)

        script_name = os.path.join(job_path, 'prepare_networks.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters

        tasks = []

        for net in networks:

            readyfile = net + ".ready"
            if not os.path.exists(os.path.join(args.models_path, net, readyfile)):
                tasks.append(
                    (net, os.path.join(args.models_path, net), "2019_01"))
                if not os.path.exists(os.path.join(args.models_path, net)):
                    os.makedirs(os.path.join(args.models_path, net), 0o775)

        if len(tasks) > 0:
            params = {}
            params["num_jobs"] = len(tasks)
            params["jobdir"] = job_path
            params["job_name"] = "CC_C5_nets"
            params["elements"] = tasks
            params["wait"] = True
            params["memory"] = 2 * cpu
            params["cpu"] = cpu
            # job command
            singularity_image = '/aloy/home/oguitart/sing-images/cc.simg'
            command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
                singularity_image, script_name)
            # submit jobs
            cluster = HPC(Config())
            cluster.submitMultiJob(command, **params)

            if cluster.status() == 'error':
                main._log.error(
                    "Networks job produced some errors. The preprocess script can't continue")
                sys.exit(1)

        # os.path.dirname(os.path.abspath(__file__))[-6:]
        dataset_code = 'B4.001'
        dataset = Dataset.get(dataset_code)
        map_files = {}

        for ds in dataset.datasources:
            map_files[ds.name] = ds.data_path

        bindingdb_file = os.path.join(
            map_files["bindingdb"], "BindingDB_All.tsv")

        main._log.info(" Parsing ChEMBL")
        ACTS = parse_chembl()
        main._log.info(" Parsing BindingDB")
        ACTS = parse_bindingdb(ACTS, bindingdb_file)
        main._log.info(" Processing activity and assigning target classes")
        ACTS = process_activity_according_to_pharos(ACTS)

    if args.method == "predict":

        ACTS = collections.defaultdict(list)

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        default_value = 10

        if args.entry_point == entry_point_full:
            default_value = 1

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if args.entry_point == entry_point_neigh_networks:
                    if items[1] not in features:
                        continue
                    if len(items) < 3:
                        ACTS[items[0]] += [(items[1], default_value)]
                    else:
                        ACTS[items[0]] += [(items[1], int(items[2]))]
                else:
                    if len(items) < 3:
                        ACTS[(items[0], items[1])] = default_value
                    else:
                        ACTS[(items[0], items[1])] = int(items[2])

    if args.entry_point == entry_point_full:

        # os.path.dirname(os.path.abspath(__file__))[-6:]
        dataset_code = 'C5.001'
        dataset = Dataset.get(dataset_code)
        map_files = {}
        for ds in dataset.datasources:
            map_files[ds.name] = ds.data_path

        main._log.debug(
            "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

        id_conversion = os.path.join(
            map_files["metaphors_id_conversion"], "id_conversion.txt")
        file_9606 = os.path.join(map_files["metaphors_9606"], "9606.txt")
        human_proteome = os.path.join(
            map_files["human_proteome"], "human_proteome.tab")

        main._log.info("Human MetPhors")
        any_human = human_metaphors(id_conversion, file_9606, human_proteome)

        main._log.info("Fetch activities")
        ACTS = fetch_binding(any_human, ACTS)
        main._log.info("Reading HotNet output")
        inchikey_data = read_hotnet_output(args.models_path, ACTS)
        if features is not None:
            inchikey_raw = collections.defaultdict(list)
            for k, v in inchikey_data.items():
                for val in v:
                    if val[0] not in features:
                        continue
                    inchikey_raw[k] += [val]
        else:
            inchikey_raw = inchikey_data

    if args.entry_point == entry_point_neigh:

        inchikey_raw = collections.defaultdict(list)

        for network in networks:

            if network == 'string_net':
                network = 'string'

            for k, v in ACTS.items():
                item = network + "_" + k[1]
                if item not in features:
                    continue
                inchikey_raw[k[0]] += [(item, v)]

    if args.entry_point == entry_point_neigh_networks:
        inchikey_raw = ACTS

    main._log.info("Saving raws")

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
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))

    if args.method == "fit":
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=np.array(orderwords))

if __name__ == '__main__':
    main()
