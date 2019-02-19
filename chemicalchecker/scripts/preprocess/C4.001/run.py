import sys
import argparse
import os
import networkx as nx
import collections
import h5py
import numpy as np
import shutil
import math
sys.path.append(
    "/aloy/home/oguitart/projects/source/chemical_checker/chemicalchecker")
from chemicalchecker.util import logged, Config
from chemicalchecker.database import Dataset
from chemicalchecker.util import HPC
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo

# Variables

chembl_dbname = 'chembl'
entry_point_full = "proteins"
entry_point_process = "processes"
features_file = "features.h5"


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


def fetch_binding(any_human, ACTS, prot_allgos_dir):

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

    any_human.clear()

    def get_allgos(uniprot_ac):
        if not os.path.exists("%s/%s.tsv" % (prot_allgos_dir, uniprot_ac)):
            return None
        gos = set()
        f = open("%s/%s.tsv" % (prot_allgos_dir, uniprot_ac), "r")
        for l in f:
            go = l.rstrip("\n")
            gos.update([go])
        f.close()
        return gos

    # Get prot and go

    prots = set([x[1] for x in ACTS_new.keys()])

    class_prot = {}
    for prot in prots:
        gos = get_allgos(prot)
        if not gos:
            continue
        class_prot[prot] = gos

    GOS = collections.defaultdict(list)
    for k, v in ACTS_new.items():
        if k[1] not in class_prot:
            continue
        for go in class_prot[k[1]]:
            GOS[(k[0], go)] += [v]
    GOS = dict((k, np.max(v)) for k, v in GOS.items())

    return GOS

# Parse arguments


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'B4.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    bindingdb_file = os.path.join(
        map_files["bindingdb"], "BindingDB_All.tsv")

    if args.entry_point is None:
        args.entry_point = entry_point_full

    features = None

    dataset_code = 'C4.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

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
    goa_human = os.path.join(
        map_files["goa_human"], "goa_human.gaf")
    gobasic = os.path.join(
        map_files["gobasic"], "go-basic.obo")

    WD = os.path.dirname(os.path.realpath(__file__))

    prot2allgos_script = WD + "/prot2allgos.py"
    prot_allgos_dir = os.path.join(args.models_path, "prot_allgos")

    if args.method == "fit":

        main._log.info(" Parsing ChEMBL")

        ACTS = parse_chembl()

        main._log.info(" Parsing BindingDB")

        ACTS = parse_bindingdb(ACTS, bindingdb_file)

        main._log.info(" Processing activity and assigning target classes")

        ACTS = process_activity_according_to_pharos(ACTS)

        G = nx.DiGraph()

        f = open(gobasic, "r")
        S = f.read().split("[Term]")[1:]
        f.close()

        for chunk in S:
            s = chunk.split("\n")
            is_bp = False
            for l in s:
                if l[:3] == "id:":
                    child = l.split("id: ")[1]
                if l[:29] == "namespace: biological_process":
                    is_bp = True
                if l[:5] == "is_a:" and is_bp:
                    parent = l.split("is_a: ")[1].split(" !")[0]
                    if parent == "regulates":
                        continue
                    G.add_edge(parent, child)

        f = open(os.path.join(args.models_path, "bp.tsv"), "w")
        for e in G.edges():
            f.write("%s\t%s\n" % (e[0], e[1]))
        f.close()

        uniprots = set()
        f = open(goa_human, "r")
        for l in f:
            if l[0] == "!":
                continue
            l = l.split("\t")
            if l[8] != "P":
                continue
            uniprots.update([l[1]])
        f.close()

        job_path = os.path.join(Config().PATH.CC_TMP, "job_bpan")

        if not os.path.exists(prot_allgos_dir):
            os.makedirs(prot_allgos_dir, 0o775)

        status = ''

        readyfile = "prots.ready"
        if os.path.exists(os.path.join(prot_allgos_dir, readyfile)):
            status = "ready"

        if status != "ready":

            if os.path.isdir(job_path):
                shutil.rmtree(job_path)
            os.mkdir(job_path)

            params = {}

            params["num_jobs"] = len(uniprots) / 20
            params["jobdir"] = job_path
            params["job_name"] = "CC_C4_bpan"
            params["elements"] = list(uniprots)
            params["memory"] = 4
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "singularity exec {} python {} <TASK_ID> <FILE> {} {} {}".format(
                singularity_image, prot2allgos_script, args.models_path, prot_allgos_dir, goa_human)
            # submit jobs
            cluster = HPC(Config())
            cluster.submitMultiJob(command, **params)

            if cluster.status() == 'error':
                main._log.error(
                    "Bpannos job produced some errors. The preprocess script can't continue")
                sys.exit(1)
            with open(os.path.join(prot_allgos_dir, readyfile), "w") as f:
                f.write("")

    if args.method == "predict":

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        ACTS = {}

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if args.entry_point == entry_point_process and items[1] not in features:
                    continue
                if len(items) < 3:
                    ACTS[(items[0], items[1])] = 1
                else:
                    ACTS[(items[0], items[1])] = int(items[2])

    if args.entry_point == entry_point_full:

        main._log.info("Reading human MetaPhors")
        any_human = human_metaphors(id_conversion, file_9606, human_proteome)

        main._log.info("Fetching binding data")
        GOS = fetch_binding(any_human, ACTS, prot_allgos_dir)

    else:
        GOS = ACTS

    inchikey_raw = collections.defaultdict(list)
    for k, v in GOS.items():
        if features is not None and k[1] not in features:
            continue
        inchikey_raw[k[0]] += [(k[1], v)]

    main._log.info("Saving raw data")

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
