import os
import sys
import argparse
import numpy as np
import networkx as nx
import collections
import h5py
import math
import pickle


from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo
# Variables

chembl_dbname = 'chembl'
graph_file = "graph.gpickle"
features_file = "features.h5"
class_prot_file = "class_prot.pickl"
class_path_file = "class_path.pickl"
class_prot_cutoff_file = "prot_cutoff.pickl"
entry_point_full = "proteins"
entry_point_class = "classes"

cuts = {
    'kinase': -math.log10(30e-9),
    'gpcr': -math.log10(100e-9),
    'nuclear': -math.log10(100e-9),
    'ionchannel': -math.log10(10e-6),
    'other': -math.log10(1e-6)
}


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


# Parse arguments


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


def create_class_prot():

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

    protein_cutoffs = collections.defaultdict(list)

    for k, v in class_prot.items():
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
                           for k, v in protein_cutoffs.items())

    R = psql.qstring(
        "SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for r in R:
        G.add_edge(r[1], r[0])  # The tree

    R = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R:
        class_prot[r[0]] += [r[1]]

    classes = set([c for k, v in class_prot.items() for c in v])
    class_path = collections.defaultdict(set)
    for c in classes:
        path = set()
        for sp in nx.all_simple_paths(G, 0, c):
            path.update(sp)
        class_path[c] = path

    return class_prot, protein_cutoffs, class_path, G


def process_activity_according_to_pharos(ACTS, class_prot, protein_cutoffs, class_path, G):

    ACTS = dict((k, np.max(v)) for k, v in ACTS.items())

    classACTS = collections.defaultdict(list)
    prots = set()

    for k, v in ACTS.items():
        if protein_cutoffs is not None:
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
        else:
            V = v
        classACTS[k] += [V]
        if k[1] not in class_prot:
            continue
        prots.add(k[1])
        for c in class_prot[k[1]]:
            for p in class_path[c]:
                classACTS[(k[0], "Class:%d" % p)] += [V]

    classACTS = dict((k, np.max(v)) for k, v in classACTS.items())

    return classACTS, prots


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'B4.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_full

    features = None

    if args.method == "fit":

        bindingdb_file = os.path.join(
            map_files["bindingdb"], "BindingDB_All.tsv")

        main._log.info(" Parsing ChEMBL")

        ACTS = parse_chembl()

        main._log.info(" Parsing BindingDB")

        ACTS = parse_bindingdb(ACTS, bindingdb_file)

        class_prot, protein_cutoffs, class_path, G = create_class_prot()

        nx.write_gpickle(G, os.path.join(args.models_path, graph_file))

        with open(os.path.join(args.models_path, class_prot_file), 'wb') as fh:
            pickle.dump(class_prot, fh)
        with open(os.path.join(args.models_path, class_path_file), 'wb') as fh:
            pickle.dump(class_path, fh)

    if args.method == "predict":

        ACTS = {}

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        G = nx.read_gpickle(os.path.join(args.models_path, graph_file))

        class_prot = pickle.load(
            open(os.path.join(args.models_path, class_prot_file), 'rb'))

        class_path = pickle.load(
            open(os.path.join(args.models_path, class_path_file), 'rb'))

        protein_cutoffs = None

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                if len(items) < 3:
                    ACTS[(items[0], items[1])] = 1
                else:
                    ACTS[(items[0], items[1])] = int(items[2])

    if args.entry_point == entry_point_full:

        main._log.info(" Processing activity and assigning target classes")

        ACTS, prots = process_activity_according_to_pharos(
            ACTS, class_prot, protein_cutoffs, class_path, G)

    main._log.info("Saving raws")
    inchikey_raw = collections.defaultdict(list)
    for k, v in ACTS.items():
        inchikey_raw[k[0]] += [(k[1], v)]

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
