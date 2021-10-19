import sys
import os
import argparse
import networkx as nx
import collections
import h5py
import numpy as np
import pickle
import logging
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo
from chemicalchecker.core.preprocess import Preprocess

dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
chembl_dbname = 'chembl'
graph_file = "graph.gpickle"
features_file = "features.h5"
class_prot_file = "class_prot.pickl"
class_path_file = "class_path.pickl"
entry_point_full = "proteins"
entry_point_class = "classes"


def parse_chembl():

    # Read molrepo

    molrepos = Molrepo.get_by_molrepo_name("chembl")
    chemblid_inchikey = {}
    inchikey_inchi = {}
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        chemblid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    def is_active(r):
        if r[3] is not None and r[3] >= 5:
            return True
        if r[2] == "Active" or r[2] == "active":
            return True
        return False

    # Query

    cur = psql.qstring('''
    SELECT md.chembl_id, cseq.accession, act.activity_comment, act.pchembl_value
    FROM molecule_dictionary md, activities act, assays ass, component_sequences cseq, target_components t
    WHERE
    (ass.src_id != 1 OR (ass.src_id = 1 AND ass.assay_type = 'F')) AND
    md.molregno = act.molregno AND
    act.assay_id = ass.assay_id AND
    ass.tid = t.tid AND
    t.component_id = cseq.component_id AND
    cseq.accession IS NOT NULL
    ''', chembl_dbname)

    ACTS = []
    act = 0
    notid = 0
    for r in cur:
        if r[0] not in chemblid_inchikey:
            notid += 1
            continue
        if not is_active(r):
            act += 1
            continue
        ACTS += [(chemblid_inchikey[r[0]], r[1])]
    # Use ChEMBL hierarchy

    S = psql.qstring(
        "SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for s in S:
        if s[1] is not None:
            G.add_edge(s[1], s[0])  # The tree

    S = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for s in S:
        class_prot[s[0]] += [s[1]]

    classes = set([x for k, v in class_prot.items() for x in v])
    class_paths = collections.defaultdict(set)
    for c in classes:
        path = set()
        p = nx.all_simple_paths(G, 0, c)
        for sp in p:
            path.update(sp)
        class_paths[c] = path

    return ACTS, class_prot, class_paths, G


def format_data(ACTS, class_prot, class_paths, G):

    T = set()
    prots = set()
    for r in ACTS:
        T.update([(r[0], r[1])])
        if r[1] not in class_prot:
            continue
        prots.add(r[1])
        path = set()
        for c in class_prot[r[1]]:
            path.update(class_paths[c])
        for p in path:
            T.update([(r[0], "Class:%d" % p)])

    return T, prots


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):

    args = Preprocess.get_parser().parse_args(args)

    dataset = Dataset.get(dataset_code)

    map_files = {}

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.
    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_full

    features = None

    if args.method == "fit":

        main._log.info("Parsing ChEMBL")
        ACTS, class_prot, class_paths, G = parse_chembl()

        nx.write_gpickle(G, os.path.join(args.models_path, graph_file))

        with open(os.path.join(args.models_path, class_prot_file), 'wb') as fh:
            pickle.dump(class_prot, fh)
        with open(os.path.join(args.models_path, class_path_file), 'wb') as fh:
            pickle.dump(class_paths, fh)

    if args.method == "predict":

        ACTS = []

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        G = nx.read_gpickle(os.path.join(args.models_path, graph_file))

        class_prot = pickle.load(
            open(os.path.join(args.models_path, class_prot_file), 'rb'))

        class_paths = pickle.load(
            open(os.path.join(args.models_path, class_path_file), 'rb'))

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                ACTS.append((items[0], items[1]))

    T, prots = format_data(ACTS, class_prot, class_paths, G)
    main._log.info("Saving raw data")
    inchikey_raw = collections.defaultdict(list)
    for t in T:
        inchikey_raw[t[0]] += [t[1]]

    Preprocess.save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
