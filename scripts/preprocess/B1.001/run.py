import os
import sys
import argparse

import numpy as np
import networkx as nx
import collections
import h5py
import pickle

import xml.etree.ElementTree as ET

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo


# Variables

chembl_dbname = 'chembl'
graph_file = "graph.gpickle"
features_file = "prots.h5"
class_prot_file = "class_prot.pickl"
# Parse arguments
entry_point_full = "proteins"
entry_point_class = "classes"


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

# Functions


def decide(acts):
    m = np.mean(acts)
    if m > 0:
        return 1
    else:
        return -1


def parse_chembl(ACTS=None):

    if ACTS is None:
        ACTS = collections.defaultdict(list)

    dirs = {
        'DEGRADER': -1,
        'CROSS-LINKING AGENT': -1,
        'ANTISENSE INHIBITOR': -1,
        'SEQUESTRING AGENT': -1,
        'DISRUPTING AGENT': -1,
        'CHELATING AGENT': -1,
        'SUBSTRATE': 1,
        'AGONIST': 1,
        'STABILISER': 1,
        'BLOCKER': -1,
        'POSITIVE MODULATOR': 1,
        'PARTIAL AGONIST': 1,
        'NEGATIVE ALLOSTERIC MODULATOR': -1,
        'ACTIVATOR': 1,
        'INVERSE AGONIST': -1,
        'INHIBITOR': -1,
        'ANTAGONIST': -1,
        'POSITIVE ALLOSTERIC MODULATOR': 1
    }

    # Query ChEMBL

    R = psql.qstring('''

    SELECT md.chembl_id, cs.canonical_smiles, ts.accession, m.action_type

    FROM molecule_dictionary md, drug_mechanism m, target_components tc, component_sequences ts, compound_structures cs

    WHERE

    md.molregno = m.molregno AND
    m.tid = tc.tid AND
    ts.component_id = tc.component_id AND
    m.molregno = cs.molregno AND

    ts.accession IS NOT NULL AND
    m.action_type IS NOT NULL AND
    cs.canonical_smiles IS NOT NULL

    ''', chembl_dbname)

    # Read molrepo file

    chemblid_inchikey = {}
    inchikey_inchi = {}
    molrepos = Molrepo.get_by_molrepo_name("chembl")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        chemblid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    # Iterate over results

    for r in R:
        if r[0] not in chemblid_inchikey:
            continue
        inchikey = chemblid_inchikey[r[0]]
        uniprot_ac = r[2]
        if r[3] not in dirs:
            continue
        act = dirs[r[3]]
        ACTS[(inchikey, uniprot_ac, inchikey_inchi[inchikey])] += [act]

    ACTS = dict((k, decide(v)) for k, v in ACTS.iteritems())

    return ACTS


def parse_drugbank(ACTS=None, drugbank_xml=None):

    if ACTS is None:
        ACTS = collections.defaultdict(list)

    dirs = {
        'Inhibitor': -1,
        'acetylation': -1,
        'activator': +1,
        'agonist': +1,
        'antagonist': -1,
        'binder': -1,
        'binding': -1,
        'blocker': -1,
        'cofactor': +1,
        'inducer': +1,
        'inhibitor': -1,
        'inhibitor, competitive': -1,
        'inhibitory allosteric modulator': -1,
        'intercalation': -1,
        'inverse agonist': +1,
        'ligand': -1,
        'negative modulator': -1,
        'partial agonist': +1,
        'partial antagonist': -1,
        'positive allosteric modulator': +1,
        'positive modulator': +1,
        'potentiator': +1,
        'stimulator': -1,
        'suppressor': -1}

    # Parse the molrepo

    dbid_inchikey = {}
    inchikey_inchi = {}
    molrepos = Molrepo.get_by_molrepo_name("drugbank")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        dbid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    # Parse DrugBank

    prefix = "{http://www.drugbank.ca}"

    tree = ET.parse(drugbank_xml)

    root = tree.getroot()

    DB = {}

    for drug in root:

        # Drugbank ID

        db_id = None
        for child in drug.findall(prefix + "drugbank-id"):
            if "primary" in child.attrib:
                if child.attrib["primary"] == "true":
                    db_id = child.text

        if db_id not in dbid_inchikey:
            continue
        inchikey = dbid_inchikey[db_id]

        # Targets

        targets = collections.defaultdict(list)

        for targs in drug.findall(prefix + "targets"):
            for targ in targs.findall(prefix + "target"):

                # Actions

                actions = []
                for action in targ.findall(prefix + "actions"):
                    for child in action:
                        actions += [child.text]

                # Uniprot AC

                uniprot_ac = None
                prot = targ.find(prefix + "polypeptide")
                if not prot:
                    continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac:
                    continue

                targets[uniprot_ac] = actions

        if not targets:
            continue

        DB[inchikey] = targets

    # Save activities

    for inchikey, targs in DB.iteritems():
        for uniprot_ac, actions in targs.iteritems():
            if (inchikey, uniprot_ac, inchikey_inchi[inchikey]) in ACTS:
                continue
            d = []
            for action in actions:
                if action in dirs:
                    d += [dirs[action]]
            if not d:
                continue
            act = decide(d)
            ACTS[(inchikey, uniprot_ac, inchikey_inchi[inchikey])] = act

    return ACTS


def create_class_prot():

    R = psql.qstring(
        "SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for r in R:
        G.add_edge(r[1], r[0])  # The tree

    R = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R:
        class_prot[r[0]] += [r[1]]

    return class_prot, G


def put_hierarchy(ACTS, class_prot, G):

    classACTS = {}
    prots = set()

    for k, v in ACTS.items():
        classACTS[k] = v
        if k[1] not in class_prot:
            continue
        prots.add(k[1])
        path = set()
        for x in class_prot[k[1]]:
            p = nx.all_simple_paths(G, 0, x)
            for sp in p:
                path.update(sp)
        for p in path:
            classACTS[(k[0], "Class:%d" % p)] = v

    return classACTS, prots


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'B1.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_full

    if args.method == "fit":

        drugbank_xml = os.path.join(map_files["drugbank"], "full database.xml")

        main._log.info("Parsing ChEMBL")
        ACTS = parse_chembl()

        main._log.info("Parsing DrugBank")
        ACTS = parse_drugbank(ACTS, drugbank_xml)

        class_prot, G = create_class_prot()

        nx.write_gpickle(G, os.path.join(args.models_path, graph_file))

        with open(os.path.join(args.models_path, class_prot_file), 'wb') as fh:
            pickle.dump(class_prot, fh)

    if args.method == "predict":

        ACTS = {}

        prots = None

        if args.entry_point == entry_point_full:
            with h5py.File(os.path.join(args.models_path, features_file)) as hf:
                prots = set(hf["prots"][:])

        G = nx.read_gpickle(os.path.join(args.models_path, graph_file))

        class_prot = pickle.load(
            open(os.path.join(args.models_path, class_prot_file), 'rb'))

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if prots is not None and items[1] not in prots:
                    continue
                if len(items) < 3:
                    ACTS[(items[0], items[1])] = -1
                else:
                    ACTS[(items[0], items[1])] = items[2]

    if args.entry_point == entry_point_full:
        main._log.info("Putting target hierarchy")
        ACTS, prots = put_hierarchy(ACTS, class_prot, G)

    main._log.info("Saving raws")
    RAW = collections.defaultdict(list)
    for k, v in ACTS.items():
        RAW[k[0]] += [k[1] + "(%s)" % v]
    keys = []
    words = set()
    for k in sorted(RAW.keys()):
        keys.append(str(k))
        words.update(RAW[k])

    orderwords = list(words)
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in RAW[k]:
            raws[i][wordspos[word]] = 1

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))

    if args.method == "fit":
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("prots", data=np.array(list(prots)))


if __name__ == '__main__':
    main()
