import os
import sys
import h5py
import pickle
import logging
import argparse
import collections
import numpy as np
import networkx as nx
import xml.etree.ElementTree as ET

#from chemicalchecker.util import get_parser, save_output, features_file (Nico, obsolete)
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.core.preprocess import features_file

from chemicalchecker.util import psql
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.core.signature_data import DataSignature

# Variables
dataset_code = 'B1.001'
chembl_dbname = 'chembl'
graph_file = "graph.gpickle"
class_prot_file = "class_prot.pickl"
# Parse arguments
entry_point_full = "proteins"
entry_point_class = "classes"

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

    ACTS = dict((k, decide(v)) for k, v in ACTS.items())

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

    for inchikey, targs in DB.items():
        for uniprot_ac, actions in targs.items():
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
        if r[1] is not None:
            G.add_edge(r[1], r[0])  # The tree

    R = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R:
        class_prot[r[0]] += [r[1]]

    return class_prot, G


def put_hierarchy(ACTS, class_prot, G):

    classACTS = {}

    for k, v in ACTS.items():
        classACTS[k] = v
        if k[1] not in class_prot:
            continue
        path = set()
        for x in class_prot[k[1]]:
            p = nx.all_simple_paths(G, 0, x)
            for sp in p:
                path.update(sp)
        for p in path:
            classACTS[(k[0], "Class:%d" % p)] = v

    return classACTS


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):
    # Reading arguments and getting datasource
    args = Preprocess.get_parser().parse_args(args)
    main._log.debug("Running preprocess. Saving output to %s",
                    args.output_file)
    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path
    # decide entry point, if None use default
    if args.entry_point is None:
        args.entry_point = entry_point_full

    # main FIT section
    if args.method == "fit":

        # fetch ACTS from ChEMBL and DrugBank
        main._log.info("Parsing ChEMBL.")
        ACTS = parse_chembl()

        main._log.info("Parsing DrugBank.")
    
        file_path = map_files["drugbank"]
        if( os.path.isdir(file_path) ):
            fxml = ''
            for fs in os.listdir(file_path) :
                if( fs.endswith('.xml') ):
                    fxml = fs
            drugbank_xml = os.path.join(file_path, fxml)
            
        ACTS = parse_drugbank(ACTS, drugbank_xml)

        # generate protein class dictionary and graph
        class_prot, G = create_class_prot()

        # save them to disk
        nx.write_gpickle(G, os.path.join(args.models_path, graph_file))
        with open(os.path.join(args.models_path, class_prot_file), 'wb') as fh:
            pickle.dump(class_prot, fh)

        # features will be calculated later
        features = None

    # main PREDICT section
    if args.method == "predict":

        # fetch ACTS from input file
        ACTS = {}
        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if len(items) < 3:
                    v = -1
                else:
                    v = int(items[2])
                if (items[1] + "(" + str(v) + ")") not in features:
                    continue
                ACTS[(items[0], items[1])] = v

        # read protein class dictionary and graph
        G = nx.read_gpickle(os.path.join(args.models_path, graph_file))
        class_prot = pickle.load(
            open(os.path.join(args.models_path, class_prot_file), 'rb'))

        # load features (saved at FIT time)
        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

    # two entry point options, when protein are used put hierarchy
    if args.entry_point == entry_point_full:
        main._log.info("Putting target hierarchy.")
        ACTS = put_hierarchy(ACTS, class_prot, G)

    # save raw values
    main._log.info("Saving raws.")
    inchikey_raw = collections.defaultdict(list)
    features = set()
    for k, v in ACTS.items():
        feat = k[1] + "(" + str(v) + ")"
        inchikey_raw[k[0]] += [(feat, 1)]
        features.add(feat)
    features = list(features)

    Preprocess.save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features)


if __name__ == '__main__':
    main(sys.argv[1:])
