import os
import sys
import argparse
import networkx as nx
import collections
import h5py
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import logging

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo
from chemicalchecker.core.preprocess import Preprocess

# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
chembl_dbname = 'chembl'
graph_file = "graph.gpickle"
features_file = "features.h5"
class_prot_file = "class_prot.pickl"
entry_point_full = "proteins"
entry_point_class = "classes"


def parse_chembl(ACTS=None):

    if ACTS is None:
        ACTS = set()

    # Query ChEMBL

    R = psql.qstring('''

    SELECT md.chembl_id, cs.accession

    FROM molecule_dictionary md, compound_records cr, metabolism m, target_components tc, component_sequences cs

    WHERE

    md.molregno = cr.molregno AND
    cr.record_id = m.drug_record_id AND
    m.enzyme_tid = tc.tid AND
    tc.component_id = cs.component_id AND
    cs.accession IS NOT NULL

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
        uniprot_ac = r[1]
        ACTS.update([(inchikey, uniprot_ac, inchikey_inchi[inchikey])])

    return ACTS


def parse_drugbank(ACTS=None, drugbank_xml=None):

    if ACTS is None:
        ACTS = set()

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

        # Enzymes, transporters, carriers

        for ps in drug.findall(prefix + "enzymes"):
            for p in ps.findall(prefix + "enzyme"):

                # Uniprot AC

                uniprot_ac = None
                prot = p.find(prefix + "polypeptide")
                if not prot:
                    continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac:
                    continue

                ACTS.update([(inchikey, uniprot_ac, inchikey_inchi[inchikey])])

        for ps in drug.findall(prefix + "transporters"):
            for p in ps.findall(prefix + "transporter"):

                # Uniprot AC

                uniprot_ac = None
                prot = p.find(prefix + "polypeptide")
                if not prot:
                    continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac:
                    continue

                ACTS.update([(inchikey, uniprot_ac, inchikey_inchi[inchikey])])

        for ps in drug.findall(prefix + "carriers"):
            for p in ps.findall(prefix + "carrier"):

                # Uniprot AC

                uniprot_ac = None
                prot = p.find(prefix + "polypeptide")
                if not prot:
                    continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac:
                    continue

                ACTS.update([(inchikey, uniprot_ac, inchikey_inchi[inchikey])])

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

    classACTS = set()

    for k in ACTS:
        classACTS.update([k])
        if k[1] not in class_prot:
            continue
        path = set()
        for x in class_prot[k[1]]:
            p = nx.all_simple_paths(G, 0, x)
            for sp in p:
                path.update(sp)
        for p in path:
            classACTS.update([(k[0], "Class:%d" % p)])

    return classACTS


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):

    args = Preprocess.get_parser().parse_args(args)

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_full

    features = None

    if args.method == "fit":
    
        file_path = map_files["drugbank"]
        if( os.path.isdir(file_path) ):
            fxml = ''
            for fs in os.listdir(file_path) :
                if( fs.endswith('.xml') ):
                    fxml = fs
            drugbank_xml = os.path.join(file_path, fxml)

        #drugbank_xml = os.path.join(map_files["drugbank"], "full database.xml")

        main._log.info("Parsing ChEMBL")
        ACTS = parse_chembl()

        main._log.info("Parsing DrugBank")
        ACTS = parse_drugbank(ACTS, drugbank_xml)

        class_prot, G = create_class_prot()

        nx.write_gpickle(G, os.path.join(args.models_path, graph_file))

        with open(os.path.join(args.models_path, class_prot_file), 'wb') as fh:
            pickle.dump(class_prot, fh)

    if args.method == "predict":

        ACTS = []

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        G = nx.read_gpickle(os.path.join(args.models_path, graph_file))

        class_prot = pickle.load(
            open(os.path.join(args.models_path, class_prot_file), 'rb'))

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                ACTS.append((items[0], items[1]))

    if args.entry_point == entry_point_full:
        main._log.info("Putting target hierarchy")
        ACTS = put_hierarchy(ACTS, class_prot, G)

    main._log.info("Saving raws")
    RAW = collections.defaultdict(list)
    for k in ACTS:
        RAW[k[0]] += [k[1]]

    Preprocess.save_output(args.output_file, RAW, args.method,
                args.models_path, True, features)


if __name__ == '__main__':
    main(sys.argv[1:])
