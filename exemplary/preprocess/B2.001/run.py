import os
import sys
import argparse
import networkx as nx
import collections
import h5py
import numpy as np

import xml.etree.ElementTree as ET


from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo


# Variables

chembl_dbname = 'chembl'

# Parse arguments


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


def put_hierarchy(ACTS):

    R = psql.qstring(
        "SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for r in R:
        G.add_edge(r[1], r[0])  # The tree

    R = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R:
        class_prot[r[0]] += [r[1]]

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
            classACTS.update([(k[0], "Class:%d" % p, k[2])])

    return classACTS


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
    return parser


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'B2.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    drugbank_xml = os.path.join(map_files["drugbank"], "full database.xml")

    main._log.info("Parsing ChEMBL")
    ACTS = parse_chembl()

    main._log.info("Parsing DrugBank")
    ACTS = parse_drugbank(ACTS, drugbank_xml)

    main._log.info("Putting target hierarchy")
    ACTS = put_hierarchy(ACTS)

    main._log.info("Saving raws")
    RAW = collections.defaultdict(list)
    for k in ACTS:
        RAW[k[0]] += [k[1]]
    keys = []
    raws = []
    for k in sorted(RAW.iterkeys()):
        raws.append(",".join(RAW[k]))
        keys.append(str(k))

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=np.array(raws))


if __name__ == '__main__':
    main()
