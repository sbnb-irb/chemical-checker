import os
import sys
import argparse

import numpy as np
import networkx as nx
import collections

import xml.etree.ElementTree as ET

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo


# Variables

chembl_dbname = 'chembl'
# Parse arguments


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
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

    classACTS = {}

    for k, v in ACTS.iteritems():
        classACTS[k] = v
        if k[1] not in class_prot:
            continue
        path = set()
        for x in class_prot[k[1]]:
            p = nx.all_simple_paths(G, 0, x)
            for sp in p:
                path.update(sp)
        for p in path:
            classACTS[(k[0], "Class:%d" % p, k[2])] = v

    return classACTS


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'B1.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running validation for dataset MOA. Saving output in " + args.output_file)

    drugbank_xml = os.path.join(map_files["drugbank"], "full database.xml")

    main._log.info("Parsing ChEMBL")
    ACTS = parse_chembl()

    main._log.info("Parsing DrugBank")
    ACTS = parse_drugbank(ACTS, drugbank_xml)

    main._log.info("Putting target hierarchy")
    ACTS = put_hierarchy(ACTS)

    main._log.info("Saving validation set")
    RAW = collections.defaultdict(list)
    for k, v in ACTS.items():
        RAW[k[0]] += [k[1] + "(%s)" % v]

    d = {}
    for k, v in RAW.items():
        d[k] = set([x for x in v if "Class" not in x])

    keys = sorted(d.keys())

    f = open(args.output_file, "w")
    for i in range(len(keys) - 1):
        for j in range(i + 1, len(keys)):
            if keys[i].split("-")[0] == keys[j].split("-")[0]:
                continue
            com = len(d[keys[i]].intersection(d[keys[j]]))
            if com > 0:
                v = 1
            else:
                v = 0
            f.write("%s\t%s\t%d\n" % (keys[i], keys[j], v))
    f.close()


if __name__ == '__main__':
    main()
