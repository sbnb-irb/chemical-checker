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


def parse_kegg(br_file=None):

    inchikey_atc = collections.defaultdict(set)

    # Read molrepo
    kegg_inchikey = {}
    inchikey_inchi = {}
    molrepos = Molrepo.get_by_molrepo_name("kegg")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        kegg_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    # Read drug KEGG branch
    with open(br_file, "r") as f:
        for l in f:
            if l[0] == "E":
                atc = l.split()[1]
            if l[0] == "F":
                drug = l.split()[1]
                if drug not in kegg_inchikey:
                    continue
                inchikey_atc[kegg_inchikey[drug]].update([atc])

    return inchikey_atc, inchikey_inchi


def parse_drugbank(inchikey_inchi, inchikey_atc=None, drugbank_xml=None):

    if not inchikey_atc:
        inchikey_atc = collections.defaultdict(set)

    # DrugBank molrepo
    dbid_inchikey = {}
    inchikey_inchi = {}
    molrepos = Molrepo.get_by_molrepo_name("drugbank")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        dbid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    # Read DrugBank

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

        # ATCs

        for atcs in drug.findall(prefix + "atc-codes"):
            for atc in atcs:
                inchikey_atc[inchikey].update([atc.attrib["code"]])

    return inchikey_atc


def break_atcs(inchikey_atc):

    def break_atc(atc):
        A = "A:%s" % atc[0]
        B = "B:%s" % atc[:3]
        C = "C:%s" % atc[:4]
        D = "D:%s" % atc[:5]
        E = "E:%s" % atc
        return [A, B, C, D, E]

    inchikey_raw = collections.defaultdict(set)
    for k, v in inchikey_atc.items():
        for x in v:
            inchikey_raw[k].update(break_atc(x))

    return inchikey_raw


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'E1.001'

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running validation for dataset ATC. Saving output in " + args.output_file)

    drugbank_xml = os.path.join(map_files["drugbank"], "full database.xml")

    br_file = os.path.join(map_files["kegg_br"], "br08303.keg")

    main._log.info("Parsing KEGG...")
    inchikey_atc, inchikey_inchi = parse_kegg(br_file)

    main._log.info("Parsing DrugBank...")
    inchikey_atc = parse_drugbank(inchikey_inchi, inchikey_atc, drugbank_xml)

    main._log.info("Breaking ATCs...")
    inchikey_raw = break_atcs(inchikey_atc)

    main._log.info("Saving validation set")

    d = {}
    for k, v in inchikey_raw.items():
        d[k] = set([x.split(":")[1] for x in v if "C:" in x])

    root = {}
    for k, v in inchikey_raw.items():
        root[k] = set([x.split(":")[1]
                       for x in v if "A:" in x])

    keys = sorted(d.keys())

    f = open(args.output_file, "w")
    for i in xrange(len(keys) - 1):
        for j in range(i + 1, len(keys)):
            if keys[i].split("-")[0] == keys[j].split("-")[0]:
                continue
            com = len(d[keys[i]].intersection(d[keys[j]]))
            if com > 0:
                v = 1
            else:
                rootcom = len(root[keys[i]].intersection(root[keys[j]]))
                if rootcom == 0:
                    v = 0
                else:
                    continue
            f.write("%s\t%s\t%d\n" % (keys[i], keys[j], v))
    f.close()


if __name__ == '__main__':
    main()
