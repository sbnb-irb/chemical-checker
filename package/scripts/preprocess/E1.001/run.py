import os
import sys
import h5py
import logging
import argparse
import numpy as np
import collections
import xml.etree.ElementTree as ET

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.core.preprocess import Preprocess

# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
# Parse arguments
entry_point_full = "atc"


def parse_kegg(br_file, inchikey_atc=None):
    if not inchikey_atc:
        inchikey_atc = collections.defaultdict(set)

    # KEGG molrepo
    kegg_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("kegg")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        kegg_inchikey[molrepo.src_id] = molrepo.inchikey

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

    return inchikey_atc


def parse_drugbank(inchikey_atc=None, drugbank_xml=None):

    if not inchikey_atc:
        inchikey_atc = collections.defaultdict(set)

    # DrugBank molrepo
    dbid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("drugbank")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        dbid_inchikey[molrepo.src_id] = molrepo.inchikey

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

    inchikey_raw_temp = collections.defaultdict(set)
    for k, v in inchikey_atc.items():
        for x in v:
            inchikey_raw_temp[k].update(break_atc(x))

    inchikey_raw = {k: list(v) for k,v in inchikey_raw_temp.items()}
    del inchikey_raw_temp

    return inchikey_raw


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):
    # Reading arguments and getting datasource
    args = Preprocess.get_parser().parse_args(args)
    main._log.debug("Running preprocess. Saving output to %s",
                    args.output_file)
    dataset = Dataset.get(dataset_code)

    map_files = {}

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.
    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path

    # decide entry point, if None use default
    if args.entry_point is None:
        args.entry_point = entry_point_full

    # main FIT section
    if args.method == "fit":

        # fetch Anatomical Therapeutic Chemical (ATC) from KEGG and DrugBank
        main._log.info("Parsing KEGG.")
        kegg_br = os.path.join(map_files["kegg_br"], "br08303.keg")
        ATCS = parse_kegg(kegg_br)

        main._log.info("Parsing DrugBank.")
    
        file_path = map_files["drugbank"]
        if( os.path.isdir(file_path) ):
            fxml = ''
            for fs in os.listdir(file_path) :
                if( fs.endswith('.xml') ):
                    fxml = fs
            drugbank_xml = os.path.join(file_path, fxml)
        #drugbank_xml = os.path.join(map_files["drugbank"], "full database.xml")
        ATCS = parse_drugbank(ATCS, drugbank_xml)

        # break ATCs
        main._log.info("Breaking ATCs.")
        inchikey_raw = break_atcs(ATCS)

        # features will be calculated later
        features = None
        features_list = None

    # main PREDICT section
    if args.method == "predict":

        # fetch ATCS from input file
        ATCS = collections.defaultdict(set)
        with open(args.input_file) as f:
            for l in f:
                items = l.rstrip().split("\t")
                ATCS[items[0]].add(items[1])

        # break ATCs
        main._log.info("Breaking ATCs.")
        inchikey_raw = break_atcs(ATCS)

        # load features (saved at FIT time)
        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

    # save raw values
    main._log.info("Saving raw data.")

    Preprocess.save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features_list)


if __name__ == '__main__':
    main(sys.argv[1:])
