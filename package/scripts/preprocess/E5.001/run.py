import os
import sys
import argparse
import numpy as np
import collections
import h5py
import xml.etree.ElementTree as ET
import logging

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.core.preprocess import Preprocess

# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
entry_point_full = "drug"

# Parse arguments

def parse_ddis(drugbank_xml):

    tree = ET.parse(drugbank_xml)

    dbid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("drugbank")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        dbid_inchikey[molrepo.src_id] = molrepo.inchikey

    inchikey_ddi = collections.defaultdict(list)

    root = tree.getroot()
    prefix = "{http://www.drugbank.ca}"
    for drug in root:
        for child in drug.findall(prefix + "drugbank-id"):
            if "primary" not in child.attrib:
                continue
            if child.attrib["primary"] == "true":
                # print "primary: " + child.text
                db_id = child.text
                if db_id not in dbid_inchikey:
                    continue
                drug_interactions = drug.find(prefix + 'drug-interactions')
                drug_inter = drug_interactions.findall(
                    prefix + 'drug-interaction')
                # print len(drug_inter)
                for inter in drug_inter:
                    for child_did in inter.findall(prefix + "drugbank-id"):
                        # print child_did.text
                        inchikey_ddi[dbid_inchikey[db_id]] += [child_did.text]

    return inchikey_ddi


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
    features_list = None

    if args.method == "fit":
    
        file_path = map_files["drugbank"]
        if( os.path.isdir(file_path) ):
            fxml = ''
            for fs in os.listdir(file_path) :
                if( fs.endswith('.xml') ):
                    fxml = fs
            drugbank_xml = os.path.join(file_path, fxml)

        #drugbank_xml = os.path.join( map_files["drugbank"], "full database.xml")

        main._log.info("Parsing DDIs...")
        inchikey_ddi = parse_ddis(drugbank_xml)

    if args.method == "predict":

        inchikey_ddi = collections.defaultdict(list)

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                inchikey_ddi[items[0]] += [items[1]]

    main._log.info("Saving raws")

    Preprocess.save_output(args.output_file, inchikey_ddi, args.method,
                args.models_path, dataset.discrete, features_list)

if __name__ == '__main__':
    main(sys.argv[1:])
