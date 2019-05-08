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
# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
entry_point_full = "drug"


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

    args = get_parser().parse_args(args)

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

        drugbank_xml = os.path.join(
            map_files["drugbank"], "full database.xml")

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

    keys = []
    words = set()
    for k in sorted(inchikey_ddi.keys()):

        keys.append(str(k))
        words.update(inchikey_ddi[k])

    if features is not None:
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_ddi[k]:
            raws[i][wordspos[word]] = 1

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))

    if args.method == "fit":
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=np.array(orderwords))

if __name__ == '__main__':
    main(sys.argv[1:])
