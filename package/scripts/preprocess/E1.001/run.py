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


# Variables
dataset_code = 'E1.001'
features_file = "features.h5"
# Parse arguments
entry_point_full = "atc"


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_file',
                        type=str,
                        required=False,
                        default='.',
                        help='Input file only for predict method')
    parser.add_argument('-o', '--output_file',
                        type=str,
                        required=False,
                        default='.',
                        help='Output file')
    parser.add_argument('-m', '--method',
                        type=str,
                        required=False,
                        default='fit',
                        help='Method: fit or predict')
    parser.add_argument('-mp', '--models_path',
                        type=str,
                        required=False,
                        default='',
                        help='The models path')
    parser.add_argument('-ep', '--entry_point',
                        type=str,
                        required=False,
                        default=None,
                        help='The predict entry point')
    return parser

# Functions


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

    inchikey_raw = collections.defaultdict(set)
    for k, v in inchikey_atc.items():
        for x in v:
            inchikey_raw[k].update(break_atc(x))

    return inchikey_raw


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):
    # Reading arguments and getting datasource
    args = get_parser().parse_args(args)
    dataset = Dataset.get(dataset_code)
    main._log.debug("Running preprocess. Saving output to %s",
                    args.output_file)
    map_files = {}
    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

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
        drugbank_xml = os.path.join(map_files["drugbank"], "full database.xml")
        ATCS = parse_drugbank(ATCS, drugbank_xml)

        # break ATCs
        main._log.info("Breaking ATCs.")
        inchikey_raw = break_atcs(ATCS)

        # features will be calculated later
        features = None

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

    keys = []
    words = set()
    for k in sorted(inchikey_raw.keys()):
        keys.append(str(k))
        words.update(inchikey_raw[k])

    if features is not None:
        orderwords = features_list
        main._log.info("Predict entries have a total of %s features," +
                       " %s overlap with trainset and will be considered.",
                       len(words), len(features & words))
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        shared_features = inchikey_raw[k] & set(orderwords)
        if len(shared_features) == 0:
            main._log.warn("%s has no shared features with trainset.", k)
        for word in shared_features:
            raws[i][wordspos[word]] = 1

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))

    if args.method == "fit":
        features_path = os.path.join(args.models_path, features_file)
        with h5py.File(features_path, "w") as hf:
            hf.create_dataset("features", data=np.array(orderwords))


if __name__ == '__main__':
    main(sys.argv[1:])
