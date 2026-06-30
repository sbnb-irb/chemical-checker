import os
import sys
import argparse
import numpy as np
import collections
import h5py
import logging

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.core.preprocess import Preprocess

# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
entry_point_full = "side_effect"


def parse_sider(sider_file):

    cid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("sider")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        cid_inchikey[molrepo.src_id] = molrepo.inchikey

    inchikey_raw_temp = collections.defaultdict(set)
    with open(sider_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            cid = l[1]
            if cid not in cid_inchikey:
                continue
            inchikey_raw_temp[cid_inchikey[cid]].update([l[2]])

    inchikey_raw = {k: list(v) for k,v in inchikey_raw_temp.items()}
    del inchikey_raw_temp

    return inchikey_raw


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):

    args = Preprocess.get_parser().parse_args(args)

    dataset_code = 'E3.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

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

        sider_file = os.path.join(
            map_files["meddra_all_se"], "meddra_all_se.tsv")

        main._log.info("Parsing SIDER")
        inchikey_raw = parse_sider(sider_file)

    if args.method == "predict":

        inchikey_raw = collections.defaultdict(list)

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                inchikey_raw[items[0]] += [items[1]]

    main._log.info("Saving raws")

    Preprocess.save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features_list)


if __name__ == '__main__':
    main(sys.argv[1:])
