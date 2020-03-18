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
entry_point_full = "disease"

# Parse arguments

def parse_ctd(disfile, chemdis_file):

    ctd_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("ctd")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        ctd_inchikey[molrepo.src_id] = molrepo.inchikey

    dis_tree = collections.defaultdict(list)
    f = open(disfile, "r")
    for l in f:
        if l[0] == "#":
            continue
        l = l.rstrip("\n").split("\t")
        dis_tree[l[1]] = l[5].split("|")
    f.close()

    tree_dis = collections.defaultdict(list)
    for k, v in dis_tree.items():
        for x in v:
            tree_dis[x] += [k]

    def expand_tree(tn):
        tns = []
        x = tn.split("/")[0].split(".")
        for i in range(len(x)):
            tns += [".".join(x[:i + 1])]
        tns += [tn]
        return tns

    f = open(chemdis_file, "r")
    inchikey_raw = collections.defaultdict(set)

    for l in f:
        if l[0] == "#":
            continue
        l = l.rstrip("\n").split("\t")
        if l[5] == "":
            continue
        dis = l[4]
        cid = l[1]
        if cid not in ctd_inchikey:
            continue
        inchikey = ctd_inchikey[cid]
        ev = l[5]
        for tn in dis_tree[dis]:
            exp_tns = expand_tree(tn)
            exp_dis = set()
            for exp_tn in exp_tns:
                exp_dis.update(tree_dis[exp_tn])
        exp_dis = sorted(exp_dis)
        for d in exp_dis:
            if "MESH" not in d:
                continue
            x = []
            if ev == "marker/mechanism":
                x += [d + "(M)"]
            if ev == "therapeutic":
                x += [d + "(T)"]
            for y in x:
                inchikey_raw[inchikey].update([y.split("MESH:")[1]])
    f.close()

    return inchikey_raw


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

        chemdis_file = os.path.join(
            map_files["CTD_chemicals_diseases"], "CTD_chemicals_diseases.tsv")

        disfile = os.path.join(
            map_files["CTD_diseases"], "CTD_diseases.tsv")

        main._log.info("Parsing CTD...")
        inchikey_raw_temp = parse_ctd(disfile, chemdis_file)
        inchikey_raw = {k: list(v) for k,v in inchikey_raw_temp.items()}

    if args.method == "predict":

        inchikey_raw = collections.defaultdict(list)

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                val = items[1] + "(" + items[2] + ")"
                if val not in features:
                    continue
                inchikey_raw[items[0]] += [val]

    main._log.info("Saving raws")

    Preprocess.save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features_list)


if __name__ == '__main__':
    main(sys.argv[1:])
