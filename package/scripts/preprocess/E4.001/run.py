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
# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
entry_point_full = "disease"


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
    for k, v in dis_tree.iteritems():
        for x in v:
            tree_dis[x] += [k]

    def expand_tree(tn):
        tns = []
        x = tn.split("/")[0].split(".")
        for i in xrange(len(x)):
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

        chemdis_file = os.path.join(
            map_files["CTD_chemicals_diseases"], "CTD_chemicals_diseases.tsv")

        disfile = os.path.join(
            map_files["CTD_diseases"], "CTD_diseases.tsv")

        main._log.info("Parsing CTD...")
        inchikey_raw = parse_ctd(disfile, chemdis_file)

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

    keys = []
    words = set()
    for k in sorted(inchikey_raw.keys()):

        keys.append(str(k))
        words.update(inchikey_raw[k])

    if features is not None:
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_raw[k]:
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
