import os
import sys
import argparse
import numpy as np
import collections
import h5py


from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
# Variables

features_file = "features.h5"
entry_point_full = "side_effect"


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

def parse_sider(sider_file):

    cid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("sider")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        cid_inchikey[molrepo.src_id] = molrepo.inchikey

    inchikey_raw = collections.defaultdict(set)
    with open(sider_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            cid = l[1]
            if cid not in cid_inchikey:
                continue
            inchikey_raw[cid_inchikey[cid]].update([l[2]])

    return inchikey_raw


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'E3.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_full

    features = None

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
    main()
