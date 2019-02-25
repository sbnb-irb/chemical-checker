import sys
import os
import argparse
import collections
import h5py
import numpy as np

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo

features_file = "features.h5"

entry_point_full = "strain"

pval_01 = 3.37
pval_001 = 7.12


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


# Functions

def read_mosaic_predictions(all_conditions, comb_gt_preds):

    molrepos = Molrepo.get_by_molrepo_name("mosaic")
    cgid_inchikey = {}
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        cgid_inchikey[molrepo.src_id] = molrepo.inchikey

    conds = {}
    f = open(all_conditions, "r")
    H = f.next().split("\t")[1:]
    f.close()

    for h in H:
        cond, Id = h.split("_")
        if Id not in cgid_inchikey:
            continue
        conds[cond] = cgid_inchikey[Id]

    sig = collections.defaultdict(list)
    with open(comb_gt_preds, "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[0] not in conds:
                continue
            ik = conds[l[0]]
            s = float(l[2])
            if s > pval_01:  # P-value 0.01
                if s > pval_001:  # P-value 0.001
                    d = 2
                else:
                    d = 1
                sig[ik] += [(l[1], d)]
                #sig[ik].update(["%s(%d)" % (l[1], d)])

    return sig


@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = 'D3.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

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

        all_conditions = os.path.join(
            map_files["mosaic_all_conditions"], "All_conditions.txt")
        comb_gt_preds = os.path.join(
            map_files["mosaic"], "combined_gene-target-predictions.txt")

        inchikey_raw = read_mosaic_predictions(
            all_conditions, comb_gt_preds)

    if args.method == "predict":

        ACTS = {}

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                if len(items) < 3:
                    ACTS[(items[0], items[1])] = 1
                else:
                    ACTS[(items[0], items[1])] = int(items[2])

        inchikey_raw = collections.defaultdict(list)
        for k, v in ACTS.items():
            inchikey_raw[k[0]] += [(k[1], v)]

    main._log.info("Saving raws")

    keys = []
    words = set()
    for k in sorted(inchikey_raw.keys()):

        keys.append(str(k))
        words.update([x[0] for x in inchikey_raw[k]])

    if features is not None:
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_raw[k]:
            raws[i][wordspos[word[0]]] = word[1]

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))

    if args.method == "fit":
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=np.array(orderwords))

if __name__ == '__main__':
    main(sys.argv[1:])
