import os
import sys
import argparse
import collections
import h5py
import numpy as np


from chemicalchecker.util import logged
from chemicalchecker.database import Molprop, Datasource
from chemicalchecker.database import Molrepo
from chemicalchecker.util import PropCalculator
from chemicalchecker.util import Converter


# Variables

features_file = "features.h5"
entry_point_keys = "inchikey"
entry_point_inchi = "inchi"
entry_point_smiles = "smiles"

name = "scaffolds"


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


@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = 'A3.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_keys

    features = None

    if args.method == "fit":

        molrepos = Datasource.get_universe_molrepos()

        main._log.info("Querying molrepos")

        ACTS = []
        inchikeys = set()

        molprop = Molprop(name)

        for molrepo in molrepos:

            molrepo = str(molrepo[0])

            inchikeys.update(Molrepo.get_fields_by_molrepo_name(
                molrepo, ["inchikey"]))
        props = molprop.get_properties_from_list([i[0] for i in inchikeys])
        ACTS.extend(props)

    if args.method == "predict":

        ACTS = []

        data = []

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                data.append(items)

        if args.entry_point == entry_point_smiles:

            inchikey_inchi = {}

            for d in data:

                try:
                    inchikey, inchi = Converter.smiles_to_inchi(d[1])
                except Exception:
                    continue

                inchikey_inchi[d[0]] = inchi

        if args.entry_point == entry_point_inchi:

            inchikey_inchi = dict(data)

        if args.entry_point != entry_point_keys:

            parse_fn = PropCalculator.calc_fn(name)

            for chunk in parse_fn(inchikey_inchi, 1000):

                for prop in chunk:
                    ACTS.append((prop["inchikey"], prop["raw"]))

        else:

            molprop = Molprop(name)
            props = molprop.get_properties_from_list([i[0] for i in data])
            ACTS.extend(props)

    main._log.info("Saving raws")
    RAW = collections.defaultdict(list)
    for k in ACTS:
        if features is None:
            vals = [str(t) for t in k[1].split(",")]
        else:
            vals = [str(t) for t in k[1].split(",") if str(t) in features]
        RAW[str(k[0])] = vals

    keys = []
    words = set()
    for k in sorted(RAW.keys()):
        keys.append(str(k))
        words.update(RAW[k])

    if features is not None:
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort(key=int)

    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in RAW[k]:
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
