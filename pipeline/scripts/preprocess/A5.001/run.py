import os
import sys
import argparse
import collections
import h5py
import numpy as np


from chemicalchecker.util import logged
from chemicalchecker.database import Molprop
from chemicalchecker.database import Molrepo
from chemicalchecker.util import PropCalculator
from chemicalchecker.util import Converter


# Variables

entry_point_keys = "inchikey"
entry_point_inchi = "inchi"
entry_point_smiles = "smiles"

name = "physchem"


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

    dataset_code = 'A5.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_keys

    molrepos = ["bindingdb"]

    if args.method == "fit":

        main._log.info("Querying molrepos")

        ACTS = []

        molprop = Molprop(name)

        for molrepo in molrepos:

            inchikeys = Molrepo.get_fields_by_molrepo_name(
                molrepo, ["inchikey"])
            props = molprop.get_properties_from_list([i[0] for i in inchikeys])
            ACTS.extend(props)

    if args.method == "predict":

        ACTS = []

        data = []

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

    sigs = collections.defaultdict(list)
    for k in ACTS:
        if k[1] is None:
            continue
        vals = [float(t) for t in k[1].split(",")]
        sigs[str(k[0])] = vals

    keys = []
    for k in sigs.keys():
        keys.append(str(k))
    keys = np.array(keys)
    inds = keys.argsort()
    data = []

    for i in inds:
        data.append(sigs[keys[i]])

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=keys[inds])
        hf.create_dataset("V", data=np.array(data))


if __name__ == '__main__':
    main(sys.argv[1:])
