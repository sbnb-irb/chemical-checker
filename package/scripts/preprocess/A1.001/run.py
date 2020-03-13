import os
import sys
import argparse
import collections
import h5py
import numpy as np
import logging

from chemicalchecker.util import logged
from chemicalchecker.database import Calcdata
from chemicalchecker.database import Molrepo, Datasource
from chemicalchecker.util.parser import DataCalculator
from chemicalchecker.util.parser import Converter
from chemicalchecker.core.preprocess import Preprocess


# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
entry_point_keys = "inchikey"
entry_point_inchi = "inchi"
entry_point_smiles = "smiles"

name = "morgan_fp_r2_2048"

# Parse arguments


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):

    args = Preprocess.get_parser().parse_args(args)

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_keys

    features = None

    if args.method == "fit":

        molrepos = Molrepo.get_universe_molrepos()

        main._log.info("Querying molrepos")

        ACTS = []
        inchikeys = set()

        molprop = Calcdata(name)

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

            converter = Converter()

            for d in data:

                try:
                    inchikey, inchi = converter.smiles_to_inchi(d[1])
                except Exception:
                    continue

                inchikey_inchi[d[0]] = inchi

        if args.entry_point == entry_point_inchi:

            inchikey_inchi = dict(data)

        if args.entry_point != entry_point_keys:

            parse_fn = DataCalculator.calc_fn(name)

            for chunk in parse_fn(inchikey_inchi, 1000):

                for prop in chunk:
                    ACTS.append((prop["inchikey"], prop["raw"]))

        else:

            molprop = Calcdata(name)
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

    Preprocess.save_output(args.output_file, RAW, args.method,
                args.models_path, True, features, features_int=True)


if __name__ == '__main__':
    main(sys.argv[1:])
