import sys
import os
import collections
import h5py
import logging
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset, Molrepo, Calcdata
from chemicalchecker.util.parser import DataCalculator
from chemicalchecker.util.parser import Converter
from chemicalchecker.core.preprocess import Preprocess

entry_point_keys = "inchikey"
entry_point_inchi = "inchi"
entry_point_smiles = "smiles"
entry_point_proteins = "proteins"
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
default_weight = 9

name = "chembl_target_predictions_v23_10um"


def key_raw_from_props(props, key_raw=None, features=None):
    if not key_raw:
        key_raw = collections.defaultdict(list)
    if features is not None:
        features_set = set(features)
    for prop in props:
        if prop[1] is None:
            continue
        for feat in prop[1].split(","):
            v = str(feat.split("(")[0])
            if features is not None:
                if v not in features_set:
                    continue
            w = int(feat.split("(")[1].split(")")[0])
            key_raw[str(prop[0])] += [(v, w)]
    return key_raw


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
        args.entry_point = entry_point_keys

    features = None

    # The entry point is available through the variable args.entry_point

    if args.method == "fit":

        main._log.info("Fitting")

        # Read the data from the datasources

        molrepos = Molrepo.get_universe_molrepos()

        main._log.info("Querying molrepos")

        inchikeys = set()

        molprop = Calcdata(name)

        for molrepo in molrepos:

            molrepo = str(molrepo[0])

            inchikeys.update(Molrepo.get_fields_by_molrepo_name(
                             molrepo, ["inchikey"]))

        props = molprop.get_properties_from_list([i[0] for i in inchikeys])

        key_raw = key_raw_from_props(props)

    if args.method == "predict":

        main._log.info("Predicting")

        key_raw = collections.defaultdict(list)

        data = []

        with h5py.File(os.path.join(args.models_path, Preprocess.features_file)) as hf:
            features = hf["features"][:]

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

            if args.entry_point == entry_point_proteins:

                features_set = set(features)

                for d in data:

                    if d[1] not in features_set:
                        continue

                    if len(d) == 2:
                        key_raw[str(d[0])] += [(d[1], default_weight)]
                    else:
                        key_raw[str(d[0])] += [(d[1], int(d[2]))]

            else:

                parse_fn = DataCalculator.calc_fn(name)

                for chunk in parse_fn(inchikey_inchi, 1000):

                    chunk = [(str(r["inchikey"]), r["raw"]) for r in chunk]

                    key_raw = key_raw_from_props(
                        chunk, key_raw, features=features)

        else:

            molprop = Calcdata(name)
            props = molprop.get_properties_from_list([i[0] for i in data])
            key_raw = key_raw_from_props(props, features=features)

    main._log.info("Saving raw data")

    Preprocess.save_output(args.output_file, key_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
