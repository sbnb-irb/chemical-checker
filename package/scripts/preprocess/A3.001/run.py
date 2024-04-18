import os
import sys
import collections
import h5py
import logging

from chemicalchecker.core.chemcheck import ChemicalChecker
from chemicalchecker.util import logged, Config
from chemicalchecker.database import Calcdata
from chemicalchecker.database import Molrepo
from chemicalchecker.util.parser import DataCalculator
from chemicalchecker.util.parser import Converter
from chemicalchecker.core.preprocess import Preprocess


# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
entry_point_keys = "inchikey"
entry_point_inchi = "inchi"
entry_point_smiles = "smiles"

name = "murcko_1024_cframe_1024"


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):

    args = Preprocess.get_parser().parse_args(args)

    main._log.info("Running %s preprocess. Saving output in %s" %
                   (dataset_code, args.output_file))
    main._log.debug("ARGS: %s" % args)

    if args.entry_point is None:
        args.entry_point = entry_point_keys

    features = None

    if args.method == "fit":
        ccinstance = ChemicalChecker( os.path.realpath(Config().PATH.CC_ROOT) )
        universe = set( ccinstance.universe )
        
        ACTS = []
        inchikeys = set()
        key_list = []
        molprop = Calcdata(name)
        if( len(universe) == 0 ):
            molrepos = Molrepo.get_universe_molrepos()
            main._log.info("Querying molrepos")
            
            for molrepo in molrepos:
                molrepo = str(molrepo[0])
                inchikeys.update(Molrepo.get_fields_by_molrepo_name(
                    molrepo, ["inchikey"]))
                key_list = [ i[0] for i in inchikeys ]
        else:
            inchikeys = universe
            key_list = [i for i in inchikeys]
            
        props = molprop.get_properties_from_list( key_list )
        ACTS.extend(props)

    if args.method == "predict":

        ACTS = []
        data = []
        features_path = os.path.join(args.models_path, features_file)
        if os.path.isfile(features_path):
            with h5py.File(os.path.join(args.models_path, features_file)) as hf:
                features_list = hf["features"][:]
                features = list([f.decode() for f in features_list])
            main._log.debug("Loaded %s features" % len(features))

        with open(args.input_file) as fh:
            for line in fh:
                items = line.rstrip().split("\t")
                data.append(items)
        main._log.debug("Predicting %s molecules" % len(data))

        # input is SMILES
        if args.entry_point == entry_point_smiles:
            inchikey_inchi = {}
            converter = Converter()
            for d in data:
                try:
                    inchikey, inchi = converter.smiles_to_inchi(d[1])
                except Exception:
                    continue
                inchikey_inchi[d[0]] = inchi

        # input is InChIKey InChI
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

    main._log.info("Saving raws for %s molecules" % len(ACTS))
    RAW = collections.defaultdict(list)
    # ACTS contains the dense format for raw data
    for k in ACTS:
        if k[1] == '' or k[1] is None or k[0] is None:
            main._log.warning('Could not calculate signature for %s' % k[0])
            continue
        if features is None:
            vals = [str(t) for t in k[1].split(",")]
        else:
            vals = [str(t) for t in k[1].split(",") if str(t) in features]
        RAW[str(k[0])] = vals

    Preprocess.save_output(args.output_file, RAW, args.method,
                           args.models_path, True, features)


if __name__ == '__main__':
    main(sys.argv[1:])
