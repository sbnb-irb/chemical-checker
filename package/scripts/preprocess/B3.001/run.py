import os
import sys
import argparse
import numpy as np
import collections
import h5py
import pickle

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset, Molrepo


# Variables
features_file = "features.h5"
map_family_file = "family.pickl"
map_pdb_file = "pdb.pickl"
entry_point_structures = "structures"
entry_point_domains = "domains"
entry_point_dm_hierch = "domain_hierarchies"


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

def parse_ecod(ecod_domains):

    # Read molrepo

    ligand_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("pdb")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        ligand_inchikey[molrepo.src_id] = molrepo.inchikey

    # Parse ECOD
    # [X-group].[H-group].[T-group].[F-group]

    inchikey_ecod = collections.defaultdict(set)
    map_pdb = collections.defaultdict(set)
    map_family_id = collections.defaultdict(set)

    f = open(ecod_domains, "r")
    for l in f:
        if l[0] == "#":
            continue
        l = l.rstrip("\n").split("\t")
        map_family_id[l[1]].update([l[3]])
        map_pdb[l[4]].update([l[1]])
        s = "E:" + l[1]
        f_id = l[3].split(".")
        s += ",X:" + f_id[0]
        s += ",H:" + f_id[1]
        s += ",T:" + f_id[2]
        if len(f_id) == 4:
            s += ",F:" + f_id[3]
        lig_ids = l[-1].split(",")
        for lig_id in lig_ids:
            if lig_id not in ligand_inchikey:
                continue
            inchikey_ecod[ligand_inchikey[lig_id]].update([s])
    f.close()
    return inchikey_ecod, map_family_id, map_pdb


def parse_data_ecodid(data, map_family_id):

    inchikey_ecod = collections.defaultdict(set)
    for k, ecod in data.items():
        f_ids = map_family_id[ecod]
        if f_ids is None:
            continue
        s = "E:" + ecod
        for f_id in f_ids:
            s += ",X:" + f_id[0]
            s += ",H:" + f_id[1]
            s += ",T:" + f_id[2]
            if len(f_id) == 4:
                s += ",F:" + f_id[3]
        inchikey_ecod[k].update([s])

    return inchikey_ecod


def parse_data_pdb(data, map_family_id, map_pdb):

    inchikey_ecod = collections.defaultdict(set)
    for k, pdb in data.items():
        for ecod in map_pdb[pdb]:
            f_ids = map_family_id[ecod]
            if f_ids is None:
                continue
            s = "E:" + ecod
            for f_id in f_ids:
                s += ",X:" + f_id[0]
                s += ",H:" + f_id[1]
                s += ",T:" + f_id[2]
                if len(f_id) == 4:
                    s += ",F:" + f_id[3]
            inchikey_ecod[k].update([s])

    return inchikey_ecod


@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = 'B3.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_domains

    features = None
    RAW = collections.defaultdict(list)

    if args.method == "fit":

        ecod_domains = os.path.join(
            map_files["ecod"], "ecod.latest.domains.txt")

        main._log.info("Reading ECOD")
        inchikey_ecod, map_family_id, map_pdb = parse_ecod(ecod_domains)

        with open(os.path.join(args.models_path, map_family_file), 'wb') as fh:
            pickle.dump(map_family_id, fh)

        with open(os.path.join(args.models_path, map_pdb_file), 'wb') as fh:
            pickle.dump(map_pdb, fh)

    if args.method == "predict":

        data = {}

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        map_family_id = pickle.load(
            open(os.path.join(args.models_path, map_family_file), 'rb'))

        map_pdb = pickle.load(
            open(os.path.join(args.models_path, map_pdb_file), 'rb'))

        with open(args.input_file) as f:

            if args.entry_point == entry_point_dm_hierch:
                for l in f:
                    items = l.rstrip().split("\t")
                    if items[1] not in features:
                        continue
                    RAW[items[0]] += [items[1]]
            else:

                for l in f:
                    items = l.rstrip().split("\t")
                    data[items[0]] = items[1]

        if args.entry_point == entry_point_structures:
            inchikey_ecod = parse_data_ecodid(data, map_family_id)

        if args.entry_point == entry_point_domains:
            inchikey_ecod = parse_data_pdb(data, map_family_id, map_pdb)

    main._log.info("Saving raws")

    if args.entry_point != entry_point_dm_hierch:
        for k, v in inchikey_ecod.items():
            for ele in v:
                wl = ele.split(",")
                for w in wl:
                    if features is not None and w not in features:
                        continue
                    RAW[k] += [w]

    keys = []
    words = set()

    for k in sorted(RAW.keys()):
        keys.append(str(k))
        words.update(RAW[k])

    if features is not None:
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in RAW[k]:
            raws[i][wordspos[word]] += 1

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))

    if args.method == "fit":
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=np.array(orderwords))

if __name__ == '__main__':
    main(sys.argv[1:])
