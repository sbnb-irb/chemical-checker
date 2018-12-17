import os
import sys
import argparse
import numpy as np
import collections
import h5py


from chemicalchecker.util import logged
from chemicalchecker.database import Dataset, Molrepo


# Variables


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

    f = open(ecod_domains, "r")
    for l in f:
        if l[0] == "#":
            continue
        l = l.rstrip("\n").split("\t")
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
    return inchikey_ecod


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
    return parser


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'B3.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset[0].datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    ecod_domains = os.path.join(map_files["ecod"], "ecod.latest.domains.txt")

    main._log.info("Reading ECOD")
    inchikey_ecod = parse_ecod(ecod_domains)

    main._log.info("Saving raws")
    keys = []
    raws = []
    for k in sorted(inchikey_ecod.iterkeys()):
        raws.append(",".join(inchikey_ecod[k]))
        keys.append(str(k))

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=np.array(raws))

if __name__ == '__main__':
    main()
