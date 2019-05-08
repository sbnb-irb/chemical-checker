'''
DeepCodex signatures.

Test file snippet:

entry_point = proteins:

s1  Q05209  -1
s2  Q9UBU9  -1
s3  P32243  1
'''
import sys
import os
import collections
import h5py
import logging
from chemicalchecker.util import logged, get_parser, save_output, features_file
from chemicalchecker.database import Dataset, Molrepo

import csv
import numpy as np

# Variables

dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]

# Entry points

entry_point_full = "proteins"

# Functions


def dcx_to_pertid(map_files):
    filename = map_files["deepcodex_map"] + "/dcx_map.csv"
    dcx_pertid = collections.defaultdict(list)
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for r in reader:
            dcx_pertid[r[0]] += [r[1]]
    return dcx_pertid


def pertid_to_meta(map_files):
    filenames = [map_files["GSE70138_Broad_LINCS_pert_info"] + "/GSE70138_Broad_LINCS_pert_info.txt",
                 map_files["GSE92742_Broad_LINCS_pert_info"] + "/GSE92742_Broad_LINCS_pert_info.txt"]
    pertid_meta = collections.defaultdict(set)
    for filename in filenames:
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            h = next(reader)
            pert_id_idx = h.index("pert_id")
            pert_iname_idx = h.index("pert_iname")
            pert_type_idx = h.index("pert_type")
            for r in reader:
                pertid_meta[r[pert_id_idx]].update(
                    [(r[pert_iname_idx], r[pert_type_idx])])
    return pertid_meta


def parse_molrepo():
    lincs_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("lincs")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        lincs_inchikey[molrepo.src_id] = molrepo.inchikey
    return lincs_inchikey


def dcx_to_cc_dicts(map_files):
    lincs_inchikey = parse_molrepo()
    dcx_pertid = dcx_to_pertid(map_files)
    pertid_meta = pertid_to_meta(map_files)
    dcx_cc = collections.defaultdict(set)
    cc_dcx = collections.defaultdict(set)
    for dcx, pertids in dcx_pertid.iteritems():
        for pertid in pertids:
            if pertid in lincs_inchikey:
                k = (lincs_inchikey[pertid], "cp")
                dcx_cc[dcx].update([k])
                cc_dcx[k].update([dcx])
            else:
                if pertid in pertid_meta:
                    for pert_iname, pert_itype in pertid_meta[pertid]:
                        if "trt_oe" in pert_itype:
                            k = (pert_iname, "oe")
                        elif "trt_sh" in pert_itype:
                            k = (pert_iname, "sh")
                        else:
                            continue
                        dcx_cc[dcx].update([k])
                        cc_dcx[k].update([dcx])
    return dcx_cc, cc_dcx


def accession_to_genename(map_files):
    prot2gene = collections.defaultdict(set)
    with open(map_files["human_proteome"] + "/download.wget", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        entry_idx = header.index("Entry")
        genenames_idx = header.index("Gene names")
        for r in reader:
            prot = r[entry_idx]
            gns = r[genenames_idx].split(" ")
            for gn in gns:
                prot2gene[prot].update([gn])
    return prot2gene


def parse_deepcodex(dcx, dcx_cc, map_files):
    foldername = map_files["deepcodex_download"] + "/download/"
    with open(foldername + "/" + dcx, "r") as f:
        reader = csv.reader(f)
        next(reader)
        hits = []
        for r in reader:
            hits += [(r[1], r[3])]
        n = len(hits)
        hits = [(x, n - i) for i, x in enumerate(hits)]
        hits_cc = []
        for x, n in hits:
            if x[0] not in dcx_cc:
                continue
            for y in dcx_cc[x[0]]:
                hits_cc += [(y, int(n / 10.) + 1)]
        return hits_cc

# Main


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

    dcx_cc, cc_dcx = dcx_to_cc_dicts(map_files)

    features = None

    if args.method == "fit":

        main._log.info("Fitting")

        # Read the data from the datasources
        key_pairs = collections.defaultdict(list)
        for dcx in os.listdir(map_files["deepcodex_download"] + "/download/"):
            if dcx not in dcx_cc:
                continue
            for cc in dcx_cc[dcx]:
                if cc[1] != "cp":
                    continue
                hits = parse_deepcodex(dcx, dcx_cc, map_files)
                for hit in hits:
                    key_pairs[(cc, hit[0])] += [hit[1]]
        key_pairs = dict((k, np.max(v)) for k, v in key_pairs.iteritems())
        key_raw = collections.defaultdict(list)
        for k, v in key_pairs.iteritems():
            key_raw[str(k[0][0])] += [(str(k[1][0] + "_" + k[1][1]), v)]
        features = sorted(set([x[0] for v in key_raw.values() for x in v]))

    if args.method == "predict":

        main._log.info("Predicting")

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features = hf["features"][:]
            features_set = set(features)

        prot2gene = accession_to_genename(map_files)

        # Read the data from the args.input_file
        # The entry point is available through the variable args.entry_point

        if args.entry_point is None:
            args.entry_point = entry_point_full

        key_pairs = collections.defaultdict(list)
        with open(args.input_file, "r") as f:
            for r in csv.reader(f, delimiter="\t"):
                # Get the key
                key = r[0]
                # Get the direction
                if r[2] == "-1":
                    direction = "sh"
                elif r[2] == "1":
                    direction = "oe"
                else:
                    continue
                # Get the genenames
                prot = r[1]
                if prot not in prot2gene:
                    continue
                for gn in prot2gene[prot]:
                    if (gn, direction) not in cc_dcx:
                        continue
                    for dcx in cc_dcx[(gn, direction)]:
                        if dcx not in dcx_cc:
                            continue
                        hits = parse_deepcodex(dcx, dcx_cc, map_files)
                        for hit in hits:
                            key_pairs[((key, direction), hit[0])] += [hit[1]]
        key_pairs = dict((k, np.max(v)) for k, v in key_pairs.iteritems())
        key_raw = collections.defaultdict(list)
        for k, v in key_pairs.iteritems():
            feat = str(k[1][0] + "_" + k[1][1])
            if feat not in features_set:
                continue
            key_raw[str(k[0][0])] += [(feat, v)]

    main._log.info("Saving raw data")

    save_output(args.output_file, key_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
