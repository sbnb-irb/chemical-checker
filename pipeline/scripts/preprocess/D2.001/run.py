import sys
import argparse
import os
import collections
import h5py
import numpy as np
import csv
import pickle
import shutil

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.util import gaussian_scale_impute
# Variables

entry_point_full = "profile"
cell_headers_file = "cell_names.pcl"
up_down_file = "up_down.pcl"


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


def Float(x):
    try:
        return float(x)
    except:
        return np.nan


def parse_nci60(sigs, models_path=None, up=None, dw=None):

    # Read the NCI60 data

    def count_nans(v):
        return len([1 for x in v if np.isnan(x)])

    def find_strongest_signature(v):
        if len(v) == 1:
            return v[0]
        my_i = 0
        my_nonan = 0
        for i in xrange(len(v)):
            nonan = len([1 for x in v[i] if not np.isnan(x)])
            if nonan > my_nonan:
                my_i = i
                my_nonan = nonan
        return v[my_i]

    sigs = dict((k, find_strongest_signature(v)) for k, v in sigs.items())
    sigs = dict((k, v) for k, v in sigs.items() if count_nans(v) < 10)

    # Scale the signatures, and impute
    rowNames = []
    X_incomplete = []
    for k, v in sigs.items():
        rowNames += [k]
        X_incomplete += [v]
    X_incomplete = np.array(X_incomplete)

    # Scale and impute

    X = gaussian_scale_impute.scaleimpute(
        X_incomplete, models_path=models_path, up=up, dw=dw)

    return X, rowNames


# Parse arguments


@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = 'D2.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    if args.entry_point is None:
        args.entry_point = entry_point_full

    up = None
    dw = None
    sigs = collections.defaultdict(list)

    if args.method == "fit":

        if os.path.isdir(args.models_path):
            shutil.rmtree(args.models_path)
        os.mkdir(args.models_path)

        nci_inchikey = {}
        molrepos = Molrepo.get_by_molrepo_name("nci60")
        for molrepo in molrepos:
            if not molrepo.inchikey:
                continue
            nci_inchikey[molrepo.src_id] = molrepo.inchikey

        dtp_data = os.path.join(
            map_files["DTP_NCI60_ZSCORE"], "output/DTP_NCI60_ZSCORE.csv")

        with open(dtp_data, "r") as f:
            for head in f:
                if "NSC" in head:
                    headers = head.split(",")
                    cell_names = headers[6:-2]
                    break
            for l in csv.reader(f):
                if l[0] not in nci_inchikey:
                    continue
                inchikey = nci_inchikey[l[0]]
                v = [Float(x) for x in l[6:-2]]
                sigs[inchikey] += [v]

            with open(os.path.join(args.models_path, cell_headers_file), 'wb') as fh:
                pickle.dump({k: v for v, k in enumerate(cell_names)}, fh)

    if args.method == "predict":

        pre_data = []

        cell_names_map = pickle.load(
            open(os.path.join(args.models_path, cell_headers_file), 'rb'))

        up_dw_map = pickle.load(
            open(os.path.join(args.models_path, up_down_file), 'rb'))

        up = up_dw_map["up"]
        dw = up_dw_map["dw"]

        post_data = np.empty((len(cell_names_map)))

        with open(args.input_file, "r") as f:
            l = f.next()
            headers = l.rstrip().split("\t")[1:]
            for l in f:
                l = l.rstrip().split("\t")
                post_data[:] = np.nan
                pre_data = [Float(x) for x in l[1:]]
                for j, hdr in enumerate(headers):
                    if hdr in cell_names_map:
                        post_data[cell_names_map[hdr]] = pre_data[j]

                sigs[l[0]] += [post_data.tolist()]

    main._log.info("Parsing NCI-60")
    X, rowNames = parse_nci60(sigs, args.models_path, up, dw)

    if args.method == "fit":
        with open(os.path.join(args.models_path, up_down_file), 'wb') as fh:
            pickle.dump({"up": X[1], "dw": X[2]}, fh)

    main._log.info("Saving raw data")
    keys = []
    for k in rowNames:
        keys.append(str(k))
    keys = np.array(keys)
    inds = keys.argsort()
    data = []

    for i in inds:
        data.append(X[0][i])

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=keys[inds])
        hf.create_dataset("V", data=np.array(data))


if __name__ == '__main__':
    main(sys.argv[1:])
