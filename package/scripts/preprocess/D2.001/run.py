import sys
import argparse
import os
import collections
import h5py
import numpy as np
import pandas as pd
import csv
import pickle
import shutil
import logging
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.util.transform import gaussian_scale_impute
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.core.signature_data import DataSignature

# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
entry_point_full = "profile"
features_file = "features.h5"
cell_headers_file = "cell_names.pcl"
up_down_file = "up_down.pcl"


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
        for i in range(len(v)):
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

    X = gaussian_scale_impute(
        X_incomplete, models_path=models_path, up=up, dw=dw)

    return X, rowNames


# Parse arguments


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
        args.entry_point = entry_point_full

    up = None
    dw = None
    cell_names = []
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
            map_files["DTP_NCI60_ZSCORE"], "output/DTP_NCI60_ZSCORE.xlsx")

        print("Converting NCI60 Zscore xlsx file to csv")
        data_xls = pd.read_excel(dtp_data, index_col=0)
        csv_path = dtp_data[:-5] + ".csv"
        data_xls.to_csv(csv_path, encoding='utf-8')
        
        with open(csv_path, "r") as f:
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

        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=np.array(cell_names, DataSignature.string_dtype()))
            # pickle.dump({k: v for v, k in enumerate(cell_names)}, fh)

    if args.method == "predict":

        pre_data = []

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            cell_names = hf["features"][:]

        cell_names_map = {k: v for v, k in enumerate(cell_names)}

        up_dw_map = pickle.load(
            open(os.path.join(args.models_path, up_down_file), 'rb'))

        up = up_dw_map["up"]
        dw = up_dw_map["dw"]

        post_data = np.empty((len(cell_names_map)))

        with open(args.input_file, "r") as f:
            l = f.readline()
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
        hf.create_dataset("keys", data=np.array(keys[inds], DataSignature.string_dtype()))
        hf.create_dataset("X", data=np.array(data))
        hf.create_dataset("features", data=np.array(cell_names, DataSignature.string_dtype()))


if __name__ == '__main__':
    main(sys.argv[1:])
