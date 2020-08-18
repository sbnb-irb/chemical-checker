import os
import sys
import h5py
import pickle
import shutil
import random
import logging
import collections
import numpy as np
from cmapPy.pandasGEXpress import parse

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.util.transform import gaussian_scale_impute
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.core.signature_data import DataSignature

# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
entry_point_full = "measure"
cell_headers_file = "cell_names.pcl"
features_file = "features.h5"
up_down_file = "up_down.pcl"


perc = 95
Br = 5
B = 100000


# Functions


def parse_morphology(lds_file, models_path=None, up=None, dw=None):

    pd = parse.parse(lds_file)
    rownames = []
    for c in list(pd.col_metadata_df["pert_id"]):
        pertid = c.split("-")
        pertid = "%s-%s" % (pertid[0], pertid[1])
        rownames += [pertid]
    X = np.array(pd.data_df).T
    X, up, dw = gaussian_scale_impute(
        X, models_path=models_path, up=up, dw=dw)
    rownames = np.array(rownames)
    cols = list(np.array(pd.row_metadata_df["ImageFeature"]))

    return X, rownames, cols, up, dw


def find_strongest_signature(v):
    if len(v) == 1:
        return v[0]
    my_i = 0
    Sum = 0
    for i in range(len(v)):
        x = np.sum(np.abs(v[i]))
        if x > Sum:
            my_i = i
            Sum = x
    return v[my_i]


def Float(x):
    try:
        return float(x)
    except:
        return np.nan


def colshuffled_matrix(X):
    Xr = np.array(X)
    for j in range(Xr.shape[1]):
        shuffled = sorted(Xr[:, j], key=lambda k: random.random())
        Xr[:, j] = shuffled
    return Xr


def filter_data(X, rownames, pertid_inchikey=None):

    # Xd
    Xd = np.random.sample(size=(B, X.shape[1]))

    # Xr
    Arrs = []
    for _ in range(Br):
        Arrs += [colshuffled_matrix(X)]
    Xr = np.vstack(tuple(Arrs))

    # Cuts
    A = np.array([np.sum(np.abs(x)) for x in Xr])
    D = np.array([np.sum(np.abs(x)) for x in Xd])
    T = np.array([np.sum(np.abs(x)) for x in X])
    cutoff = int(0.5 * np.percentile(A, perc) + 0.5 * np.percentile(D, perc))

    rownames_f = np.array(rownames)[T > cutoff]
    X_f = X[T > cutoff]

    # To signatures
    sigs = collections.defaultdict(list)
    for i in range(len(rownames_f)):
        if pertid_inchikey is not None:
            if rownames_f[i] not in pertid_inchikey:
                continue
            ik = pertid_inchikey[rownames_f[i]]
        else:
            ik = rownames_f[i]
        sigs[ik] += [X_f[i]]

    sigs = dict((k, find_strongest_signature(v)) for k, v in sigs.items())

    return sigs


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
    sigs = collections.defaultdict(list)
    features = None

    if args.method == "fit":

        if os.path.isdir(args.models_path):
            shutil.rmtree(args.models_path)
        os.mkdir(args.models_path)

        lds_1195 = os.path.join(
            map_files["morphlincs_LDS-1195"], "LDS-1195/Data/cdrp.img.profiles_n30440x812.gctx")

        pertid_inchikey = {}
        molrepos = Molrepo.get_by_molrepo_name("morphlincs")
        for molrepo in molrepos:
            if not molrepo.inchikey:
                continue
            pertid_inchikey[molrepo.src_id] = molrepo.inchikey

        X, rownames, col_names, up, dw = parse_morphology(
            lds_1195, args.models_path, up, dw)

        features = col_names

        with open(os.path.join(args.models_path, up_down_file), 'wb') as fh:
            pickle.dump({"up": up, "dw": dw}, fh)

        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=DataSignature.h5_str(features))

        main._log.info("Filtering...")
        sigs = filter_data(X, rownames, pertid_inchikey)

    if args.method == "predict":

        pertid_inchikey = None
        pre_data = []
        X = []
        rownames = []

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features = hf["features"][:]

        cell_names_map = {k: v for v, k in enumerate(features)}

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

                X.append(post_data.tolist())
                rownames.append(l[0])
        X, up, dw = gaussian_scale_impute(
            X, models_path=args.models_path, up=up, dw=dw)
        for i in range(0, len(rownames)):
            sigs[rownames[i]] = X[i]

    main._log.info("Saving raw data")

    Preprocess.save_output(args.output_file, sigs, args.method,
                           args.models_path, dataset.discrete, features)


if __name__ == '__main__':
    main(sys.argv[1:])
