import sys
import os
import argparse
import collections
import h5py
import numpy as np

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset

features_file = "features.h5"


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


def save_output(output_file, inchikey_raw, method, models_path, discrete, features):

    keys = []

    if discrete:
        words = set()
        for k in sorted(inchikey_raw.keys()):
            keys.append(str(k))
            words.update(inchikey_raw[k])

        if features is not None:
            orderwords = features
        else:
            orderwords = list(words)
            orderwords.sort()
        raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
        wordspos = {k: v for v, k in enumerate(orderwords)}

        categ = False

        if isinstance(inchikey_raw[keys[0]], tuple):
            categ = True

        for i, k in enumerate(keys):
            for word in inchikey_raw[k]:
                if categ:
                    raws[i][wordspos[word[0]]] = word[1]
                else:
                    raws[i][wordspos[word]] = 1

        with h5py.File(output_file, "w") as hf:
            hf.create_dataset("keys", data=np.array(keys))
            hf.create_dataset("V", data=raws)
            hf.create_dataset("features", data=np.array(orderwords))

        if method == "fit":
            with h5py.File(os.path.join(models_path, features_file), "w") as hf:
                hf.create_dataset("features", data=np.array(orderwords))

    else:

        for k in inchikey_raw.keys():
            keys.append(str(k))
        keys = np.array(keys)
        inds = keys.argsort()
        data = []

        for i in inds:
            data.append(inchikey_raw[keys[i]])

        with h5py.File(output_file, "w") as hf:
            hf.create_dataset("keys", data=keys[inds])
            hf.create_dataset("V", data=np.array(data))


##########################################################################

# Write the methods to preprocess the dataset here

##########################################################################


@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = '<DATASET_CODE>'

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    features = None

    # The entry point is available through the variable args.entry_point

    if args.method == "fit":

        main._log.info("Fitting")

        # Read the data from the datasources

    if args.method == "predict":

        main._log.info("Predicting")

        # If we are in the predict method and the dataset is discrete, we need to read the features from
        # the features_file that was saved in the models_path
        if dataset.discrete:
            with h5py.File(os.path.join(args.models_path, features_file)) as hf:
                features = hf["features"][:]

        # Read the data from the args.input_file


###############################################################################

# Call the methods to preprocess the input data

###############################################################################

    main._log.info("Saving raw data")
    # To save the signature0, the data needs to be in the form of a dictionary
    # where the keys are inchikeys and the values are a list with the data
    # If the data is discrete, it can be categorized or not.
    # For categorized data, the list is a list of tuples as (word,integer)
    # For not categorized data, the list is just a list of words
    inchikey_raw = collections.defaultdict(list)

    save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
