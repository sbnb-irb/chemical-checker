import os
import argparse
import h5py
import numpy as np


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


def save_output(output_file, inchikey_raw, method, models_path, discrete, features, chunk=2000):

    keys = []

    if discrete:

        categ = False

        for k, v in inchikey_raw.items():
            if len(v) > 0:
                if isinstance(v[0], tuple):
                    categ = True
                break

        words = set()
        for k in sorted(inchikey_raw.keys()):
            keys.append(str(k))
            if categ:
                for word in inchikey_raw[k]:
                    words.add(word[0])
            else:
                words.update(inchikey_raw[k])

        if features is not None:
            orderwords = features
        else:
            orderwords = list(words)
            orderwords.sort()

        with h5py.File(output_file, "w") as hf:
            hf.create_dataset("keys", data=np.array(keys))
            hf.create_dataset("V", (len(keys), len(orderwords)), dtype='i8')
            hf.create_dataset("features", data=np.array(orderwords))

        raws = np.zeros((chunk, len(orderwords)), dtype=np.int8)
        wordspos = {k: v for v, k in enumerate(orderwords)}
        index = 0

        for i, k in enumerate(keys):
            for word in inchikey_raw[k]:
                if categ:
                    raws[index][wordspos[word[0]]] = word[1]
                else:
                    raws[index][wordspos[word]] = 1
            index += 1

            if index == chunk or i == len(keys) - 1:
                end = i + 1
                if index != chunk:
                    chunk = index
                with h5py.File(output_file, "r+") as hf:
                    dataset = hf["V"]
                    dataset[end - chunk:end] = raws[:chunk]

                raws = np.zeros((chunk, len(orderwords)), dtype=np.int8)
                index = 0

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
            hf.create_dataset("features", data=np.array(features))

        if method == "fit":
            with h5py.File(os.path.join(models_path, features_file), "w") as hf:
                hf.create_dataset("features", data=np.array(features))
