import os
import sys
import argparse
import networkx as nx
import collections
import h5py
import numpy as np
from sklearn.preprocessing import normalize

from chemicalchecker.util import logged, HotnetNetwork
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo

features_file = "features.h5"
pcomms_file = "pcomms.tsv"
entry_point_full = "metabolites_neighbors"


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
# Variables


def prepare_hotnet_input(outdir, all_binary_sif):

    # Read ChEBI molrepo

    # Get molecules
    molrepos = Molrepo.get_by_molrepo_name("chebi")
    chebi_inchikey = {}
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        chebi_inchikey[molrepo.src_id] = molrepo.inchikey

    # Read graph

    G = nx.Graph()

    with open(all_binary_sif, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[0] in chebi_inchikey and l[2] in chebi_inchikey:
                ik1 = chebi_inchikey[l[0]]
                ik2 = chebi_inchikey[l[2]]
                G.add_edge(ik1, ik2)

    with open(os.path.join(outdir, pcomms_file), "w") as f:
        for e in G.edges():
            f.write("%s\t%s\n" % (e[0], e[1]))


# Hotnet-specific functions

class Sm:

    def __init__(self, A, names):
        self.A = A
        self.names = names

    def get_profile(self, n):
        if n not in self.names:
            return None
        return self.A[:, self.names.index(n)]


def load_matrix(net_folder):
    f = open(net_folder + "/idx2node.tsv", )
    names = [l.rstrip("\n").split("\t")[1] for l in f]
    f.close()
    f = h5py.File(net_folder + "/similarity_matrix.h5")
    A = f['PPR'].value
    f.close()
    return Sm(A, names)


def network_impact(sm, node_scores):
    P, S = [], []
    for n, s in node_scores.iteritems():
        prof = sm.get_profile(sm, n)
        if not prof:
            continue
        P += [prof]
        S += [s]
    P = np.array(P)


def scale_by_non_diagonal_max(A):
    S = A[:]
    np.fill_diagonal(S, 0.)
    for j in range(A.shape[1]):
        S[j, j] = np.max(S[:, j])
    S = normalize(S, norm="max", axis=0)
    return S


def read_hotnet_output(outdir):

    sm = load_matrix(outdir)
    sm.A = scale_by_non_diagonal_max(sm.A)

    profiles = collections.defaultdict(list)
    for ik in sm.names:
        P = []
        prof = sm.get_profile(ik)
        for i in range(len(prof)):
            w = int(prof[i] * 10)  # Convert weights to integers
            if w == 0:
                continue
            P += [(sm.names[i], w)]
        profiles[ik] += P

    return profiles

# Parse arguments


@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = 'C2.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    features = None

    if args.method == "fit":

        all_binary_sif = os.path.join(map_files["molpathways"], "PathwayCommons11.All.hgnc.sif")
        main._log.info("Preparing HotNet input")

        prepare_hotnet_input(args.models_path, all_binary_sif)

        main._log.info("Running HotNet")

        HotnetNetwork.prepare(os.path.join(
            args.models_path, pcomms_file), args.models_path)

        main._log.info("Reading HotNet output")

        inchikey_raw = read_hotnet_output(args.models_path)

    if args.method == "predict":

        inchikey_raw = collections.defaultdict(list)

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                if len(items) < 3:
                    inchikey_raw[items[0]] += [(items[1], 5)]
                else:
                    inchikey_raw[items[0]] += [(items[1], int(items[2]))]

    main._log.info("Saving raws")

    keys = []
    words = set()
    for k in sorted(inchikey_raw.keys()):

        keys.append(str(k))
        words.update([x[0] for x in inchikey_raw[k]])

    if features is not None:
        orderwords = features_list
    else:
        orderwords = list(words)
        orderwords.sort()
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_raw[k]:
            raws[i][wordspos[word[0]]] += word[1]

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))

    if args.method == "fit":
        with h5py.File(os.path.join(args.models_path, features_file), "w") as hf:
            hf.create_dataset("features", data=np.array(orderwords))

if __name__ == '__main__':
    main(sys.argv[1:])
