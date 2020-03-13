import sys
import argparse
import os
import networkx as nx
import collections
import h5py
import numpy as np
import logging
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.core.preprocess import Preprocess
# Variables

Role = "CHEBI:50906"
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
entry_point_full = "terms"
features_file = "features.h5"
graph_file = "graph.gpickle"

def parse_chebi(chebi_obo):

    # Parse ChEBI graph

    CHEBI = nx.DiGraph()

    cuts = ["is_a: ",
            "relationship: is_conjugate_acid_of ",
            "relationship: is_conjugate_base_of ",
            "relationship: is_tautomer_of ",
            "relationship: is_enantiomer_of ",
            "relationship: has_role "]

    f = open(chebi_obo, "r")
    terms = f.read().split("[Term]\n")
    for term in terms[1:]:
        term = term.split("\n")
        chebi_id = term[0].split("id: ")[1]
        CHEBI.add_node(chebi_id)
        parents = []
        for l in term[1:]:
            for cut in cuts:
                if cut in l:
                    parent_chebi_id = l.split(cut)[1]
                    parents += [parent_chebi_id]
        parents = sorted(set(parents))
        for p in parents:
            CHEBI.add_edge(chebi_id, p)
    f.close()

    return CHEBI


def find_paths(CHEBI, inchikey_chebi):

    # Find paths

    inchikey_paths = {}
    for k, v in inchikey_chebi.items():
        path = set()
        for chebi_id in v:
            if chebi_id not in CHEBI.nodes():
                continue
            path.update([str(n) for p in nx.all_simple_paths(
                CHEBI, chebi_id, Role) for n in p])
        path = sorted(path)
        if not path:
            continue
        inchikey_paths[k] = path

    return inchikey_paths

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

    inchikey_chebi = collections.defaultdict(set)
    features = None

    if args.method == "fit":

        chebi_obo = os.path.join(map_files["chebi"], "chebi.obo")

        # Get molecules
        molrepos = Molrepo.get_by_molrepo_name("chebi")
        for molrepo in molrepos:
            if not molrepo.inchikey:
                continue
            inchikey_chebi[molrepo.inchikey].update([molrepo.src_id])

        main._log.info("Parsing the ChEBI ontology...")
        CHEBI = parse_chebi(chebi_obo)

        nx.write_gpickle(CHEBI, os.path.join(args.models_path, graph_file))

    if args.method == "predict":

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        CHEBI = nx.read_gpickle(os.path.join(args.models_path, graph_file))

        with open(args.input_file) as f:

            for l in f:
                items = l.rstrip().split("\t")
                if items[1] not in features:
                    continue
                inchikey_chebi[items[0]].update([items[1]])

    main._log.info("Finding paths")
    inchikey_paths = find_paths(CHEBI, inchikey_chebi)
    main._log.info("Saving raw data")

    Preprocess.save_output(args.output_file, inchikey_paths, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
