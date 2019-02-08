import sys
import argparse
import os
import networkx as nx
import collections
import h5py
import numpy as np

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
# Variables

Role = "CHEBI:50906"

# Functions


def parse_chebi(chebi_obo):

    # Get molecules

    inchikey_chebi = collections.defaultdict(set)
    molrepos = Molrepo.get_by_molrepo_name("chebi")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        inchikey_chebi[molrepo.inchikey].update([molrepo.src_id])

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
        inchikey_paths[k] = path

    return inchikey_paths

# Parse arguments


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
    return parser


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'C1.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    chebi_obo = os.path.join(map_files["chebi"], "chebi.obo")

    main._log.info("Parsing the ChEBI ontology...")
    inchikey_paths = parse_chebi(chebi_obo)

    main._log.info("Saving raw data")

    keys = []
    words = set()
    for k in sorted(inchikey_paths.keys()):
        keys.append(str(k))
        words.update(inchikey_paths[k])

    orderwords = list(words)
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_paths[k]:
            raws[i][wordspos[word]] = 1

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))


if __name__ == '__main__':
    main()
