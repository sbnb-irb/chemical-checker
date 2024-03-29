import sys
import os
import argparse
import collections
import h5py
import numpy as np
import networkx as nx
import logging

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.util import psql
from chemicalchecker.core.preprocess import Preprocess


features_file = "features.h5"
chembl_dbname = 'chembl'
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
entry_point_full = "cell"

# Functions

def fetch_chembl():

    # ChEMBL - InChIKey

    molrepos = Molrepo.get_by_molrepo_name("chembl")
    chemblid_inchikey = {}
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        chemblid_inchikey[molrepo.src_id] = molrepo.inchikey

    # Query

    cur = psql.qstring('''
    SELECT md.chembl_id, ass.assay_id, ass.src_id, act.standard_relation, act.standard_type, act.standard_value, act.standard_units, cd.cellosaurus_id

    FROM molecule_dictionary md, activities act, assays ass, target_dictionary td, cell_dictionary cd

    WHERE

    act.molregno = md.molregno AND
    act.assay_id = ass.assay_id AND
    ass.tid = td.tid AND
    td.target_type = 'CELL-LINE' AND
    ass.cell_id IS NOT NULL AND
    cd.cell_id = ass.cell_id AND
    cd.cellosaurus_id IS NOT NULL AND
    ass.src_id = 1 AND act.standard_flag = 1
    ''', chembl_dbname)

    R = []
    for r in cur:
        if r[0] not in chemblid_inchikey:
            continue
        if r[2] == 1:  # Literature
            if r[3] != '=':
                continue
            if (r[4], r[6]) in [("IC50", "nM"), ("GI50", "nM"), ("LC50", "nM"), ("LD50", "nM"), ("CC50", "nM"), ("EC50", "nM")]:
                if r[5] < 1000:
                    R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
            elif (r[4], r[6]) == ("Activity", "%"):
                if r[5] <= 50:
                    R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
            elif (r[4], r[6]) == ("GI", "%"):
                if r[5] >= 50:
                    R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
        elif r[2] == 7:  # PubChem
            R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
        else:
            continue

    return R


def parse_cellosaurus(R, cellosaurus_obo):

    f = open(cellosaurus_obo, "r")
    O = f.read().split("[Term]\n")
    f.close()

    G = nx.DiGraph()

    for term in O:
        term = term.split("\n")
        for l in term:
            if l[:4] == "id: ":
                child = l.split("id: ")[1]
                G.add_node(child)
                
            if( l.find("xref: PubChem_Cell_line") != -1 ): # Increase vocabulary using cross references to other database synonyms cell lines
                child = l.split("PubChem_Cell_line:")[1]
                if( child.startswith('CVCL_') ):
                    G.add_node(child)
                
            if "relationship: derived_from " in l:
                parent = l.split("derived_from ")[1].split(" !")[0]
                if parent is not None:
                    G.add_edge(parent, child)
            # if "relationship: originate_from_same_individual_as" in l:
            #    parent = l.split("originate_from_same_individual_as ")[1].split(" !")[0]
            #    G.add_edge(parent, child)

    # Add a root *
    nodes = list(G.nodes())
    for n in nodes:
        if not nx.ancestors(G, n):
            if n is not None:
                G.add_edge(n, "*")

    # Cell hierarchy

    cells = set([r[-1] for r in R])
    cell_hier = collections.defaultdict(set)

    for cell in cells:
        if( cell in nodes ):
            for c in nx.all_simple_paths(G, cell, "*"):
                if c == "*":
                    continue
                for x in c:
                    cell_hier[cell].update([x])

    return cell_hier


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

    features = None

    if args.method == "fit":

        cellosaurus_obo = os.path.join(
            map_files["cellosaurus"], "cellosaurus.obo")

        main._log.info("Fetch from ChEMBL")
        R = fetch_chembl()

        main._log.info("Reading Cellosaurus")
        cell_hier = parse_cellosaurus(R, cellosaurus_obo)

        inchikey_raw_temp = collections.defaultdict(set)
        for r in R:
            for c in cell_hier[r[-1]]:
                inchikey_raw_temp[r[0]].update([r[-1]])

        inchikey_raw = {k: list(v) for k,v in inchikey_raw_temp.items()}
        del inchikey_raw_temp

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
                if len(items) == 2:
                    inchikey_raw[items[0]] += [items[1]]

    main._log.info("Saving raws")

    Preprocess.save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
