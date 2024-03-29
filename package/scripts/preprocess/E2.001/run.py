import os
import sys
import csv
import h5py
import logging
import argparse
import numpy as np
import collections
import networkx as nx
from tqdm import tqdm

from chemicalchecker.util import psql
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo
from chemicalchecker.core.preprocess import Preprocess


# Variables
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
features_file = "features.h5"
chembl_dbname = 'chembl'
# Parse arguments
entry_point_full = "atc"


def parse_repodb(repodb, umls2mesh, IND=None):

    if IND is None:
        IND = collections.defaultdict(list)

    # DrugBank molrepo
    dbid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("drugbank")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        dbid_inchikey[molrepo.src_id] = molrepo.inchikey

    # Read UMLS to MESH
    umls_mesh = collections.defaultdict(set)
    f = open(umls2mesh, "r")
    for l in f:
        print("SHERLOCK l-->",l)
        if l[0] == "#":
            continue
       # l = l.rstrip("\n").split("|")
        l = l.rstrip("\n").split("\t")
        if l[0] == "diseaseId":
            continue
        if l[2] == "MSH":
            umls_mesh[l[0]].update([l[3]])
    f.close()

    # Parse RepoDB
    f = open(repodb, "r")
    f.readline()
    for l in csv.reader(f):
        if l[1] not in dbid_inchikey:
            continue
        if l[3] not in umls_mesh:
            continue
        for meshid in umls_mesh[l[3]]:
            if l[5] == "Withdrawn" or l[5] == "NA" or l[5] == "Suspended":
                continue
            if l[5] == "Approved":
                phase = 4
            else:
                if "Phase 3" in l[6]:
                    phase = 3
                else:
                    if "Phase 2" in l[6]:
                        phase = 2
                    else:
                        if "Phase 1" in l[6]:
                            phase = 1
                        else:
                            if "Phase 0" in l[6]:
                                phase = 0
                            else:
                                continue
            IND[(dbid_inchikey[l[1]], meshid)] += [ float(phase) ]
    f.close()
    return IND


def parse_chembl(IND=None):

    if IND is None:
        IND = collections.defaultdict(list)

    # ChEMBL molrepo
    chemblid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("chembl")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        chemblid_inchikey[molrepo.src_id] = molrepo.inchikey

    # Query ChEMBL
    R = psql.qstring('''
    SELECT md.chembl_id, di.mesh_id, di.max_phase_for_ind

    FROM molecule_dictionary md, drug_indication di

    WHERE

    md.molregno = di.molregno''', chembl_dbname)

    for r in R:
        if r[0] not in chemblid_inchikey:
            continue
        IND[(chemblid_inchikey[r[0]], r[1])] += [ float(r[2]) ]

    IND = dict((k, np.max(v)) for k, v in IND.items())

    return IND


def include_mesh(ctd_diseases, IND):

    G = nx.DiGraph()
    with open(ctd_diseases, "r") as f:
        for l in f:
            if l[0] == "#":
                continue
            l = l.rstrip("\n").split("\t")
            disid = l[1]
            pardisids = l[4].split("|")
            if pardisids == [""]:
                pardisids = ["ROOT"]
            for pardisid in pardisids:
                if pardisid is not None:
                    G.add_edge(pardisid, disid)

    classIND = collections.defaultdict(list)
    for k, v in tqdm(IND.items()):
        classIND[k] = [ float(v) ]
        node = "MESH:" + k[1]
        if node not in G:
            continue
        path = nx.all_simple_paths(G, "ROOT", node)
        dis = [d.split("MESH:")[1] for p in path for d in p if "MESH:" in d]
        for d in dis:
            classIND[(k[0], d)] += [ float(v) ]

    #for k, v in classIND.items():
    #    print(k, v)
    #    print( '\t----', np.max(v) )
    classIND = dict( (k, np.max(v)) for k, v in classIND.items())

    return classIND


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):
    # Reading arguments and getting datasource
    args = Preprocess.get_parser().parse_args(args)
    main._log.debug("Running preprocess. Saving output to %s",
                    args.output_file)
    dataset = Dataset.get(dataset_code)

    map_files = {}

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.
    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path

    # decide entry point, if None use default
    if args.entry_point is None:
        args.entry_point = entry_point_full

    # main FIT section
    if args.method == "fit":

        # fetch Indication from RepoDB and ChEMBL
        main._log.info("Parsing RepoDB.")
        repodb = os.path.join(map_files["repodb"], "repodb.csv")
        umls2mesh = os.path.join(map_files["disease_mappings"],
                                 "disease_mappings.tsv")
        IND = parse_repodb(repodb, umls2mesh)

        main._log.info("Parsing ChEMBL")
        IND = parse_chembl(IND)

        # include MeSH hierarchy
        main._log.info("Including MeSH hierarchy")
        ctd_diseases = os.path.join(
            map_files["CTD_diseases"], "CTD_diseases.tsv")
        classIND = include_mesh(ctd_diseases, IND)

        inchikey_raw = collections.defaultdict(list)
        for k, v in classIND.items():
            inchikey_raw[k[0]] += [k[1] + "(" + str(v) + ")"]

        # features will be calculated later
        features = None
        features_list = None

    # main PREDICT section
    if args.method == "predict":

        # load features (saved at FIT time)
        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features_list = hf["features"][:]
            features = set(features_list)

        # read input file format: INK MESH (Phase)
        inchikey_raw = collections.defaultdict(list)
        with open(args.input_file) as f:
            for l in f:
                items = l.rstrip().split("\t")
                if len(items) == 2:
                    val = 2  # default value
                else:
                    val = int(items[2])
                inchikey_raw[items[0]] += [items[1] + "(" + str(val) + ")"] 

    # save raw values
    main._log.info("Saving raw data.")

    Preprocess.save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features_list)

if __name__ == '__main__':
    main(sys.argv[1:])
