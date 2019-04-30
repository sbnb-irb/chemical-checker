# Imports

import sys
import os
import collections
import h5py

from chemicalchecker.util import logged, get_parser, save_output, features_file
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo

from lxml import etree as ET
import csv

# Variables

features_file = "features.h5"

# Entry points

entry_point_full = "proteins"

# Functions

def read_data(file_path):

    # HMDB molrepo
    hmdbid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("hmdb")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        hmdbid_inchikey[molrepo.src_id] = molrepo.inchikey

    ns = "{http://www.hmdb.ca}"

    enzymes = collections.defaultdict(set)

    def fast_iter(context, func):
        for event, elem in context:
            yield func(elem)
            elem.clear()
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]
        del context

    def process_elem(elem):
        src_id   = elem.find(ns+"accession")
        prot_ass = elem.find(ns+"protein_associations")
        if prot_ass is None or src_id is None: return None, None
        src_id = src_id.text
        enzs   = set()
        for prot in prot_ass:
            acc = prot.find(ns+"uniprot_id")
            if acc is None: continue
            enzs.update([acc.text])
        if not enzs: return None, None
        return src_id, enzs

    context = ET.iterparse(file_path + "/hmdb_metabolites.xml", events = ("end", ), tag = ns+"metabolite")

    key_enzyme = collections.defaultdict(set)

    for src_id, enzs in fast_iter(context, process_elem):
        if src_id is None or enzs is None: continue
        if src_id not in hmdbid_inchikey: continue
        key_enzyme[hmdbid_inchikey[src_id]].update(enzs)

    return key_enzyme

# Main

@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = 'B2.002'

    dataset = Dataset.get(dataset_code)

    map_files = {}

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.
    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    features = None

    if args.method == "fit":

        main._log.info("Fitting")

        # Read the data from the datasources
        key_enzyme = read_data(map_files['hmdb_metabolites'])
        features = sorted(set([x for v in key_enzyme.values() for x in v]))

    if args.method == "predict":

        main._log.info("Predicting")

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features = hf["features"][:]
            features_set = set(features)

        # Read the data from the args.input_file
        # The entry point is available through the variable args.entry_point

        if args.entry_point is None:
            args.entry_point = entry_point_full

        key_enzyme = collections.defaultdict(set)

        with open(args.input_file, "r") as f:
            for r in csv.reader(f, delimiter = "\t"):
                if r[1] not in features_set: continue
                key_enzyme[r[0]].update([r[1]])

    main._log.info("Saving raw data")
    # To save the signature0, the data needs to be in the form of a dictionary
    # where the keys are inchikeys and the values are a list with the data.
    # For sparse data, we can use [word] or [(word, integer)].
    key_raw = dict((k, sorted(v)) for k,v in key_enzyme.iteritems())

    save_output(args.output_file, key_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
