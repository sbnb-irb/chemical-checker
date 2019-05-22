'''
PharmacoDB.

Test file snippet:

entry_point = genesets like in admice_signatures_noaggregation_updw.gmt
'''
import sys
import os
import collections
import h5py
import csv
import logging
from chemicalchecker.util import logged, get_parser, save_output, features_file
from chemicalchecker.database import Dataset, Molrepo

from chemicalchecker.util import psql

# Variables

db_name = "pharmacodb"
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]
min_pvalue = 0.01
top_genes = 250

# Entry points

entry_point_full = "genesets"

# Functions

def fetch_drug_gene_correlations():
    cmd = '''
          SELECT t1.drug_id, t2.gene_name, t1.estimate, t1.pvalue
            FROM gene_drugs t1, genes t2
              WHERE t1.gene_id = t2.gene_id
                AND t1.pvalue < 0.01
                AND t1."mDataType" = 'mRNA'
          '''
    R = psql.qstring(cmd, db_name)
    return R


def parse_molrepo():
    pharmacodb_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("pharmacodb")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        pharmacodb_inchikey[molrepo.src_id] = molrepo.inchikey
    return pharmacodb_inchikey

# Main


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):

    args = get_parser().parse_args(args)

    dataset = Dataset.get(dataset_code)

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    features = None

    if args.method == "fit":

        main._log.info("Fitting")

        pharmacodb_inchikey = parse_molrepo()

        R = fetch_drug_gene_correlations()

        key_raw = collections.defaultdict(set)
        for r in R:
            src_id = "pharmacodb_%d" % r[0]
            if r[2] < 0:
                direction = "-1"
            else:
                direction = "1"
            feat = "%s(%s)" % (r[1], direction)
            if src_id not in pharmacodb_inchikey:
                continue
            key_raw[pharmacodb_inchikey[src_id]].update([(feat, r[3])])

        key_raw = dict((k, [y[0] for y in sorted(v, key=lambda x: x[1])[
                       :top_genes]]) for k, v in key_raw.iteritems())
        features = sorted(set([x for v in key_raw.itervalues() for x in v]))

    if args.method == "predict":

        main._log.info("Predicting")

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features = hf["features"][:]
            features_set = set(features)

        if args.entry_point is None:
            args.entry_point = entry_point_full

        key_raw = collections.defaultdict(set)
        with open(args.input_file, "r") as f:
            for r in csv.reader(f, delimiter="\t"):
                if not r[1]:
                    key = r[0]
                else:
                    key = r[1]
                up = r[2].split(",")
                dw = r[3].split(",")
                feats = ["%s(1)" % x for x in up] + ["%s(-1)" % x for x in dw]
                feats = [x for x in feats if x in features_set]
                key_raw[key].update(feats)
        key_raw = dict((k, list(v)) for k, v in key_raw.iteritems())

    main._log.info("Saving raw data")

    save_output(args.output_file, key_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
