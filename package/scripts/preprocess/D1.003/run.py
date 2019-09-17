import sys
import os
import collections
import h5py
import csv

from chemicalchecker.util import logged, get_parser, save_output, features_file
from chemicalchecker.database import Dataset, Molrepo

from cmapPy.pandasGEXpress import parse
import numpy as np

##########################################################################

# Write the methods to preprocess the dataset here

##########################################################################

def pertid_to_meta(map_files):
    filenames = [map_files["GSE92742_Broad_LINCS_pert_info"]]
    pertid_meta = collections.defaultdict(set)
    for filename in filenames:
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter = "\t")
            h = next(reader)
            pert_id_idx = h.index("pert_id")
            pert_iname_idx = h.index("pert_iname")
            pert_type_idx  = h.index("pert_type")
            for r in reader:
                pertid_meta[r[pert_id_idx]].update([(r[pert_iname_idx], r[pert_type_idx])])
    return pertid_meta

def parse_molrepo():
    lincs_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("lincs")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        lincs_inchikey[molrepo.src_id] = molrepo.inchikey
    return lincs_inchikey

@logged
def main(args):

    which_datasources = ["touchstone_conn_SUMMLY"]

    args = get_parser().parse_args(args)

    dataset_code = 'D1.003'

    dataset = Dataset.get(dataset_code)

    map_files = {}

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.
    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path + "/" + ds.filename

    #pertid_meta = pertid_to_meta(map_files)

    #lincs_molrepo = parse_molrepo()

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    features = None

    if args.method == "fit":
        from tqdm import tqdm
        main._log.info("Fitting")
        lincs_inchikey = parse_molrepo()
        pairs = collections.defaultdict(list)
        for ds in which_datasources:
            fn = map_files[ds]
            S  = parse.parse(fn)
            V = np.array(S.data_df)
            for i, row in tqdm(enumerate(np.array(S.row_metadata_df.index))):
                for j, col in enumerate(np.array(S.col_metadata_df.index)):
                    if V[i,j] < 90:
                        continue
                    else:
                        if V[i,j] < 95:
                            #v = 1
                            continue
                        else:
                            #v = 2
                            v = 1
                    if col not in lincs_inchikey: continue
                    pairs[(lincs_inchikey[col], row)] += [v]
        pairs = dict((k, np.max(v)) for k,v in pairs.iteritems())
        key_raw = collections.defaultdict(list)
        for k, v in pairs.iteritems():
            key_raw[str(k[0])] += [(str(k[1]), v)]
        features = sorted(set([x[0] for v in key_raw.values() for x in v]))

    if args.method == "predict":

        main._log.info("Predicting")

        with h5py.File(os.path.join(args.models_path, features_file)) as hf:
            features = hf["features"][:]

        # Read the data from the args.input_file
        # The entry point is available through the variable args.entry_point



###############################################################################

# Call the methods to preprocess the input data

###############################################################################

    main._log.info("Saving raw data")
    # To save the signature0, the data needs to be in the form of a dictionary
    # where the keys are inchikeys and the values are a list with the data.
    # For sparse data, we can use [word] or [(word, integer)].

    save_output(args.output_file, key_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
