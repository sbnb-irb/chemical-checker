import sys
import os
import collections
import h5py

from chemicalchecker.util import logged, get_parser, save_output, features_file
from chemicalchecker.database import Dataset


##########################################################################

# Write the methods to preprocess the dataset here

##########################################################################


@logged
def main(args):

    args = get_parser().parse_args(args)

    dataset_code = '<DATASET_CODE>'

    dataset = Dataset.get(dataset_code)

    map_files = {}

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths
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
