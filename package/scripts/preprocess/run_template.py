import sys
import os
import collections
import h5py
import logging

from chemicalchecker.util import logged, get_parser, save_output, features_file
from chemicalchecker.database import Dataset

dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:]

##########################################################################

# Write the methods to preprocess the dataset here

##########################################################################


@logged(logging.getLogger("[ pre-process %s ]" % dataset_code))
def main(args):

    args = get_parser().parse_args(args)

    dataset = Dataset.get(dataset_code)

    map_files = {}

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.
    for ds in dataset.datasources:
        map_files[ds.datasource_name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    features = None

    if args.method == "fit":

        main._log.info("Fitting")

        # Read the data from the datasources

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
    key_raw = collections.defaultdict(list)

    save_output(args.output_file, key_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])
