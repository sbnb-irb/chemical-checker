'''
DeepCodex signatures.

Test file snippet:

entry_point = proteins:

s1  Q05209  -1
s2  Q9UBU9  -1
s3  P32243  1
'''
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
    inchikey_raw = collections.defaultdict(list)

    save_output(args.output_file, inchikey_raw, args.method,
                args.models_path, dataset.discrete, features)

if __name__ == '__main__':
    main(sys.argv[1:])