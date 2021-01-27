# And use the data calculator class to get all A spaces preprocessed data
# Inspired from the cc pipeline
# And use the data calculator class to get all A spaces preprocessed data
# Inspired from the cc pipeline
import collections
from chemicalchecker.util.parser import DataCalculator
from chemicalchecker.core.preprocess import Preprocess


# type of data, space_name, format size
# Note A3 contains both caffold and framework each encoded on 1024 bits, so we have 2048 total ex: f647,c92 
# A4: supposed to be 166 groups but 152 keys are present in raw 2020_01 preprocess.h5
# A5: 
data_calculators = {
    'morgan_fp_r2_2048': 'A1',
    'e3fp_3conf_1024': 'A2' 
    'murcko_1024_cframe_1024': 'A3'
    'maccs_keys_166': 'A4',
    'general_physchem_properties': 'A5',
    #'chembl_target_predictions_v23_10um':'A5',
}


# TASK: Calculate data (defined for universe)
def calculate_data_fn(type_data, tmpdir, dict_inchikey_inchi):

    print("Calculating data for " + type_data)
    parse_fn = DataCalculator.calc_fn(type_data)

    for i,chunk in enumerate(parse_fn(dict_inchikey_inchi)):
        print("CHUNK",i)
        print(chunk)
        if len(chunk) == 0:
            continue
        else:
            return chunk
        

def calculate_mol_properties(dict_inchikey_inchi)
result=dict()
        
for data_calc,space  in data_calculators.items():
    result[space]=calculate_data_fn(type_data=data_calc, tmpdir='tmp', dict_inchikey_inchi=inchikey2inchi_dict)

    # dictionary {'A1': [{'inchikey': 'ASXBYYWOLISCLQ-UHFFFAOYSA-N', 'raw': ..raw_string}, {}...]}
    return result

# Trying to save the result of A1 (from A1)

def import_A_features(Aspace):
    pass


def create_h5_from_inchikeys_inchi(result,Aspace):

    # ACTS must be a list of tuples (inchikey, raw string)

    features = import_A_features(Aspace)
    outputfile="A1_absent.h5"
    method='fit'
    model_path='tmp'
    discrete=True


    ACTS = ACTS= [(dic['inchikey'],dic['raw']) for dic in result['A1']]
    # Add a dummy line to get all the features (2048)
    ACTS.append('DUMMY', ','.join(list(range(1,2049))))

    RAW = collections.defaultdict(list)
    # NS: ACTS contains the dense format for raw data
    for k in ACTS:
        if features is None:
            vals = [str(t) for t in k[1].split(",")]
        else:
            vals = [str(t) for t in k[1].split(",") if str(t) in features]
        RAW[str(k[0])] = vals



    # NS: Preprocess.save_output will then convert the dense format into binary data
    Preprocess.save_output(outputfile, RAW, method,
                model_path, discrete, features, features_int=True)