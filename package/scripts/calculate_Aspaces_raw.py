# Nico: Jan 2021
# Uses the data calculator class to get all A spaces preprocessed data
# Inspired from the cc pipeline
import os
import collections
from chemicalchecker.util.parser import DataCalculator
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.util.parser.features_A_spaces import fetch_features_A

# type of data, space_name, format size
# Note A3 contains both caffold and framework each encoded on 1024 bits, so we have 2048 total ex: f647,c92 
# A4: supposed to be 166 groups but 152 keys are present in raw 2020_01 preprocess.h5 

Aspaces= ('A1', 'A2', 'A3', 'A4', 'A5')

def calculate_data_fn(space, dict_inchikey_inchi):
    """
    Launch the (chemistry) data calculation for one type of space

    Returns: a list containing the molecular properties in dense format
    ex: [{'inchikey': 'ASXBYYWOLISCLQ-UHFFFAOYSA-N', 'raw': ..raw_string}, {}...]

    Arguments:
    - space (str): either A1, A2, A3, A4, A5, A5
    - dict_inchikey_inchi (dict): mapping of the molecules to calculate properties from

    """

    data_calculators = {
        'A1': 'morgan_fp_r2_2048',
        'A1': 'e3fp_3conf_1024', 
        'A1': 'murcko_1024_cframe_1024',
        'A1': 'maccs_keys_166',
        'A1': 'general_physchem_properties',
        #'chembl_target_predictions_v23_10um':'A5',
    }

    type_data= data_calculators.get(space.upper(),None)
    assert type_data is not None, "Space "+space+" is not part of the CC A spaces!"

    print("Calculating data for " + type_data)
    parse_fn = DataCalculator.calc_fn(type_data)

    for i,chunk in enumerate(parse_fn(dict_inchikey_inchi)):
        print("CHUNK",i)
        print(chunk)
        if len(chunk) == 0:
            continue
        else:
            return chunk

    return None
        

def calculate_mol_properties(dict_inchikey_inchi):
    """
    Calls calculate_data_fn for all spaces

    Returns: a dict containing the molecular properties in dense format for all spaces

    Arguments:
    - space (str): either A1, A2, A3, A4, A5, A5
    - dict_inchikey_inchi (dict): mapping of the molecules to calculate properties from
    """
    result=dict()
            
    for space in Aspaces:
        result[space]=calculate_data_fn(space, dict_inchikey_inchi=inchikey2inchi_dict)

        # dictionary {'A1': [{'inchikey': 'ASXBYYWOLISCLQ-UHFFFAOYSA-N', 'raw': ..raw_string}, {}...]}
        return result

# Trying to save the result of A1 (from A1)


def create_h5_from_inchikeys_inchi(dict_inchikey_inchi, outDir='tmp'):
    """
    Create the preprocessed h5 data files
    for spaces A1 to A5 for the molecules specified in the input inchikey_inchi dictionary

    Here we cannot use the predict method from Preprocess.save_output since it requires connection to the database
    whereas we work locally.
    """

    if not os.path.exists(outDir):
        try:
            os.makedirs(outDir)
        except Exception as e:
            print("ERROR: cannot create output directory for ")
            print(e)
            return

    outputfiles= {space: os.path.join(outDir,space+'_outsideUniv.h5') for space in Aspaces}
    method='fit'
    model_path='tmp'  # Dummy tmp folder
    

    # Compute the raw properties

    all_properties= calculate_mol_properties(dict_inchikey_inchi)
    all_features= fetch_features_A()

    # Here we need to preprocess the dense format according to which space we have
    for space in Aspaces:

        # Dense chemical properties for all spaces
        raw_prop= all_properties.get(space,None)
        features= all_features.get(space, None)
        feature_int= False if space in ('A3','A5') else True
        discrete= False if space == 'A5' else True

        if raw_prop is None or features is None:
            print("WARNING, space",space,"properties or features is None, space skipped")
            continue

        # format the dense chemical properties in tuples (inchikey, raw_str)
        ACTS= [(dic['inchikey'],dic['raw']) for dic in result['A1']]

        if space in ('A1','A2','A3'):

            # Copied from the end of A1 preprocess script
            RAW = collections.defaultdict(list)
            for k in ACTS:
                if features is None:
                    vals = [str(t) for t in k[1].split(",")]
                else:
                    vals = [str(t) for t in k[1].split(",") if str(t) in features]
                RAW[str(k[0])] = vals


        elif space == 'A4':
            for k in ACTS:
                if k[1] == '' or k[1] is None or k[0] is None:
                    continue
                if features is None:
                    vals = [str(t) for t in k[1].split(",")]
                else:
                    vals = [str(t) for t in k[1].split(",") if str(t) in features]
                RAW[str(k[0])] = vals


        elif space == 'A5':
            sigs = collections.defaultdict(list)
            words = []
            first = True
            for k in ACTS:
                if k[1] is None:
                    continue
                data = k[1].split(",")
                vals = []
                for d in data:
                    ele = d.split("(")
                    if first:
                        words.append(str(ele[0]))
                    vals.append(float(ele[1][:-1]))
                sigs[str(k[0])] = vals
                first = False

            RAW=sigs

        # NS: Preprocess.save_output will then convert the dense format into binary data
        Preprocess.save_output(outputfiles[space], RAW, method, model_path, discrete, features, features_int=feature_int)