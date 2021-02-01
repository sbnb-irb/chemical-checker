# Nico: Jan 2021
# Uses the data calculator class to get all A spaces preprocessed data
# Inspired from the cc pipeline
import os, json
import collections

from chemicalchecker.util.parser import DataCalculator
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.util.parser import fetch_features_A
from chemicalchecker.util.parser import Converter
from chemicalchecker.core.chemcheck import ChemicalChecker

# type of data, space_name, format size
# Note A3 contains both caffold and framework each encoded on 1024 bits, so we have 2048 total ex: f647,c92 
# A4: supposed to be 166 groups but 152 keys are present in raw 2020_01 preprocess.h5 


class Aspaces_prop_calculator(object):


    def __init__(self, inchikey_list=None, output_directory='tmp', inchikey2inchi_map=None):
        """
        Class to create the preprocessed h5 data files
        for spaces A1 to A5 for the molecules specified in the input inchikey_inchi dictionary
        It can then create a sign0 object for each class and try to predict sign1 and 2

        Here we cannot use the predict method from Preprocess.save_output since it requires connection to the database
        whereas we work locally.

        Arguments:
            - inchikey_list: list or set of inchikeys, (inchis will the be recovered)
            - inchikey2inchi_map (dict): mapping between inchikeys and inchis entered
            - outDir (str): where to put the output h5 files

        # Note: you have to entered the inchikeys by either inchikey_list (then the inchis will be retrieved automatically)
                or by a dictionary with inchikeys as keys and inchis as values. Or by a path to a json file.

        """

        
        if inchikey_list is None and inchikey2inchi_map is None:
            print("Please enter the inchikeys to process by either inchikey_list or inchikey2inchi_map arguments")
            return

        self.inchikey_list= inchikey_list

        self.data_calculators = {
             'A1': 'morgan_fp_r2_2048',
             'A2': 'e3fp_3conf_1024', 
             'A3': 'murcko_1024_cframe_1024',
             'A4': 'maccs_keys_166',
             'A5': 'general_physchem_properties'}

        self.Aspaces= ('A1', 'A2', 'A3', 'A4', 'A5')
        self.converter= Converter()

        if inchikey2inchi_map is None:
            self.dict_inchikey_inchi= self.inchikey2inchi()

        elif type(inchikey2inchi_map) is str:
            # json file
            try:
                with open(inchikey2inchi_map) as f:
                    self.dict_inchikey_inchi=json.load(f)
            except:
                print("Please provide a dictionary or a path to a json file for inchikey2inchi_map, currently (",inchikey2inchi_map,")")
                return
        else:
            #mapping
            self.dict_inchikey_inchi= inchikey2inchi_map

        self.outDir= output_directory
        if not os.path.exists(self.outDir):
            try:
                os.makedirs(self.outDir)
            except Exception as e:
                print("ERROR: cannot create output directory for ",self.outDir)
                print(e)

        # Our cc instance
        # Put this CC instance inside our output directory
        cc_directory= os.path.join(self.outDir,'cc_absent')

        if not os.path.exists(cc_directory):
            os.makedirs(cc_directory)

        self.cc=ChemicalChecker(cc_root= cc_directory,dbconnect=False)


    def save_inchikey2inchi(self, outputFile="inchikey2inchis.json"):

        with open(outputFile, 'w') as f:
            json.dump(self.dict_inchikey_inchi,f)
            print("Mapping inchikeys / InChIs saved as",outputFile)

    def inchikey2inchi(self):
        # Try with the Molecule class, otherwise use Converter (requires web)
        itWorks=False # no inchikey so far
        dict_inchikey_inchi=dict()

        setIn= set(self.inchikey_list)
        print("Recovering InChI for the ", len(setIn),"unique inchickeys entered")
        print("Please wait..")
        for ink in setIn:
            try:
                inchi= self.converter.inchikey_to_inchi(ink)[0]["standardinchi"]
                itWorks=True
                dict_inchikey_inchi[ink]=inchi
            except Exception as e:
                print("ERROR: ",e)
                dict_inchikey_inchi[ink]=None


        # If no inchi could be retrieved:
        if not itWorks:
            print("ERROR: no inchi could be retrieved from the input inchikey list")
            print("Please check your internet connection (required by rdkit)")
            print(self.inchikey_list)
            return None
        else:
            return dict_inchikey_inchi


    def calculate_data_fn(self, space):
        """
        Launch the (chemistry) data calculation for one type of space

        Returns: a list containing the molecular properties in dense format
        ex: [{'inchikey': 'ASXBYYWOLISCLQ-UHFFFAOYSA-N', 'raw': ..raw_string}, {}...]

        Arguments:
        - space (str): either A1, A2, A3, A4, A5, A5
        - dict_inchikey_inchi (dict): mapping of the molecules to calculate properties from

        """

        type_data= self.data_calculators.get(space,None)
        assert type_data is not None, "Space "+space+" is not part of the CC A spaces!"

        print("\nCalculating data for " + type_data)
        parse_fn = DataCalculator.calc_fn(type_data)

        for i,chunk in enumerate(parse_fn(self.dict_inchikey_inchi)):
            if len(chunk) == 0:
                continue
            else:
                return chunk

        return None
            

    def calculate_mol_properties(self):
        """
        Calls calculate_data_fn for all spaces

        Returns: a dict containing the molecular properties in dense format for all spaces

        Arguments:
        - space (str): either A1, A2, A3, A4, A5, A5
        - dict_inchikey_inchi (dict): mapping of the molecules to calculate properties from
        """
        result=dict()
                
        for space in self.Aspaces:
            result[space]=self.calculate_data_fn(space)

            # dictionary {'A1': [{'inchikey': 'ASXBYYWOLISCLQ-UHFFFAOYSA-N', 'raw': ..raw_string}, {}...]}
        return result

    # Trying to save the result of A1 (from A1)


    def create_h5(self):


        print("Retrieving InChI strings from the list of input InChIkeys")

        for i, (inchikey, inchi) in enumerate(self.dict_inchikey_inchi.items()):
            print(i+1,')',inchikey)
            print(inchi)
            print('\n')


        outputfiles= {space: os.path.join(self.outDir,space+'_outsideUniv.h5') for space in self.Aspaces}
        method='predict'
        model_path='tmp'  # Dummy tmp folder
        

        # Compute the raw properties

        all_properties= self.calculate_mol_properties()
        all_features= fetch_features_A()

        print('all_properties',all_properties)
        print('all_features',all_features)

        # Here we need to preprocess the dense format according to which space we have
        for space in self.Aspaces:

            # Dense chemical properties for all spaces
            raw_prop= all_properties.get(space,None)
            features= all_features.get(space, None)
            feature_int= False if space in ('A3','A5') else True
            discrete= False if space == 'A5' else True

            if raw_prop is None or features is None:
                print("WARNING, space",space,"properties or features is None, space skipped")
                continue

            # format the dense chemical properties in tuples (inchikey, raw_str)
            ACTS= [(dic['inchikey'],dic['raw']) for dic in all_properties[space]]

            if space in ('A1','A2','A3','A4'):

                # Copied from the end of A1 preprocess script
                RAW = collections.defaultdict(list)
                for k in ACTS:
                    # k[0] is the inchikey while k[1] is the raw string of chemical properties
                    if k[1] == '' or k[1] is None or k[0] is None:
                        print("\nWARNING (space ",space,": no property for inchikey",k[0],'\n')
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
            print("Saving the chemical properties for space", space, "into", outputfiles[space])
            Preprocess.save_output(outputfiles[space], RAW, method, model_path, discrete, features, features_int=feature_int)
            print("\n")

        # dict space: path to raw file
        return outputfiles

    def createSign0(self, dict_of_Aspaces_h5, sanitize=False):
        """
        Create sign0 from all raw A spaces h5 files created with create_h5_from_inchikeys_inchi
        Here we take in a list of 5 paths to raw data (A1 o A5) and return a cc instance that contains sign0 for these 5 spaces
        """

        # Now creating sign0 for each of the input raw files
        for space, fp in dict_of_Aspaces_h5.items():
            print("\nCalculating sign0 for space", space)
            sign0 = self.cc.get_signature('sign0', 'full',space)
            sign0.fit(data_file=fp,do_triplets=False, overwrite=True,sanitize=sanitize)

        # Then we can use this cc instance to predict sign1
        return self.cc


    def predictSign1(self):

        return self.cc

    def predictSign2(self):
        
        return self.cc

