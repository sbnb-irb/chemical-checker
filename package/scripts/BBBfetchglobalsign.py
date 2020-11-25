# script to fetch global signatures for the BBB predictor
# and save them as numpy array


import os, glob,sys

import numpy as np
import pandas as pd



# Martino's signaturizer
#from signaturizer import Signaturizer
from chemicalchecker import ChemicalChecker

os.environ["CC_CONFIG"] = "/home/nsoler/CODE/chemical_checker/setup/cc_config.json"


# Deepchem (includes MoleculeNet)
import rdkit.Chem as Chem
#from rdkit.Chem.Scaffolds import MurckoScaffold 
#from deepchem.splits import ScaffoldSplitter


DATA=sys.argv[1] # csv file with the data
SAVEDIR="/home/nsoler/NOTEBOOKS/BBBpredictor/2020NPY"

#----------------

def smiles2inchiKey(sm):
    try:
        mol = Chem.MolFromSmiles(sm)
        inchi = Chem.MolToInchiKey(mol)
    except:
        return 'none'
    else:
        return inchi



# CC object
cc =ChemicalChecker("/aloy/web_checker/package_cc/2020_01/")
universe=set(cc.universe)


# Loading the downloaded data
df = pd.read_csv(DATA)
df['num'] = df.num.apply(lambda x: int(x))
df.set_index('num',inplace=True)

# Adding the inchiKey col
df['inchiKey']= df.smiles.apply(lambda x:smiles2inchiKey(x))


co=0
for i,row in df.iterrows():
    print("MOLECULE",i)
    try:
        #mol = cc.get_molecule(row.inchiKey, 'inchikey')
        res = row.inchiKey in cc.universe
        
        # If a molecule is in the checker, attempt to recover the signature
        if res==True:
            sign3 = cc.get_global_signature(row.inchiKey, str_type='inchikey')
            co+=1
            np.save(os.path.join(SAVEDIR,str(i)),sign3)
            print('saved',os.path.join(SAVEDIR,str(i)+'.npy'))
    except Exception as e:
        print("problem",e)
        
    print('\n')
    #df.loc[i,'signature'] = cc.get_global_signature(row.smiles, str_type='smiles')
print(co)
