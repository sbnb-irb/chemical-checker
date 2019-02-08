
# Imports

import h5py
import numpy as np
import sys, os
import collections
import pybel
from e3fp import pipeline
from rdkit import Chem as sChem
from rdkit.Chem import Descriptors, rdMolDescriptors
import time
import timeout_decorator

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
import Psql
from rdkit.Chem import AllChem as Chem

# Functions

root = os.path.dirname(os.path.realpath(__file__))

@timeout_decorator.timeout(100,use_signals=False)
def fprints_from_inchi(inchi, inchikey, confgen_params={}, fprint_params={}, save=False):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
    
        if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11: return None
    
        smiles = Chem.MolToSmiles(mol)
        return pipeline.fprints_from_smiles(smiles, inchikey, confgen_params, fprint_params, save)


# Main

if __name__ == '__main__':
    
    dbname = sys.argv[1]
    param = sys.argv[2]
    ik,v = param.split("-----")  
    
    
   
    params = pipeline.params_to_dicts(root+"/../files/defaults.cfg")
    
    #v = inchikey_inchi[ik]
    try:
        fps = fprints_from_inchi(v, ik, params[0], params[1])
    except Exception as inst:
        print 'Timeout inchikey: ' + ik
        fps = None
        #raise e
    if not fps: 
        Psql.query("INSERT INTO fp3d (inchikey, raw) VALUES ('%s', NULL) ON CONFLICT DO NOTHING" % (ik), dbname)  
    else:
        s = ",".join([str(x) for fp in fps for x in fp.indices])
        Psql.query("INSERT INTO fp3d (inchikey, raw) VALUES ('%s', '%s') ON CONFLICT DO NOTHING" % (ik, s), dbname)  
