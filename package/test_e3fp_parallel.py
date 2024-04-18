import time

from chemicalchecker.core.chemcheck import ChemicalChecker
from chemicalchecker.database.molecule import Molecule

from e3fp import pipeline
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from python_utilities.parallel import Parallelizer

def fprints_from_inchi(inchi, inchikey, confgen_params={}, fprint_params={}, save=False):
    #try:
    #inchi= inchi[0]
    #inchikey=inchi[1]
    print(inchi, inchikey, confgen_params, fprint_params, save)
   
    result = None
    if(inchi != None):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
        if(mol != None):
            if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11:
                result = None
            else:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                result = pipeline.fprints_from_smiles( smiles, inchikey, confgen_params, fprint_params, save )
        #except Exception:
        #    result = None
    if not result:
        result = {
            "inchikey": inchikey,
            "raw": result
        }
    else:
        s = ",".join([str(x) for fp in result for x in fp.indices])
        result = {
            "inchikey": inchikey,
            "raw": result
        }
    return result
    
cc = ChemicalChecker('/aloy/web_checker/package_2024_update/')
root = '/aloy/home/ymartins/Documents/cc_update/chemical_checker/package/chemicalchecker/util/parser/data/defaults.cfg'
params = pipeline.params_to_dicts(root )
mols = cc.get_signature('sign0', 'full', 'B5.001')
keys = mols.keys
ckeys = list(keys)[:50000] 
#ckeys = list(keys)
#ckeys = list(keys)[:50] 
inchikey_inchi = Molecule.get_inchikey_inchi_mapping(ckeys)
inchi_iter = ((inchi, key) for key, inchi in inchikey_inchi.items())

kwargs = {"confgen_params": params[0], "fprint_params": params[1] }
parallelizer = Parallelizer(parallel_mode="processes", num_proc=25)

start_time = time.time()
fprints_list=[]
fprints_list = parallelizer.run(fprints_from_inchi, inchi_iter, kwargs=kwargs) 
#for i in inchi_iter:
#    fprints_list.append( fprints_from_inchi(i, confgen_params=params[0], fprint_params=params[1] ) )
print(fprints_list[:3])
print( 'length: ', len(fprints_list))
print("--- %s seconds ---" % (time.time() - start_time))
