from __future__ import division
import sys
import os
import pickle
import pubchempy
import time
from rdkit.Chem import AllChem
from rdkit import Chem
from chemicalchecker.database import Molecule
from chemicalchecker.tool.mol2svg import Mol2svg


converter = Mol2svg()


def draw(inchikey, smiles, inchi, path):

    if not os.path.exists(path):
        original_umask = os.umask(0)
        os.makedirs(path, 0o775)
        os.umask(original_umask)

    if smiles is None:
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
    else:
        mol = Chem.MolFromSmiles(smiles)
    # convert to smiles
    # use openeye to obtain canonical smiles (isomeric)
    # read mol again from smiles
    AllChem.Compute2DCoords(mol)
    with open(path + "/2d.mol", "w") as f:
        try:
            f.write(Chem.MolToMolBlock(mol))
        except:
            f.write(Chem.MolToMolBlock(mol, kekulize=False))

    converter.mol2svg(path)

    os.remove("%s/2d.mol" % path)

task_id = sys.argv[1]
filename = sys.argv[2]
MOLECULES_PATH = sys.argv[3]
inputs = pickle.load(open(filename, 'rb'))
iks = inputs[task_id]


missing_keys = list()

for key in iks:

    key_path = os.path.join(
        MOLECULES_PATH, key[0:2], key[2:4], key)

    if not os.path.exists(key_path + "/2d.svg"):
        missing_keys.append(key)

if len(missing_keys) > 0:

    print("Generating molecule plots for " +
          str(len(missing_keys)) + " molecules")

    mappings_inchi = Molecule.get_inchikey_inchi_mapping(missing_keys)

    attempts = 20
    while attempts > 0:

        try:

            props = pubchempy.get_properties(
                ['IsomericSMILES', 'InChIKey'], missing_keys, "inchikey")
            break
        except Exception as e:
            print ("Connection failed to REST Pubchem API. Retrying...")
            time.sleep(4)
            attempts -= 1

    if attempts == 0:
        print("Failed to get data from pubchem")
        continue

    data_set = set(missing_keys)

    for prop in props:

        key = prop["InChIKey"]
        smile = None
        if key not in data_set:
            continue
        inchi = mappings_inchi[str(key)]
        if "IsomericSMILES" in prop and prop["IsomericSMILES"] is not None:
            smile = prop["IsomericSMILES"]

        key_path = os.path.join(
            MOLECULES_PATH, key[0:2], key[2:4], key)

        draw(str(key), smile, str(inchi), key_path)

        if not os.path.exists(key_path + "/2d.svg"):
            raise Exception(
                "Molecular plot for inchikey " + key + " not present.")
