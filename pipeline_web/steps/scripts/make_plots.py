from __future__ import division
import sys
import os
import pickle
from rdkit.Chem import AllChem
from rdkit import Chem
from chemicalchecker.database import Molecule
from chemicalchecker.tool.mol2svg import Mol2svg


converter = Mol2svg()


def draw(inchikey, inchi, path):

    if not os.path.exists(path):
        original_umask = os.umask(0)
        os.makedirs(path, 0o775)
        os.umask(original_umask)

    mol = Chem.rdinchi.InchiToMol(inchi)[0]
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

print("Generating molecule plots for " +
      str(len(missing_keys)) + " molecules")


mappings_inchi = Molecule.get_inchikey_inchi_mapping(missing_keys)

for key, inchi in mappings_inchi.iteritems():

    key_path = os.path.join(
        MOLECULES_PATH, key[0:2], key[2:4], key)

    draw(str(key), str(inchi), key_path)

    if not os.path.exists(key_path + "/2d.svg"):
        raise Exception(
            "Molecular plot for inchikey " + key + " not present.")
