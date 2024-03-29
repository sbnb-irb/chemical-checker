import os
import sys
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from chemicalchecker.database import Molecule
from chemicalchecker.util.mol2svg import Mol2svg


converter = Mol2svg()


def draw(inchikey, inchi, mol_path):
    # get isomeric smiles
    mol = Chem.rdinchi.InchiToMol(inchi)[0]
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    mol_file = os.path.join(mol_path, "2d.mol")
    with open(mol_file, "w") as f:
        try:
            f.write(Chem.MolToMolBlock(mol))
        except Exception as ex:
            print(str(ex))
            f.write(Chem.MolToMolBlock(mol, kekulize=False))
    converter.mol2svg(mol_path)
    os.remove(mol_file)
    mol_svg = os.path.join(mol_path, '2d.svg')
    if not os.path.isfile(mol_svg):
        raise Exception("Could not draw %s in path %s" % (key, mol_svg))


task_id = sys.argv[1]
filename = sys.argv[2]
MOLECULES_PATH = sys.argv[3]
inputs = pickle.load(open(filename, 'rb'))
iks = inputs[task_id]


missing_svg = list()
missing_paths = list()
missing_keys = list()
for key in tqdm(iks, desc='check 2d.svg'):
    mol_path = os.path.join(MOLECULES_PATH, key[0:2], key[2:4], key)
    if not os.path.isdir(mol_path):
        missing_paths.append(mol_path)
        missing_keys.append(key)
    if not os.path.exists(os.path.join(mol_path, "2d.svg")):
        missing_svg.append(key)
        missing_keys.append(key)

print("Missing paths:", len(missing_paths))
print("Missing svgs:", len(missing_svg))
missing_keys = list(set(missing_keys))

if len(missing_keys) == 0:
    print('All molecules already present, nothing to do.')
    sys.exit()

mappings_inchi = Molecule.get_inchikey_inchi_mapping(missing_keys)

for key, inchi in tqdm(mappings_inchi.items(), desc='generate 2d.svg'):
    mol_path = os.path.join(MOLECULES_PATH, key[0:2], key[2:4], key)
    if not os.path.exists(mol_path):
        original_umask = os.umask(0)
        os.makedirs(mol_path, 0o775)
        os.umask(original_umask)
    draw(str(key), str(inchi), mol_path)

"""
attempts = 20
while attempts > 0:
    try:
        props = pubchempy.get_properties(
            ['IsomericSMILES', 'InChIKey'], missing_keys, "inchikey")
        break
    except Exception as e:
        print("Connection failed to REST Pubchem API. Retrying...")
        time.sleep(4)
        attempts -= 1

if attempts == 0:
    print("Failed to get data from pubchem")
    for key, inchi in mappings_inchi.items():

        key_path = os.path.join(
            MOLECULES_PATH, key[0:2], key[2:4], key)

        draw(str(key), None, str(inchi), key_path)

        if not os.path.exists(key_path + "/2d.svg"):
            raise Exception(
                "Molecular plot for inchikey " + key + " not present.")
else:

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
"""
