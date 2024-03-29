import numpy as np

from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Scaffolds import MurckoScaffold

from standardiser import standardise

from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file


def maccs_matrix(smiles):
    fps = np.zeros((len(smiles), 167)).astype(int)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        fpon = sorted(MACCSkeys.GenMACCSKeys(mol).GetOnBits())
        if not fpon:
            continue
        fps[i, fpon] = 1
    return fps


def morgan_matrix(smiles, radius = 2, nBits = 2048):
    smiles = list(smiles)
    fps = np.zeros((len(smiles), nBits), dtype = int)
    for i, smi in enumerate(smiles):
        try:
            arr = np.zeros((0,), dtype=int)
            mol = Chem.MolFromSmiles(smi)
            fp  = Chem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        except:
            arr = np.full(nBits, np.nan)
        fps[i] = arr
    return fps


def morgan_arena(smiles, file_name):
    smiles = [(smi, i) for i, smi in enumerate(smiles)]
    create_db_file(smiles, file_name, 'Morgan', {'radius': 2, 'nBits': 1024})
    arena = FPSim2Engine(file_name)
    return arena


def generate_scaffold(smiles, include_chirality=False):
    """Bemis-Murcko scaffold"""
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def load_morgan_arena(file_name):
    arena = FPSim2Engine(file_name)
    return arena


def similarity_matrix(smiles, arena, arena_size):
    similarities = np.zeros((len(smiles), arena_size))
    for i, smi in enumerate(smiles):
        results = arena.similarity(smi, 0.0, n_workers=1)
        for j, score in results:
            similarities[i, j] = score
    return similarities


def read_molecule(molec, standardize, min_mw=100, max_mw=1000, inchi = False):
    if not inchi:
        try:
            mol = Chem.MolFromSmiles(molec)
            if standardize:
                mol = standardise.run(mol)
        except Exception:
            return None
        if not mol:
            return None
        mw = MolWt(mol)
        if mw < min_mw or mw > max_mw:
            return None
        ik = Chem.rdinchi.InchiToInchiKey(Chem.rdinchi.MolToInchi(mol)[0])
        molec = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(molec)
        if not mol:
            return None
        return ik, molec
    else:
        try:
            mol = Chem.MolFromInchi(molec)
            if standardize:
                mol = standardise.run(mol)
        except Exception:
            return None
        if not mol:
            return None
        mw = MolWt(mol)
        if mw < min_mw or mw > max_mw:
            return None
        ik = Chem.rdinchi.InchiToInchiKey(Chem.rdinchi.MolToInchi(mol)[0])
        molec =Chem.rdinchi.MolToInchi(mol)[0]
        mol = Chem.MolFromInchi(molec)
        if not mol:
            return None
        return ik, molec



def read_smiles(smi, standardize, min_mw=100, max_mw=1000):
    try:
        mol = Chem.MolFromSmiles(smi)
        if standardize:
            mol = standardise.run(mol)
    except Exception:
        return None
    if not mol:
        return None
    mw = MolWt(mol)
    if mw < min_mw or mw > max_mw:
        return None
    ik = Chem.rdinchi.InchiToInchiKey(Chem.rdinchi.MolToInchi(mol)[0])
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None
    return ik, smi
