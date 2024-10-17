from __future__ import division
import sys
import os
import h5py
import shutil
import numpy as np
import bisect
import pickle
import uuid
import collections

from rdkit import Chem
from rdkit.Chem import AllChem

from chemicalchecker.util.mol2svg import Mol2svg
from chemicalchecker.database import Dataset, Molecule
from chemicalchecker.core import ChemicalChecker

converter = Mol2svg()

# Variables

cutoff = 0.001

# Functions

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
    else:
        svg = open( mol_svg, 'r' ).read().replace('\n', ' ')
        os.remove(mol_svg)
        shutil.rmtree( mol_path, ignore_errors=True)
        return smiles, svg

def weights(coords, coord_idxs):
    idxs = [coord_idxs[c] for c in coords]
    return np.mean(W[idxs, :][:, idxs], axis=1) + 1e-6


def get_props(cc, iks):
    sign0 = cc.get_signature("sign0", "full", "A5.001")
    _, props = sign0.get_vectors(iks, include_nan=True)
    feat_names = sign0.get_h5_dataset('features')
    return props, feat_names


def get_scores(i, cc, coords_obs, vals_obs, vals_pred, dataset_pairs, cut_idx,
               coord_idxs):
    # Fetch data
    w_obs = weights(coords_obs, coord_idxs)
    coords_prd = coord_idxs.keys()
    w_prd = weights(coords_prd, coord_idxs)
    # Popularity
    popu = (np.sum(w_obs) - min_popu) / (max_popu - min_popu)
    if popu > 1:
        popu = 1.
    if popu < 0:
        popu = 0.
    # Singularity
    v = []
    for c in coords_obs:
        v += [max(0, vals_obs[c][i])]
    if np.sum(v) == 0:
        sing = 0.
    else:
        sing = np.average(np.array(v), weights=w_obs)
    # Mappability
    v = []
    for c in coords_prd:
        v += [max(0, vals_pred[c][i])]
    if np.sum(v) == 0:
        mapp = 0.
    else:
        mapp = np.average(np.array(v), weights=w_prd)
    return popu, sing, mapp


# get script arguments
task_id = sys.argv[1]
filename = sys.argv[2]
consensus = sys.argv[3]
output_path = sys.argv[4]
CC_ROOT = sys.argv[5]
temp_dir = sys.argv[6]

# input is a chunk of universe inchikey
iks = pickle.load(open(filename, 'rb'))[task_id]

# read the consensus file
# FIXME where and how is it generated??
with h5py.File(consensus, "r") as hf:
    W = 1. - hf["Kn"][:]
    W[np.diag_indices(W.shape[0])] = 0.
    data = hf["coords"][:]
    if hasattr(data.flat[0], 'decode'):
        coord_idxs = dict((c.decode(), i) for i, c in enumerate(data))
    else:
        coord_idxs = dict((c, i) for i, c in enumerate(data))
max_popu = np.sum(weights(
    ["%s%d" % (j, i) for j in ["A", "B", "C", "D", "E"] for i in [1, 2, 3, 4, 5]], coord_idxs))
min_popu = np.sum(weights(["A%d" % i for i in [1, 2, 3, 4, 5]], coord_idxs))


# for each molecule which spaces are available in sign1?
cc = ChemicalChecker(CC_ROOT)
metric_obs = None
cut_idx = None
map_coords_obs = collections.defaultdict(list)
dataset_pairs = {}
for ds in Dataset.get(exemplary=True):
    dataset_pairs[ds.coordinate] = ds.dataset_code
    if metric_obs is None:
        sign1 = cc.get_signature("sign1", "reference", ds.dataset_code)
        metric_obs = sign1.get_h5_dataset('metric')
    sign1 = cc.get_signature("sign1", "full", ds.dataset_code)
    if cut_idx is None:
        cut_idx = bisect.bisect_left(sign1.PVALRANGES, cutoff)
    keys = sign1.unique_keys
    for ik in iks:
        if ik in keys:
            map_coords_obs[ik] += [ds.coordinate]

# get below cutoff distances to be normalized (not sure)
vals_obs = dict()
vals_pred = dict()
for coord in coord_idxs.keys():
    sign1 = cc.get_signature("sign1", "reference", dataset_pairs[coord])
    bg_vals_obs = sign1.background_distances(
        metric_obs[0])["distance"][cut_idx]
    similars = cc.get_signature("neig1", "full", dataset_pairs[coord])
    _, distances = similars.get_vectors(
        iks, include_nan=True, dataset_name='distances')
    masked = np.where(distances <= bg_vals_obs, distances, 0.0)
    sums = np.sum(masked, axis=1) - 1  # WHY -1?
    vals_obs[coord] = sums / (len(distances[0]) - 1)

    sign3 = cc.get_signature("sign3", "reference", dataset_pairs[coord])
    bg_vals_pred = sign3.background_distances("cosine")["distance"][cut_idx]
    similars = cc.get_signature("neig3", "full", dataset_pairs[coord])
    _, distances = similars.get_vectors(
        iks, include_nan=True, dataset_name='distances')
    masked = np.where(distances <= bg_vals_pred, distances, 0.0)
    sums = np.sum(masked, axis=1) - 1
    vals_pred[coord] = sums / (len(distances[0]) - 1)

# write scores
outfile = os.path.join(output_path, str(uuid.uuid4()))
mappings_inchi = Molecule.get_inchikey_inchi_mapping(iks)
props, feat_names = get_props(cc, iks)
mw_idx = np.argwhere(feat_names == '1').ravel()
ro5_idx = np.argwhere(feat_names == '15').ravel()
qed_idx = np.argwhere(feat_names == '17').ravel()
with open(outfile, "w") as f:
    for i, ik in enumerate(iks):
        print(i, ik)
        s = get_scores(i, cc, map_coords_obs[ik], vals_obs, vals_pred,
                       dataset_pairs, cut_idx, coord_idxs)
        
        inchi = mappings_inchi[ik]
        mol_path = os.path.join( temp_dir, ik )
        if( not os.path.isdir( mol_path ) ):
            os.mkdir( mol_path )
        smiles, svg = draw( str(ik), str(inchi), mol_path)
        
        start_pos = inchi.find('/') + 1
        end_pos = inchi.find('/', start_pos)
        formula = inchi[start_pos:end_pos]
        prop = props[i]
        f.write("%s\t%s\t%.3g\t%.3g\t%.3g\t%.2f\t%.3g\t%.3g\t%s\t%s\n" %
                ( ik, formula, s[0], s[1], s[2], prop[mw_idx], prop[ro5_idx], prop[qed_idx], smiles, svg ) )
