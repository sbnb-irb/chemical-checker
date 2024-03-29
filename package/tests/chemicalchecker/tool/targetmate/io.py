"""Input/output function utilities"""

import pandas as pd
import numpy as np
import random
import csv
import uuid
import pickle
import os

from .utils.chemistry import read_molecule

# Utility functions

def reader(data):
    # Data is a file
    if type(data) == str:
        with open(data, "r") as f:
            for r in csv.reader(f, delimiter = "\t"):
                yield r
    # Data is a list
    else:
        for r in data:
            yield r

def filter_validity(data, valid_inchikeys, only_molecules=False):
    valid_inchikeys = set(valid_inchikeys)
    if not only_molecules:
        idx       = data[0]
        smiles    = data[1]
        inchikey  = data[2]
        activity  = data[3]
        srcid     = data[4]
        idx_      = []
        smiles_   = []
        inchikey_ = []
        activity_ = []
        srcid_    = []
        for i, ik in enumerate(inchikey):
            if ik not in valid_inchikeys: continue
            idx_      += [idx[i]]
            smiles_   += [smiles[i]]
            inchikey_ += [inchikey[i]]
            activity_ += [activity[i]]
            srcid_    += [srcid[i]]
        data = (idx_, smiles_, inchikey_, activity_, srcid_)
    else:
        smiles    = data[0]
        inchikey  = data[1]
        smiles_   = []
        inchikey_ = []
        for i, ik in enumerate(inchikey):
            if ik not in valid_inchikeys: continue
            smiles_   += [smiles[i]]
            inchikey_ += [inchikey[i]]
        data = (smiles_, inchikey_)
    return data

def read_data(data,
              smiles_idx=None,
              inchi_idx=None,
              inchikey_idx=None,
              activity_idx=None,
              srcid_idx=None,
              standardize=False,
              use_inchikey=False,
              valid_inchikeys=None,
              ):
    """Read data.

    Args:
        data(str or list of tuples):
        smiles_idx: Tuple or column index where smiles is specified (default=None).

        inchikey_idx: Column where the inchikey is present (default=None).
        activity_idx: Tuple or column index where activity is specified (default=None).
        srcid_idx: Tuple or column index where the source id is specified (default=None).
        standardize(bool): Standardize structures.
        use_inchikey(bool): Use inchikey directly (default=False)

    Returns:
        InputData instance.
    """
    molecule   = []
    activity = []
    srcid    = []
    idx      = []
    inchikey = []
    if not use_inchikey:
        if smiles_idx is None and inchi_idx is None:
            raise Exception("smiles_idx or inchi_idx needs to be specified")
        j=0
        for r in reader(data):
            if smiles_idx is not None:
                molecule_idx = smiles_idx
                inchi = False
            elif inchi_idx is not None:
                molecule_idx = inchi_idx
                inchi =True
            molec = r[molecule_idx]
            m = read_molecule(molec, standardize, inchi=inchi)
            if not m: continue
            idx      += [j]
            molecule   += [m[1]]
            inchikey += [m[0]]
            j+=1
            if activity_idx is not None:
                activity += [float(r[activity_idx])]
            else:
                activity += [None]
            if srcid_idx is not None:
                srcid += [r[srcid_idx]]
            else:
                srcid += [None]
    else:
        inchi = False
        if inchikey_idx is None:
            raise Exception("inchikey_idx needs to be specified")
        for i, r in enumerate(reader(data)):
            idx += [i]
            inchikey += [r[inchikey_idx]]
            if smiles_idx is not None:
                molecule += [r[smiles_idx]]
            else:
                molecule += [None]
            if activity_idx is not None:
                activity += [float(r[activity_idx])]
            else:
                activity += [None]
            if srcid_idx is not None:
                srcid += [r[srcid_idx]]
            else:
                srcid += [None]
    data = (idx, molecule, inchikey, activity, srcid)
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys)
    if not inchi:
        return InputData(data)
    else:
        return InputData(data, moleculetype='InChI')


def reassemble_activity_sets(act, inact, putinact, valid_inchikeys=None, inchi = False):
    """Reassemble activity sets, relevant when sampling from Universe"""
    data = []
    for x in list(act):
        data += [(x[1],  1, x[0], x[-1])]
    for x in list(inact):
        data += [(x[1], -1, x[0], x[-1])]
    n = np.max([x[0] for x in data]) + 1
    if not inchi:
        for i, x in enumerate(list(putinact)):
            data += [(i + n, 0, x[0], x[-1])]
    else:
        for i, x in enumerate(list(putinact)):
            data += [(i + n, 0, x[1], x[-1])]

    idx      = []
    molecule   = []
    inchikey = []
    activity = []
    srcid    = []
    for d in data:
        idx      += [d[0]]
        molecule   += [d[2]]
        inchikey += [d[3]]
        activity += [d[1]]
    data = (idx, molecule, inchikey, activity, srcid)
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys)
    if not inchi:
        return InputData(data)
    else:
        return InputData(data, moleculetype='InChI')


def read_molecules_from_multiple_data(data_list, molecule_idx, standardize=False, sort=True, valid_inchikeys=None, inchi =False, **kwargs):
    """Read molecules from multiple datasets"""
    molecules_ = set()
    for data in data_list:
        mols = []
        for r in reader(data):
            mols += [r[molecule_idx]]
        molecules_.update(molecules_)
    molecules_ = list(molecules_)
    molecules   = []
    inchikey = []
    for mol in molecules_:
        m = read_molecule(mol, standardize, inchi = inchi)
        if not m: continue
        molecules += [m[1]]
        inchikey += [m[0]]
    data = (molecules, inchikey)
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys, only_molecules=True)
    return MoleculeData(data, sort=sort)

def read_smiles_from_multiple_data(data_list, smiles_idx, standardize=False, sort=True, valid_inchikeys=None, **kwargs):
    """Read smiles from multiple datasets"""
    smiles_ = set()
    for data in data_list:
        smis = []
        for r in reader(data):
            smis += [r[smiles_idx]]
        smiles_.update(smis)
    smiles_ = list(smiles_)
    smiles   = []
    inchikey = []
    for smi in smiles_:
        m = read_smiles(smi, standardize)
        if not m: continue
        smiles += [m[1]]
        inchikey += [m[0]]
    data = (smiles, inchikey)
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys, only_molecules=True)
    return SmilesData(data, sort=sort)

def read_inchi_from_multiple_data(data_list, inchi_idx, standardize=False, sort=True, valid_inchikeys=None, **kwargs):
    """Read inchi from multiple datasets"""
    inchi_ = set()
    for data in data_list:
        smis = []
        for r in reader(data):
            smis += [r[smiles_idx]]
        inchi_.update(smis)
    inchi_ = list(inchi_)
    inchi   = []
    inchikey = []
    for inch in inchi_:
        m = read_molecule(inch, standardize, inchi= True)
        if not m: continue
        inchi += [m[1]]
        inchikey += [m[0]]
    data = (inchi, inchikey)
    if valid_inchikeys is not None:
        data = filter_validity(data, valid_inchikeys, only_molecules=True)
    return InchiData(data, sort=sort)



def data_to_disk(data, tmp_dir):
    data.on_disk(tmp_dir)


def data_from_disk(data):
    if type(data) is str:
        with open(data, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        return data


# Classes
class InputData:
    """A simple input data class"""

    def __init__(self, data=None, moleculetype='SMILES'):
        """Initialize input data class"""
        self.tag = str(uuid.uuid4())
        if data is not None:
            idx   = np.array(data[0])
            order = np.argsort(idx)
            self.idx      = idx[order]
            self.molecule   = np.array(data[1])[order]
            self.inchikey = np.array(data[2])[order]
            if data[3] == []:
                self.activity = None
            else:
                self.activity = np.array(data[3])[order]
                try:
                    self.activity.astype(np.float)
                except:
                    raise Exception("Activities are not of numeric type!")
            if data[4] == []:
                self.srcid = None
            else:
                self.srcid = np.array(data[4])[order]
            self.moleculetype =moleculetype
    def __iter__(self):
        for idx, v in self.as_dataframe().iterrows():
            yield v

    def __getitem__(self, idxs):
        data = InputData()
        data.idx = self.idx[idxs]
        if self.activity is not None:
            data.activity = self.activity[idxs]
        else:
            data.activity = None
        data.molecule = self.molecule[idxs]
        data.inchikey = self.inchikey[idxs]
        if self.srcid is not None:
            data.srcid = self.srcid[idxs]
        else:
            data.srcid = None
        return data

    def as_dataframe(self):
        df = pd.DataFrame({
            "idx": self.idx,
            "activity": self.activity,
            "molecule": self.molecule,
            "inchikey": self.inchikey,
            "srcid": self.srcid
            })
        return df

    @staticmethod
    def sel(ary, idxs):
        if idxs is None:
            return ary
        if ary is None:
            return None
        return ary[idxs]

    def as_dict(self, idxs):
        res = {
            "idx": self.sel(self.idx, idxs),
            "activity": self.sel(self.activity, idxs),
            "molecule": self.sel(self.molecule, idxs),
            "inchikey": self.sel(self.inchikey, idxs),
            "srcid": self.sel(self.srcid, idxs)
        }
        return res

    def shuffle(self, inplace=True):
        """Shuffle data"""
        ridxs = [i for i in range(0, len(self.idx))]
        random.shuffle(ridxs)
        if inplace:
            self.idx = self.idx[ridxs]
            if self.activity is not None: self.activity = self.activity[ridxs]
            self.molecule = self.molecule[ridxs]
            self.inchikey = self.inchikey[ridxs]
            if self.srcid is not None: self.srcid = self.srcid[ridxs]
        else:
            return self.__getitem__[ridxs]

    def on_disk(self, tmp_dir):
        data_path = os.path.join(tmp_dir, "data_objects")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
        data_path = os.path.join(data_path, self.tag)
        with open(data_path, "wb") as f:
            pickle.dump(self, f, protocol=4)
            #Added by Paula: Protocol 4 allows pickling of larger data objects. Consider only using this protocol in case of large object (see difference in file size) 30/08/20
        return data_path


class MoleculeData(object):
    """A simple molecule data container"""

    def __init__(self, data, sort):
        """Initialize"""
        molecule   = np.array(data[0])
        inchikey = np.array(data[1])
        if sort:
            order     = np.argsort(molecule)
            molecule    = molecule[order]
            inchikey  = inchikey[order]
        self.molecule   = molecule
        self.inchikey = inchikey

class SmilesData(object):
    """A simple smiles data container"""

    def __init__(self, data, sort):
        """Initialize"""
        smiles   = np.array(data[0])
        inchikey = np.array(data[1])
        if sort:
            order     = np.argsort(smiles)
            smiles    = smiles[order]
            inchikey  = inchikey[order]
        self.smiles   = smiles
        self.inchikey = inchikey

class InchiData(object):
    """A simple Inchi data container"""

    def __init__(self, data, sort):
        """Initialize"""
        inchi   = np.array(data[0])
        inchikey = np.array(data[1])
        if sort:
            order     = np.argsort(inchi)
            inchi    = inchi[order]
            inchikey  = inchikey[order]
        self.inchi   = inchi
        self.inchikey = inchikey


class Prediction(object):
    """A simple prediction class"""

    def __init__(self, datasets, y_true, y_pred_calibrated, y_pred_uncalibrated, is_ensemble, weights=None):
        self.is_ensemble = is_ensemble
        self.datasets = datasets
        self.y_true = y_true
        if is_ensemble:
            self.y_pred_ens = y_pred
            self.weights = weights
            self.y_pred = self.metapredict(self.datasets)
        else:
            self.y_pred = y_pred_calibrated
            self.y_pred_uncalib = y_pred_uncalibrated


    def metapredict(self, datasets=None):
        """Metapredict using a double weighting scheme"""
        if not self.is_ensemble:
            return self.y_pred
        if not datasets:
            datasets = self.datasets
        else:
            datasets = sorted(set(self.datasets).intersection(datasets))
        idxs = [self.datasets.index(ds) for ds in datasets]
        if self.weights is None:
            weights = np.ones(len(idxs))
        else:
            weights = np.array([self.weights[ds] for ds in datasets])
        weights = weights / np.sum(weights)
        v = self.y_pred_ens[:,:,idxs]
        pweights = np.max(v, axis = 1)
        y_pred = np.zeros((v.shape[0], v.shape[1]))
        for i in range(0, y_pred.shape[0]):
            pw = pweights[i] / np.sum(pweights[i])
            w  = weights*pw
            w  = w / np.sum(w)
            for j in range(0, y_pred.shape[1]):
                vals = v[i,j]
                y_pred[i,j] = np.average(vals, weights = w)
        return y_pred


class Explanation(object):
    """Shapley explanation results"""
    def __init__(self, datasets, shaps, is_ensemble):
        self.is_ensemble = is_ensemble
        self.datasets = datasets
        self.shaps = shaps
