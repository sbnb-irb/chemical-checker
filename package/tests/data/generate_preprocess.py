
import os
import pickle
import numpy as np
from glob import glob
from chemicalchecker import ChemicalChecker
from chemicalchecker.database import Dataset
from chemicalchecker.core import DataSignature

cc_root = '/aloy/web_checker/package_cc/2020_01/'
cc = ChemicalChecker(cc_root)
preprocess_path = cc_root + 'full/*/*/*.001/sign0/raw/preprocess.h5'
destination = '/aloy/home/mbertoni/code/chemical_checker/' +\
    'package/tests/data/preprocess/%s.h5'
mini_cc_uni_file = '/aloy/home/mbertoni/code/chemical_checker/' +\
    'package/tests/data/preprocess/mini_cc_universe.pkl'

# generate or load mini CC universe
if not os.path.isfile(mini_cc_uni_file):
    universe = set()
    for ds in Dataset.get():
        if not ds.derived:
            continue
        if not ds.essential:
            continue
        if ds.code >= 'D':
            print(ds.code)
            s0 = cc.get_signature('sign0', 'full', ds.code)
            universe.update(s0.unique_keys)
    mini_cc_uni = np.random.choice(list(universe), 10000, replace=False)
    mini_cc_uni = mini_cc_uni[np.argsort(mini_cc_uni)]
    pickle.dump(mini_cc_uni, open(mini_cc_uni_file, 'wb'))
mini_cc_uni = pickle.load(open(mini_cc_uni_file, 'rb'))

# make filtered copy of preprocessed files
for file in sorted(glob(preprocess_path)):
    ds = file[44:46]
    s = DataSignature(file)
    print('IN  ', ds, s.info_h5)
    if os.path.isfile(destination % ds):
        n = DataSignature(destination % ds)
        print('OUT ', ds, n.info_h5)
    else:
        if 'keys' in s.info_h5:
            mask = np.isin(
                list(s.keys), list(mini_cc_uni), assume_unique=True)
            s.make_filtered_copy(destination % ds, mask, True)
        elif 'pairs' in s.info_h5:
            pairs = s.get_h5_dataset('pairs')
            mask = np.isin(list(pairs[:, 0]), list(mini_cc_uni))
            s.make_filtered_copy(destination % ds, mask, True)
        n = DataSignature(destination % ds)
        print('OUT ', ds, n.info_h5)
