# Re-generating the reference datasets after shrinking the universe of A spaces
from chemicalchecker import ChemicalChecker
from chemicalchecker.util.remove_near_duplicates import RNDuplicates
import os, h5py

os.environ['CC_CONFIG'] = "/opt/chemical_checker/setup/users/cc_config_nico.json"

cc= ChemicalChecker("/aloy/web_checker/package_cc/2020_01/")

Aspaces = ('A1.001','A2.001', 'A3.001', 'A4.001', 'A5.001', 'B4.002')
for sp in Aspaces:
    sign0_full = cc.get_signature('sign0', 'full', sp)
    sign0_ref = cc.get_signature('sign0', 'reference', sp)
    sign0_ref.clean()
    rnd = RNDuplicates(cpu=10)
    rnd.remove(sign0_full.data_path, save_dest=sign0_ref.data_path)
    
    with h5py.File(sign0_full.data_path, "r") as hf:
        features = hf["features"][:]
    with h5py.File(sign0_ref.data_path, 'a') as hf:
        hf.create_dataset('features', data=features)