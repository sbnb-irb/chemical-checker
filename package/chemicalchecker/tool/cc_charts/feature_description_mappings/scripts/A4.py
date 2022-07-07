import pickle
import pandas as pd
import os

def create_mapping(src_file, dst_folder):
    from rdkit.Chem import MACCSkeys
    a4_dict = {str(key): value for key, value in MACCSkeys.smartsPatts.items()}

    os.makedirs(dst_folder, exist_ok=True)

    dest = os.path.join(dst_folder, 'A4')

    with open(dest, 'wb') as fh:
        pickle.dump(a4_dict, fh)
        
if __name__ == "__main__":
    
    from chemicalchecker.util import Config
    
    cfg = os.environ['CC_CONFIG']
    
    cc_path = Config(cfg).PATH.CC_REPO

    dst_folder = os.path.join(cc_path, 'package/scripts/feature_description_mappings')
    
    create_mapping(None, dst_folder)
        
