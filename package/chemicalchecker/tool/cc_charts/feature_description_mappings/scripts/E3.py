import pickle
import pandas as pd
import os

def create_mapping(src_file, dst_folder):

    df = pd.read_table(src_file, header=None)
    identifiers = df.loc[:, 3]

    descriptions = df.loc[:, 6]
    e3_dict = {i: d for i, d in zip(identifiers, descriptions)}

    os.makedirs(dst_folder, exist_ok=True)
    
    dest = os.path.join(dst_folder, 'E3')
    
    with open(dest, 'wb') as fh:
        pickle.dump(e3_dict, fh)
        
if __name__ == "__main__":
    
    from chemicalchecker.util import Config
    
    cfg = os.environ['CC_CONFIG']
    
    cc_path = Config(cfg).PATH.CC_REPO
    
    src_file = os.path.join(cc_path, 'package/scripts/feature_description_mappings/data/', 
                            'meddra_all_label_se.tsv')
    
    dst_folder = os.path.join(cc_path, 'package/scripts/feature_description_mappings')
    
    create_mapping(src_file, dst_folder)