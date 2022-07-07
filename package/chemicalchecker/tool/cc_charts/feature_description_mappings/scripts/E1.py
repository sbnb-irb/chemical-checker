import pickle
import pandas as pd
import os
import re
   
def create_mapping(src_file, dst_folder):
    
    e1_dict = dict()
    with open(src_file, 'r') as f:
        flag = False
        for l in f:
            l = re.split('\s+', l)

            if l[0] == '!':
                flag = not flag
                continue

            if flag:
                if len(l[0]) > 1:

                    l.insert(1, l[0][1])

                    l[0] = l[0][0]

                e1_dict[f'{l[0]}:{l[1]}'] = ' '.join(l[2:-1])
    e1_dict['D:L01XE'] = 'Protein kinase inhibitors'

    os.makedirs(dst_folder, exist_ok=True)

    dest = os.path.join(dst_folder, 'E1')
    
    with open(dest, 'wb') as fh:
        pickle.dump(e1_dict, fh)
        
if __name__ == "__main__":
    
    from chemicalchecker.util import Config
    
    cfg = os.environ['CC_CONFIG']
    
    cc_path = Config(cfg).PATH.CC_REPO
        
    src_file = os.path.join(cc_path, 'package/scripts/feature_description_mappings/data/', 
                        'br08303.keg')

    dst_folder = os.path.join(cc_path, 'package/scripts/feature_description_mappings')
    
    create_mapping(src_file, dst_folder)