import pickle
import pandas as pd
import os

def create_mapping(src_file, dst_folder):

    from chemicalchecker.util import psql

    chembl_dbname = 'chembl'

    chembl_class_query = psql.qstring(f'''SELECT DISTINCT pc.protein_class_id, pc.pref_name
    FROM protein_classification pc
    ''', chembl_dbname)

    chembl_classes = dict()

    for key, value in chembl_class_query:
        chembl_classes[f'Class:{key}'] = value

    chembl_protein_query = psql.qstring(f'''SELECT DISTINCT cs.accession, cs.description
    FROM component_sequences cs
    ''', chembl_dbname)
    
    drugbank_protein_query = psql.qstring('''
    SELECT DISTINCT t.uniprot_id, t.name
    FROM target t
    ''', 'drugbank')

    chembl_proteins = dict()
    for key, value in chembl_protein_query:
        chembl_proteins[key] = value
        
    drugbank_proteins = dict()
    for key, value in drugbank_protein_query:
        drugbank_proteins[key] = value

    b_dict = {**chembl_classes, **chembl_proteins, **drugbank_proteins}

    for dataset in ['B2', 'B4', 'B5']:
        dest = os.path.join(dst_folder, dataset)
        with open(dest, 'wb') as fh:
            pickle.dump(b_dict, fh)
    
    against_dict = {f'{key}(-1)': f'Against {value}' for key, value in b_dict.items()}
    favor_dict = {f'{key}(1)': f'Favors {value}' for key, value in b_dict.items()}    
    b1_dict = {**against_dict, **favor_dict}
    
    dest = os.path.join(dst_folder, 'B1')
    with open(dest, 'wb') as fh:
        pickle.dump(b1_dict, fh)
    
    

if __name__ == "__main__":
    
    from chemicalchecker.util import Config
       
    cfg = os.environ['CC_CONFIG']
    
    cc_path = Config(cfg).PATH.CC_REPO
    
    dst_folder = os.path.join(cc_path, 'package/scripts/feature_description_mappings')
    
    create_mapping(None, dst_folder)