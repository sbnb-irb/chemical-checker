import sys
import os
import pickle
import h5py
import numpy as np
from scipy.stats import fisher_exact

task_id = sys.argv[1]  # <TASK_ID>
filename = sys.argv[2]  # <FILE>

input_pickle = pickle.load(open(filename, 'rb'))
idxs_path = input_pickle[task_id][0][0]
col_idx = int(input_pickle[task_id][0][1])
data_folder = input_pickle[task_id][0][2]
res_folder = input_pickle[task_id][0][3]

with h5py.File(data_folder, 'r') as f:
    col = f['V0'][:, col_idx]

space_size = len(col)
with_features = col[col!=0].shape[0]

with h5py.File(idxs_path, 'r') as f:
    res_array = np.zeros(len(f['neighbors']), dtype=np.float32)

    for n in range(len(f['neighbors'])):
        idx = f['neighbors'][str(n)]
        a = col[idx][col[idx]!=0].shape[0]
        b = with_features - a

        c = col[idx][col[idx]==0].shape[0]
        d = space_size - with_features - c
            
            
        odds, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')
        
        res_array[n] = pvalue

with open(os.path.join(res_folder, str(col_idx)), 'wb') as fh:
    pickle.dump(res_array, fh)
    
