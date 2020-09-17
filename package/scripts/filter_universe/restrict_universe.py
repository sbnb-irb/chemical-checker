import numpy as np
import os
from chemicalchecker import ChemicalChecker

# NS filter the A spaces sign0 to keep only molecules present in the B spaces and after (bioactive molecules)
cc_repo="/aloy/web_checker/package_cc/2020_01/"
cc= ChemicalChecker(cc_repo)

# Get the union of molecules for exemplary B spaces and above 
universe = cc.universe
spaces_to_filter = ['A1.001', 'A2.001', 'A3.001', 'A4.001', 'A5.001']

for space in spaces_to_filter:

    s0 = cc.get_signature('sign0', 'full', space)

    # get the vectors corresponding to our (restricted) universe
    inchk_univ, _ = s0.get_vectors(keys=universe)

    # obtain a mask for sign0 in order to obtain a filtered h5 file
    mask= np.isin(list(s0.keys), list(inchk_univ))

    del inchk_univ

    filtered_h5=os.path.join(os.path.dirname(s0.data_path), 'sign0_univ.h5')
    print("Creating",filtered_h5)

    s0.make_filtered_copy(filtered_h5, mask)

    # After that check that your file is ok and move it to sign0.h5