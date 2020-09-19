import numpy as np
import os, shutil, sys
from chemicalchecker import ChemicalChecker

os.environ['CC_CONFIG'] = "/opt/chemical_checker/setup/users/cc_config_nico.json"

# NS filter the A spaces sign0 to keep only molecules present in the B spaces and after (bioactive molecules)
cc_repo="/aloy/web_checker/package_cc/2020_01/"
cc= ChemicalChecker(cc_repo)

# Get the union of molecules for exemplary B spaces and above 
spaces_to_filter = ['A1.001', 'A2.001', 'A3.001', 'A4.001', 'A5.001', 'B4.002']

for space in spaces_to_filter:


    print("NOW FILTERING sign0 for", space)
    s0 = cc.get_signature('sign0', 'full', space)

    # Copying back the backup to sign0.h5
    current_h5 = s0.data_path
    dirname= os.path.dirname(current_h5)
    backup = os.path.join(dirname, 'sign0BACKUP.h5')
    filtered_h5=os.path.join(os.path.dirname(s0.data_path), 'sign0_univ.h5')

    if os.path.exists(backup):
        print("Copying {} to {}".format(backup, current_h5))
        try:
            shutil.copyfile(backup, current_h5)
        except :
            print("Cannot copy", backup)
            print("Please check!")
            sys.exit(1)

        # Re-initializing sign0
        s0 = cc.get_signature('sign0', 'full', space)

    s0.restrict_to_universe()

    print("FILTERED: sign0", space)
