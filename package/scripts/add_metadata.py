# Nico (25 Jan 2021)
# Add metadata into the attr dictionary of every reference signature's h5 file 
# (on backup copies since an error can be produced if the file is being read by someone else while we try accessing it)
# Metadata to add and example:
#   cctype: 'sign2'
#   dataset_code: 'A1.001'
#   molset: 'full'

import os, shutil
import h5py

def add_metadata(version="2020_01",signatures='0123', pathrepo="/aloy/web_checker/package_cc/"):
    """
    version: (str), version of the cc package
    signature: (str or int), number refering to the signature. ex: '012' for sign0, sign1, sign2.
    path: (str), path to the cc signature repo
    """
    signatures=str(signatures) # in case we have an int.

    for molset in ('full','reference'):
        for space in "ABCDE":
            for num in "12345":
                for sign in signatures:
                    signature= 'sign'+sign
                    data_code= space+num+'.001'
                    fichero= os.path.join(pathrepo,version,molset,space,space+num, data_code, signature, signature+'.h5')
                    
                    if os.path.exists(fichero):
                        print("Making backup which will contain metadata")
                        backup_file= os.path.join(os.path.dirname(fichero),os.path.basename(fichero).split('.')[0]+'_BACKUP.h5')
                        shutil.copyfile(fichero,backup_file)

                        print("Adding metadata to", backup_file)
                        dico= dict(cctype=signature, dataset_code=data_code, molset=molset)

                        with h5py.File(backup_file,'a') as f:
                            for k,v in dico.items():
                                if k not in f.attrs:
                                    f.attrs.create(name=k,data=v)
                                else:
                                    print(k,"already in f.attrs")

                    else:
                        print(fichero, "doesn't exist, skipping")


if __name__ == '__main__':
    current_version="2020_01"
    add_metadata(version=current_version)
