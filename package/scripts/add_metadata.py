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
    pathrepo: (str), path to the cc signature repo
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
                                    print(k,"already in attrs")

                    else:
                        print(fichero, "doesn't exist, skipping")

def export_sign(target_dir, version="2020_01", signatures='2', pathrepo="/aloy/web_checker/package_cc/",molsets=('full',),copy_backup=False, add_metadata=True):
    """
    Export all signatures from a given cctype (ex: sign2) in a single folder
    Add Metadata to the output files if not present in the original h5 file

    target_dir (str): target directory where the signatures will be copied
    version (str): version of the checker
    signature: (str or int), number refering to the signature. ex: '012' for sign0, sign1, sign2.
    pathrepo: (str), path to the cc signature repo
    molsets: (list or tuple), either 'full' or 'reference'
    copy_backup (Bool): copy signx_BACKUP.h5 instead of signx.h5 if present
    add_metadata (Bool): Add metadata to the copied file
    """

    signatures=str(signatures) # in case we have an int.

    for sign in signatures:
        cctype='sign'+signature
        sign_dir= os.path.join(target_dir,cctype)

        if not os.path.exists(sign_dir):
            try:
                os.makedirs(sign_dir)
            except Exception as e:
                print("WARNING", e)
                continue

       
        for molset in molsets:
            for space in "ABCDE":
                for num in "12345":
                    signature= 'sign'+sign
                    data_code= space+num+'.001'

                    if copy_backup:
                        fichero= os.path.join(pathrepo,version,molset,space,space+num, data_code, signature, signature+'_BACKUP.h5')
                    else:
                        fichero= os.path.join(pathrepo,version,molset,space,space+num, data_code, signature, signature+'.h5')
                        
                    target_file  = os.path.join(sign_dir, cctype+'_'+space+num+'_'+molset+'.h5')

                    if os.path.exists(fichero):
                        if not os.path.exists(target_file):
                            print("Copying",fichero,"to",target_file)
                            shutil.copyfile(fichero,target_file)

                            # Adding metadata
                            if add_metadata:
                                print("Adding metadata to", target_file)
                                dico= dict(cctype=signature, dataset_code=data_code, molset=molset)

                                with h5py.File(target_file,'a') as f:
                                    for k,v in dico.items():
                                        if k not in f.attrs:
                                            f.attrs.create(name=k,data=v)
                                        else:
                                            print(k,"already in attrs")
                        else:
                            print("WARNING: file",target_file,"already exists!")
                            print("Please delete it first")
                    else:
                        print(fichero,"does not exist, skipping!")

if __name__=='__main__':

    current_version="2020_01"
    target_directory= "/aloy/scratch/nsoler/CC_related/EXPORT_SIGN"

    # Backup all h5 files and add metadata to the backups:
    #add_metadata(version=current_version)

    
    export_sign(target_directory,version="2020_01", signatures='2',molsets=['full'], copy_backup=True)


