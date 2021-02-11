# Nico (25 Jan 2021)
# Add metadata into the attr dictionary of every reference signature's h5 file 
# (on backup copies since an error can be produced if the file is being read by someone else while we try accessing it)
# Metadata to add and example:
#   cctype: 'sign2'
#   dataset_code: 'A1.001'
#   molset: 'full'

import os, shutil
import h5py

#VERSION= "2020_02"
from get_repo_version import cc_repo_version

def remove_backups(cc_repo="2020_02"):
    """
    Removes the previous signx_BACKUP.h5 so that the next function can generate them
    DANGEROUS script! Be careful.
    """
    root="/aloy/web_checker/package_cc/"
    cc_repo = os.path.join(root,cc_repo)
    signatures='0123'

    for molset in ('full','reference'):
        for space in "ABCDE":
            for num in "12345":
                for sign in signatures:
                    signature= 'sign'+sign
                    data_code= space+num+'.001'


                    fichero= os.path.join(cc_repo,molset,space,space+num, data_code, signature, signature+'_BACKUP.h5')
                    
                    if os.path.exists(fichero):
                        try:
                            os.remove(fichero)
                        except Exception as e:
                            print("WARNING", e)
                        else:
                            print("Deleted:",fichero)



def add_metadata(cc_repo=None,signatures='0123', backup=False):
    """
    cc_repo: (str) path to a cc sign repo i.e /aloy/web_checker/package_cc/2020_02
    signature: (str or int), number refering to the signature. ex: '012' for sign0, sign1, sign2.
    backup (bool): make a backup copy of the signature first and add metadata to the backup instead
    """

    if cc_repo is None:
        cc_repo = cc_repo_version()

        if cc_repo is None:
            print("ERROR, cannot guess the latest cc repository path")
            print("Please provide it as an argument")
            print("ex: cc_repo='/aloy/web_checker/package_cc/2020_02'")
            return
        else:
            print("Working with cc_repo:",cc_repo)


    signatures=str(signatures) # in case we have an int.

    for molset in ('full','reference'):
        for space in "ABCDE":
            for num in "12345":
                for sign in signatures:
                    signature= 'sign'+sign
                    data_code= space+num+'.001'


                    fichero= os.path.join(cc_repo,molset,space,space+num, data_code, signature, signature+'.h5')
                    
                    if os.path.exists(fichero):

                        if backup:
                            print("Making backup which will contain metadata")
                            backup_file= os.path.join(os.path.dirname(fichero),os.path.basename(fichero).split('.')[0]+'_BACKUP.h5')
                            if not os.path.exists(backup_file):
                                shutil.copyfile(fichero,backup_file)
                            else:
                                print("Backup file", backup_file,"already exists, just adding metadata to it.")
                            fichero=backup_file

                        print("Adding metadata to", fichero)
                        dico= dict(cctype=signature, dataset_code=data_code, molset=molset)

                        with h5py.File(fichero,'r+') as f:
                            for k,v in dico.items():
                                if k not in f.attrs:
                                    f.attrs.create(name=k,data=v)
                                else:
                                    print(k,"already in attrs")

                    else:
                        print(fichero, "doesn't exist, skipping")

                    print("\n____")

def export_sign(target_dir, cc_repo=None, signatures='2',molsets=('full',),copy_backup=False, add_metadata=True):
    """
    Export all signatures from a given cctype (ex: sign2) in a single folder
    Add Metadata to the output files if not present in the original h5 file

    target_dir (str): target directory where the signatures will be copied
    cc_repo: (str) path to a cc sign repo i.e /aloy/web_checker/package_cc/2020_02
    signature: (str or int), number refering to the signature. ex: '012' for sign0, sign1, sign2.
    molsets: (list or tuple), either 'full' or 'reference'
    copy_backup (Bool): copy signx_BACKUP.h5 instead of signx.h5 if present
    add_metadata (Bool): Add metadata to the copied file
    """

    if cc_repo is None:
        cc_repo = cc_repo_version()

        if cc_repo is None:
            print("ERROR, cannot guess the latest cc repository path")
            print("Please provide it as an argument")
            print("ex: cc_repo='/aloy/web_checker/package_cc/2020_02'")
            return
        else:
            print("Working with cc_repo:",cc_repo)

    signatures=str(signatures) # in case we have an int.

    for sign in signatures:
        cctype='sign'+sign
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
                        fichero= os.path.join(cc_repo,molset,space,space+num, data_code, signature, signature+'_BACKUP.h5')
                    else:
                        fichero= os.path.join(cc_repo,molset,space,space+num, data_code, signature, signature+'.h5')
                        
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

    target_directory= "/aloy/scratch/nsoler/CC_related/EXPORT_SIGN"

    # Backup all h5 files and add metadata to the backups:
    add_metadata()

    
    #export_sign(target_directory, signatures='2',molsets=['full'], copy_backup=True)


