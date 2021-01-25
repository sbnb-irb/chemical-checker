# Nico (25 Jan 2021)
# Add metadata into the attr dictionary of every reference signature's h5 file
# Metadata to add and example:
#   cctype: 'sign2'
#   dataset_code: 'A1.001'
#   molset: 'full'

import shutil,os


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
                    fichero= os.path.join(pathrepo,version,molset,space,space+num,space+num+'.001','sign'+sign, 'sign'+sign+'.h5')
                    print(fichero)

if __name__ == '__main__':
    current_version="2020_01"
    add_metadata(version=current_version)
