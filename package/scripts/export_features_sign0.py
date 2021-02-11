## NS (08 Feb 2021), export the features from sign0
## So as 'sanitize' (i.e remove columns) identically as in the cc-rep when a custom dataset is made
## in A spaces

import os
from chemicalchecker.core.chemcheck import ChemicalChecker
from get_repo_version import cc_repo_version
#VERSION= "2020_02"

def export_features_sign0(destination="/aloy/scratch/nsoler/CC_related/EXPORT_SIGN/sign0_features", spaces="ABCDE", cc_repo=None):

    if cc_repo is None:
        cc_repo = cc_repo_version()

        if cc_repo is None:
            print("ERROR, cannot guess the latest cc repository path")
            print("Please provide it as an argument")
            print("ex: cc_repo='/aloy/web_checker/package_cc/2020_02'")
            return
        else:
            print("Working with cc_repo:",cc_repo)

    if not os.path.exists(destination):
        try:
            os.makedirs(destination)
        except Exception as e:
            print("ERROR while attempting to create destination folder", destination)
            print(e)
        else:
            print("Created directory", destination)

    cc = ChemicalChecker(cc_repo)

    for space in spaces:
        for num in (1, 2, 3, 4, 5):
            ds= space+str(num)+'.001'

            sign0tmp = cc.get_signature('sign0', 'full', ds)
            sign0tmp.export_features(destination)

if __name__== '__main__':
    destination="/aloy/scratch/nsoler/CC_related/EXPORT_SIGN/sign0_features"
    export_features_sign0(destination=destination, spaces="A")