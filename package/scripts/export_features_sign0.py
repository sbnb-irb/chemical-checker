## NS (08 Feb 2021), export the features from sign0
## So as 'sanitize' (i.e remove columns) identically as in the cc-rep when a custom dataset is made
## in A spaces

from chemicalchecker.core.chemcheck import ChemicalChecker
from get_repo_version import cc_repo_version
#VERSION= "2020_02"

def export_features_sign0(cc_repo=None, outDir="/aloy/scratch/nsoler/CC_related/EXPORT_SIGN/sign0"):

    if cc_repo is None:
    	cc_repo = cc_repo_version()

	    if cc_repo is None:
	        print("ERROR, cannot guess the latest cc repository path")
	        print("Please provide it as an argument")
	        print("ex: cc_repo='/aloy/web_checker/package_cc/2020_02'")
	        return
	    else:
	    	print("Working with cc_repo:",cc_repo)

	cc = ChemicalChecker(cc_repo)

	for space in "ABCDE":
		for num in (1, 2, 3, 4, 5):
			ds= space+str(num)+'.001'
			sign0tmp = cc.get_signature('sign0', 'full', ds)
			sign0tmp.export_features(outDir)