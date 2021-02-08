## NS (08 Feb 2021), export the features from sign0
## So as 'sanitize' (i.e remove columns) identically as in the cc-rep when a custom dataset is made
## in A spaces

from chemicalchecker.core.chemcheck import ChemicalChecker

repo= "/aloy/web_checker/package_cc/2020_01/"
outDir= "/aloy/scratch/nsoler/CC_related/EXPORT_SIGN/sign0"

cc = ChemicalChecker(repo)

for space in "ABCDE":
	for num in (1, 2, 3, 4, 5):
		ds= space+str(num)+'.001'
		sign0tmp = cc.get_signature('sign0', 'full', ds)
		sign0tmp.export_features(outDir)