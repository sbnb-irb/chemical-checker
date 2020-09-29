from chemicalchecker import ChemicalChecker
cc =ChemicalChecker("/aloy/web_checker/package_cc/2020_01/")

s0A1 = cc.get_signature('sign0', 'full', 'A1.001')

CC_OLD_ROOT = '/aloy/web_checker/package_cc/paper'

# already restricted to the universe
raw_file="/aloy/web_checker/package_cc/2020_01/full/A/A1/A1.001/sign0/raw/preprocess.h5"

s0A1.fit(cc=CC_OLD_ROOT, data_file=raw_file, overwrite=True)