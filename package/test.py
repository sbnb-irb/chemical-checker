from chemicalchecker import ChemicalChecker
ChemicalChecker.set_verbosity('DEBUG')
from chemicalchecker.util.parser.converter import Converter
conv = Converter()
ink = 'LPXQRXLUHJKZIE-UHFFFAOYSA-N'
inchi = conv.inchikey_to_inchi(ink, local_db=False, mapping_dict={'XZWYZXLIPXDOLR-UHFFFAOYSA-N': 'InChI=1S/C4H11N5/c1-9(2)4(7)8-3(5)6/h1-2H3,(H5,5,6,7,8)'})
