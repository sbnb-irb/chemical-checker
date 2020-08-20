"""Molecule representation.

A simple class to inspect small molecules, easily convert between different
identifier, and ultimately find if signatures available.
"""
import collections

from chemicalchecker.util import logged
from chemicalchecker.util.keytype import KeyTypeDetector
from chemicalchecker.util.parser.converter import Converter


@logged
class Mol(object):
    """Mol class."""

    def __init__(self, cc, mol_str, str_type=None):
        """Initialize a Mol instance.

        Args:
            cc: Chemical Checker instance.
            mol_str: Compound identifier (e.g. SMILES string)
            str_type: Type of identifier ('inchikey', 'inchi' and 'smiles' are
                accepted) if 'None' we do our best to guess.
        """
        if str_type is None:
            str_type = KeyTypeDetector.type(mol_str)
            if str_type is None:
                raise Exception(
                    "Molecule '%s' not recognized as valid format"
                    " (options are: 'inchikey', 'inchi' and 'smiles')" %
                    mol_str)
        conv = Converter()
        if str_type == "inchikey":
            self.inchikey = mol_str
            self.inchi = conv.inchikey_to_inchi(
                self.inchikey)[0]["standardinchi"]
        if str_type == "inchi":
            self.inchi = mol_str
            self.inchikey = conv.inchi_to_inchikey(self.inchi)
        if str_type == "smiles":
            self.inchikey, self.inchi = conv.smiles_to_inchi(mol_str)
        self.smiles = conv.inchi_to_smiles(self.inchi)
        self.mol = conv.inchi_to_mol(self.inchi)
        self.cc = cc

    def isin(self, cctype, dataset_code, molset="full"):
        """Check if the molecule is in the dataset of interest"""
        sign = self.cc.get_signature(cctype, molset, dataset_code)
        try:
            keys = set(sign.keys)
        except Exception:
            keys = set(sign.row_keys)
        return self.inchikey in keys

    def signature(self, cctype, dataset_code, molset="full"):
        """Check if the molecule is in the dataset of interest"""
        sign = self.cc.get_signature(cctype, molset, dataset_code)
        return sign[self.inchikey]

    def report_available(self, dataset_code="*", cctype="*", molset="full"):
        """Check in what datasets the key is present"""
        available = self.cc.report_available(molset, dataset_code, cctype)
        d0 = {}
        for molset, datasets in available.items():
            d1 = collections.defaultdict(list)
            for dataset, cctypes in datasets.items():
                for cctype in cctypes:
                    if self.isin(cctype, dataset, molset):
                        d1[dataset] += [cctype]
            if d1:
                d0[molset] = dict((k, v) for k, v in d1.items())
        return d0

    def show(self):
        """Simply display the molecule in a Jupyter notebook"""
        from rdkit.Chem.Draw import IPythonConsole
        return self.mol
