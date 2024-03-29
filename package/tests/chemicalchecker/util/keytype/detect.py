"""Automatically detect type of keys."""
import numpy as np

from chemicalchecker.util import logged


@logged
class KeyTypeDetector(object):
    """KeyTypeDetector class."""

    def __init__(self, keys, max_checks=1000, valid=0.75):
        """Initialize a KeyTypeDetector instance.

        Args:
            keys (list): Keys to be analyzed.
            max_checks (int): Maximum number of checks to perform.
                (default=1000).
            valid (float): Proportion of valid matches to decide on a key type.
                (default=0.75)
        """
        if len(keys) > max_checks:
            self.keys = np.random.choice(keys, max_checks, replace=False)
        else:
            self.keys = keys
        self.valid = valid

    @staticmethod
    def is_inchi(key):
        """Recognize InChI"""
        return key.startswith('InChI=')

    @staticmethod
    def is_inchikey(key):
        """Recognize InChIKey"""
        if len(key) != 27:
            return False
        if key[25] != "-":
            return False
        if key[14] != "-":
            return False
        if not key[:14].isalnum():
            return False
        if not key[15:25].isalnum():
            return False
        if not key[-1].isalnum():
            return False
        return True

    @staticmethod
    def is_smiles(key):
        """Recognize SMILES"""
        from rdkit import Chem
        m = Chem.MolFromSmiles(key)
        if m is None:
            return False
        return True

    @staticmethod
    def type(key):
        if KeyTypeDetector.is_inchi(key):
            return "inchi"
        if KeyTypeDetector.is_inchikey(key):
            return "inchikey"
        if KeyTypeDetector.is_smiles(key):
            return "smiles"
        return None

    def _detect(self, func):
        vals = []
        for k in self.keys:
            vals += [func(k)]
        if np.sum(vals) / len(vals) < self.valid:
            return False
        else:
            return True

    def detect(self):
        if self._detect(self.is_inchi):
            return "inchi"
        if self._detect(self.is_inchikey):
            return "inchikey"
        if self._detect(self.is_smiles):
            return "smiles"
        return None
