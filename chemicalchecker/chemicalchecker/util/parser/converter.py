"""Standardise molecule.

ref: https://wwwdev.ebi.ac.uk/chembl/extra/francis/standardiser/
"""

from standardiser import standardise
from rdkit.Chem import AllChem as Chem


class ConversionError(Exception):
    """Conversion error."""

    def __init__(self, message, smile):
        """Initialize a ConversionError."""
        message = "Cannot convert: %s Error: %s" % (smile, message)
        super(RuntimeError, self).__init__(message)


class Converter():
    """Container for static conversion methods."""

    @staticmethod
    def smile_to_inchi(smile):
        """From smile to inchikey and inchi."""
        mol = standardise.Chem.MolFromSmiles(smile)
        if not mol:
            raise ConversionError("MolFromSmiles returned None", smile)
        try:
            mol = standardise.run(mol)
        except Exception as ex:
            raise ConversionError("'standardise.run' exception", ex.message)
        inchi = Chem.rdinchi.MolToInchi(mol)[0]
        if not inchi:
            raise ConversionError("'MolToInchi' returned None.", smile)
        inchikey = Chem.rdinchi.InchiToInchiKey(inchi)
        if not inchi:
            raise ConversionError("'InchiToInchiKey' returned None", smile)
        try:
            mol = Chem.rdinchi.InchiToMol(inchi)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", ex.message)
        return inchikey, inchi
