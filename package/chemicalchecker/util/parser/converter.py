"""Standardise molecule and convert between identifier."""
import json
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import quote

from chemicalchecker.util import logged


class ConversionError(Exception):
    """Conversion error."""

    def __init__(self, message, idx):
        """Initialize a ConversionError."""
        message = "Cannot convert: %s Message: %s" % (idx, message)
        super(Exception, self).__init__(message)


@logged
class Converter():
    """Converter class."""

    def __init__(self):
        """Initialize a Converter instance."""
        try:
            import rdkit.Chem as Chem
            self.Chem = Chem
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        try:
            from standardiser import standardise
            self.standardise = standardise
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://wwwdev.ebi.ac.uk/chembl/extra/" +
                              "francis/standardiser/")

    def smiles_to_inchi(self, smiles):
        """From SMILES to InChIKey and InChI."""
        mol = self.standardise.Chem.MolFromSmiles(smiles)
        if not mol:
            raise ConversionError("MolFromSmiles returned None", smiles)
        try:
            mol = self.standardise.run(mol)
        except Exception as ex:
            raise ConversionError("'standardise.run' exception", ex.message)
        inchi = self.Chem.rdinchi.MolToInchi(mol)[0]
        if not inchi:
            raise ConversionError("'MolToInchi' returned None.", smiles)
        inchikey = self.Chem.rdinchi.InchiToInchiKey(inchi)
        if not inchi:
            raise ConversionError("'InchiToInchiKey' returned None", smiles)
        try:
            mol = self.Chem.rdinchi.InchiToMol(inchi)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", ex.message)
        return inchikey, inchi

    def inchi_to_smiles(self, inchi):
        """From InChI to SMILES."""
        try:
            inchi_ascii = inchi.encode('ascii', 'ignore')
            mol = self.Chem.rdinchi.InchiToMol(inchi_ascii)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", ex.message)
        return self.Chem.MolToSmiles(mol, isomericSmiles=True)

    def inchi_to_inchikey(self, inchi):
        """From InChI to InChIKey."""
        try:
            inchi_ascii = inchi.encode('ascii', 'ignore')
            inchikey = self.Chem.rdinchi.InchiToInchiKey(inchi_ascii)
        except Exception as ex:
            raise ConversionError("'InchiToInchiKey' exception:", ex.message)
        return inchikey

    def inchi_to_mol(self, inchi):
        """From InChI to molecule."""
        try:
            inchi_ascii = inchi.encode("ascii", "ignore")
            mol = self.Chem.rdinchi.InchiToMol(inchi_ascii)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", ex.message)
        return mol

    @staticmethod
    def ctd_to_smiles(ctdid):
        """From CTD identifier to SMILES."""
        # convert to pubchemcid
        try:
            url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/' + \
                'sourceid/Comparative%20Toxicogenomics%20Database/' + \
                ctdid + '/cids/TXT/'
            pubchemcid = urlopen(url).read().rstrip().decode()
        except Exception as ex:
            Converter.__log.warn(str(ex))
            raise ConversionError("Cannot fetch PubChemID CID from CTD", ctdid)
        # get smiles
        try:
            url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/' + \
                'cid/%s/property/CanonicalSMILES/TXT/' % pubchemcid
            smiles = urlopen(url).read().rstrip().decode()
        except Exception as ex:
            Converter.__log.warn(str(ex))
            raise ConversionError(
                "Cannot fetch SMILES from PubChemID CID", pubchemcid)
        return smiles

    @staticmethod
    def chemical_name_to_smiles(chem_name):
        """From chemical name to SMILES."""
        try:
            chem_name = quote(chem_name)
            url = 'http://cactus.nci.nih.gov/chemical/' + \
                'structure/%s/smiles' % chem_name
            return urlopen(url).read().rstrip().decode()
        except Exception as ex:
            Converter.__log.warn(str(ex))
            raise ConversionError(
                "Cannot fetch SMILES from Chemical Name", chem_name)

    @staticmethod
    def inchikey_to_inchi(inchikey):
        """From InChIKey to InChI."""
        try:
            inchikey = quote(inchikey)
            url = 'https://www.ebi.ac.uk/unichem/rest/inchi/%s' % inchikey
            return json.loads(urlopen(url).read().rstrip().decode())
        except Exception as ex:
            Converter.__log.warn(str(ex))
            raise ConversionError(
                "Cannot fetch SMILES from Chemical Name", inchikey)
