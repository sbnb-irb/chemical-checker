"""Standardize molecule and convert between identifier."""
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
            from rdkit.Chem.Scaffolds import MurckoScaffold
            self.Chem = Chem
            self.scaffold = MurckoScaffold
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        try:
            from chembl_structure_pipeline.standardizer import standardize_mol
            self.standardize = standardize_mol
        except ImportError:
            raise ImportError("requires chembl_structure_pipeline")
        try:
            import pubchempy as pcp
            self.pcp = pcp
        except ImportError:
            raise ImportError("requires pubchempy")

    def smiles_to_scaffold(self, smiles):
        """From SMILES to the SMILES of its scaffold."""
        scaffold_smiles = self.scaffold.MurckoScaffoldSmiles(smiles)
        return scaffold_smiles

    def smiles_to_inchi(self, smiles):
        """From SMILES to InChIKey and InChI."""
        mol = self.Chem.MolFromSmiles(smiles)
        if not mol:
            raise ConversionError("MolFromSmiles returned None", smiles)
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception", ex.message)
        inchi = self.Chem.rdinchi.MolToInchi(mol)[0]
        if not inchi:
            raise ConversionError("'MolToInchi' returned None.", smiles)
        inchikey = self.Chem.rdinchi.InchiToInchiKey(inchi)
        if not inchikey:
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
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception", inchi)
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
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception", ex.message)
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
            Converter.__log.warning(str(ex))
            raise ConversionError("Cannot fetch PubChemID CID from CTD", ctdid)
        # get smiles
        try:
            url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/' + \
                'cid/%s/property/CanonicalSMILES/TXT/' % pubchemcid
            smiles = urlopen(url).read().rstrip().decode()
        except Exception as ex:
            Converter.__log.warning(str(ex))
            raise ConversionError(
                "Cannot fetch SMILES from PubChemID CID", pubchemcid)
        return smiles

    @staticmethod
    def chemical_name_to_smiles(chem_name):
        """From Chemical Name to SMILES via cactus.nci or pubchem."""
        smiles = None
        chem_name_quoted = quote(chem_name)
        smiles = Converter._chemical_name_to_smiles_cactus(chem_name_quoted)
        if smiles is not None:
            return smiles
        smiles = Converter._chemical_name_to_smiles_pubchem(chem_name)
        if smiles is None:
            raise ConversionError(
                "Cannot fetch SMILES from Chemical Name", chem_name)
        return smiles

    @staticmethod
    def chemical_name_to_inchi(chem_name):
        """From Chemical Name to InChI via cactus.nci or pubchem."""
        inchi = None
        chem_name_quoted = quote(chem_name)
        inchi = Converter._chemical_name_to_inchi_cactus(chem_name_quoted)
        if inchi is not None:
            return inchi
        inchi = Converter._chemical_name_to_inchi_pubchem(chem_name)
        if inchi is None:
            raise ConversionError(
                "Cannot fetch InChI from Chemical Name", chem_name)
        return inchi

    @staticmethod
    def _chemical_name_to_smiles_cactus(chem_name):
        """From chemical name to SMILES."""
        try:
            url = 'http://cactus.nci.nih.gov/chemical/' + \
                'structure/%s/smiles' % chem_name
            smiles = urlopen(url).read().rstrip().decode()
            return smiles
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name to SMILES (cactus.nci): %s" % chem_name)
            return None

    @staticmethod
    def _chemical_name_to_inchi_cactus(chem_name):
        """From chemical name to InChI."""
        try:
            url = 'http://cactus.nci.nih.gov/chemical/' + \
                'structure/%s/stdinchi' % chem_name
            inchi = urlopen(url).read().rstrip().decode()
            return inchi
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name to InChI (cactus.nci): %s" % chem_name)
            return None

    @staticmethod
    def _chemical_name_to_smiles_pubchem(chem_name):
        """From chemical name to SMILES."""
        try:
            cpds = self.pcp.get_compounds(chem_name, 'name')
            if len(cpds) == 0:
                Converter.__log.warning(
                    "Cannot convert Chemical Name to SMILES (pubchem): %s" % chem_name)
                return None
            if len(cpds) > 1:
                Converter.__log.warning(
                    "Multiple CIDs found, using first: %s" % str(cpds))
            return cpds[0].isomeric_smiles
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name to SMILES (pubchem): %s" % chem_name)
            return None

    @staticmethod
    def _chemical_name_to_inchi_pubchem(chem_name):
        """From chemical name to InChI."""
        try:
            cpds = self.pcp.get_compounds(chem_name, 'name')
            if len(cpds) == 0:
                Converter.__log.warning(
                    "Cannot convert Chemical Name to InChI (pubchem): %s" % chem_name)
                return None
            if len(cpds) > 1:
                Converter.__log.warning(
                    "Multiple CIDs found, using first: %s" % str(cpds))
            return cpds[0].inchi
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name to InChI (pubchem): %s" % chem_name)
            return None

    @staticmethod
    def inchikey_to_inchi(inchikey):
        """From InChIKey to InChI."""
        try:
            inchikey = quote(inchikey)
            url = 'https://www.ebi.ac.uk/unichem/rest/inchi/%s' % inchikey
            return json.loads(urlopen(url).read().rstrip().decode())
        except Exception as ex:
            Converter.__log.warning(str(ex))
            raise ConversionError(
                "Cannot fetch SMILES from Chemical Name", inchikey)
