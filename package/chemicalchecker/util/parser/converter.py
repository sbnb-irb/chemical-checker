"""Standardize molecule and convert between identifier."""

import json
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import quote

from chemicalchecker.util import logged
from .request_helpers import _cache, _urlopen_retry


class ConversionError(Exception):
    """Conversion error."""

    def __init__(self, message, idx):
        """Initialize a ConversionError."""
        message = "Cannot convert: %s Message: %s" % (idx, message)
        super(Exception, self).__init__(message)


@logged
class Converter:
    """Converter class."""

    def __init__(self):
        """Initialize a Converter instance."""
        try:
            import rdkit.Chem as Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            self.Chem = Chem
            self.scaffold = MurckoScaffold
        except ImportError:
            raise ImportError("requires rdkit " + "https://www.rdkit.org/")
        try:
            from chembl_structure_pipeline.standardizer import (
                standardize_mol,
                get_parent_mol,
            )

            self._standardize_mol = standardize_mol
            self._get_parent_mol = get_parent_mol
        except ImportError:
            raise ImportError("requires chembl_structure_pipeline")
        try:
            import pubchempy as pcp

            self.pcp = pcp
        except ImportError:
            raise ImportError("requires pubchempy")

    def standardize(self, mol):
        """Standardize and desalt a molecule the ChEMBL way.

        Reproduces the logic of the ``standardiser.run`` call used until 2020:
        normalize functional groups, then strip salts/solvents, remove isotope
        labels and neutralize charges so that, e.g., a sodium/hydrochloride salt
        collapses to the same parent structure (and InChIKey) as its free
        acid/base form. The ChEMBL pipeline keeps these steps separate, so a
        bare ``standardize_mol`` would leave counterions in place; we follow
        ChEMBL's documented order of ``standardize_mol`` then ``get_parent_mol``.
        """
        mol = self._standardize_mol(mol)
        # get_parent_mol returns (parent_mol, exclude_flag); neutralize=True by
        # default also removes the charges the old standardiser stripped.
        mol, _ = self._get_parent_mol(mol)
        # Replicate the 2020 standardiser's no_non_salt / multi_component guards.
        # Count organic fragments (those containing at least one carbon) remaining
        # after desalting. Zero → only inorganic ions survived; >1 → true mixture,
        # which has no meaningful single-compound signature in the CC.
        organic_frags = [
            f for f in self.Chem.GetMolFrags(mol, asMols=True)
            if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())
        ]
        if len(organic_frags) == 0:
            raise ConversionError("no organic fragment after desalting", mol)
        if len(organic_frags) > 1:
            raise ConversionError("multi-component mixture after desalting", mol)
        return mol

    def smiles_to_scaffold(self, smiles, generic=False):
        """From SMILES to the SMILES of its scaffold."""
        scaffold_smiles = self.scaffold.MurckoScaffoldSmiles(smiles)
        if generic:
            scaffold_mol = self.scaffold.MakeScaffoldGeneric(
                self.Chem.MolFromSmiles(scaffold_smiles)
            )
            scaffold_smiles = self.Chem.MolToSmiles(scaffold_mol)
        return scaffold_smiles

    def smiles_to_inchi(self, smiles):
        """From SMILES to InChIKey and InChI."""
        mol = self.Chem.MolFromSmiles(smiles)
        if not mol:
            raise ConversionError("MolFromSmiles returned None", smiles)
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception:", smiles)
        inchi = self.Chem.rdinchi.MolToInchi(mol)[0]
        if not inchi:
            raise ConversionError("'MolToInchi' returned None.", smiles)
        inchikey = self.Chem.rdinchi.InchiToInchiKey(inchi)
        if not inchikey:
            raise ConversionError("'InchiToInchiKey' returned None", smiles)
        try:
            mol = self.Chem.rdinchi.InchiToMol(inchi)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", smiles)
        return inchikey, inchi

    def inchi_to_smiles(self, inchi):
        """From InChI to SMILES."""
        try:
            inchi_ascii = inchi.encode("ascii", "ignore")
            mol = self.Chem.rdinchi.InchiToMol(inchi_ascii)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", inchi)
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception:", inchi)
        return self.Chem.MolToSmiles(mol, isomericSmiles=True)

    def inchi_to_inchikey(self, inchi):
        """From InChI to InChIKey."""
        try:
            inchi_ascii = inchi.encode("ascii", "ignore")
            inchikey = self.Chem.rdinchi.InchiToInchiKey(inchi_ascii)
        except Exception as ex:
            raise ConversionError("'InchiToInchiKey' exception:", inchi)
        return inchikey

    def inchi_to_mol(self, inchi):
        """From InChI to molecule."""
        try:
            inchi_ascii = inchi.encode("ascii", "ignore")
            mol = self.Chem.rdinchi.InchiToMol(inchi_ascii)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", inchi)
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception:", inchi)
        return mol

    @staticmethod
    def ctd_to_smiles(ctdid):
        """From CTD identifier to SMILES."""
        key = "ctd_smiles:" + ctdid
        cached = _cache.get(key)
        if cached is not None:
            return cached
        # convert to pubchemcid
        try:
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/"
                + "sourceid/Comparative%20Toxicogenomics%20Database/"
                + ctdid
                + "/cids/TXT/"
            )
            pubchemcid = _urlopen_retry(
                url).read().rstrip().decode().splitlines()[0].strip()
        except Exception as ex:
            Converter.__log.warning(str(ex))
            raise ConversionError("Cannot fetch PubChemID CID from CTD", ctdid)
        # get smiles
        try:
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                + "cid/%s/property/CanonicalSMILES/TXT/" % pubchemcid
            )
            smiles = _urlopen_retry(
                url).read().rstrip().decode().splitlines()[0].strip()
        except Exception as ex:
            Converter.__log.warning(str(ex))
            raise ConversionError("Cannot fetch SMILES from PubChemID CID", pubchemcid)
        _cache.set(key, smiles)
        return smiles

    @staticmethod
    def _cid_to_canonical_smiles(cid):
        """Fetch the (non-stereo) CanonicalSMILES for a PubChem CID.

        ``CTD_chemicals.tsv`` stores the CID with a ``CID:`` prefix (and may list
        several, pipe-separated); normalise to the first bare numeric id.
        """
        cid = str(cid).split("|")[0].replace("CID:", "").strip()
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            + "cid/%s/property/CanonicalSMILES/TXT/" % cid
        )
        return _urlopen_retry(url).read().rstrip().decode().splitlines()[0].strip()

    @staticmethod
    def ctd_file_to_smiles(pubchem_cid="", inchikey="", casrn=""):
        """Resolve a CTD chemical to SMILES from the structure identifiers already
        present in ``CTD_chemicals.tsv``.

        This bypasses the decayed CTD-id -> PubChem-substance bridge (whose
        substance->compound links drift between releases) by going straight to a
        stable identifier. Every route ends in ``CanonicalSMILES`` so the resulting
        InChIKey is non-stereo, consistent with :meth:`ctd_to_smiles`. Tries, in
        order, PubChem CID, then InChIKey, then CasRN. Returns SMILES or ``None``.
        """
        key = "ctd_file_smiles:%s|%s|%s" % (pubchem_cid, inchikey, casrn)
        cached = _cache.get(key)
        if cached is not None:
            return cached
        smiles = None
        # 1. PubChem CID -> CanonicalSMILES (direct, most stable)
        if pubchem_cid:
            try:
                smiles = Converter._cid_to_canonical_smiles(pubchem_cid.split("|")[0])
            except Exception as ex:
                Converter.__log.warning("ctd_file CID lookup failed: %s", str(ex))
        # 2. InChIKey -> CanonicalSMILES (normalises to the non-stereo form)
        if not smiles and inchikey:
            try:
                url = (
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                    + "inchikey/%s/property/CanonicalSMILES/TXT/"
                    % quote(inchikey.split("|")[0])
                )
                smiles = _urlopen_retry(url).read().rstrip().decode().splitlines()[0].strip()
            except Exception as ex:
                Converter.__log.warning("ctd_file InChIKey lookup failed: %s", str(ex))
        # 3. CasRN -> CID -> CanonicalSMILES
        if not smiles and casrn:
            try:
                url = (
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                    + "name/%s/cids/TXT/" % quote(casrn.split("|")[0])
                )
                cid = _urlopen_retry(url).read().rstrip().decode().splitlines()[0].strip()
                if cid.isdigit():
                    smiles = Converter._cid_to_canonical_smiles(cid)
            except Exception as ex:
                Converter.__log.warning("ctd_file CasRN lookup failed: %s", str(ex))
        if smiles:
            _cache.set(key, smiles)
        return smiles

    @staticmethod
    def chemical_name_to_smiles(chem_name):
        """From Chemical Name to SMILES via cactus.nci or pubchem."""
        key = "name_smiles:" + chem_name
        cached = _cache.get(key)
        if cached is not None:
            return cached
        smiles = None
        chem_name_quoted = quote(chem_name)
        smiles = Converter._chemical_name_to_smiles_cactus(chem_name_quoted)
        if smiles is not None:
            _cache.set(key, smiles)
            return smiles
        smiles = Converter._chemical_name_to_smiles_pubchem(chem_name)
        if smiles is None:
            raise ConversionError("Cannot fetch SMILES from Chemical Name", chem_name)
        _cache.set(key, smiles)
        return smiles

    @staticmethod
    def chemical_name_to_inchi(chem_name):
        """From Chemical Name to InChI via cactus.nci or pubchem."""
        key = "name_inchi:" + chem_name
        cached = _cache.get(key)
        if cached is not None:
            return cached
        inchi = None
        chem_name_quoted = quote(chem_name)
        inchi = Converter._chemical_name_to_inchi_cactus(chem_name_quoted)
        if inchi is not None:
            _cache.set(key, inchi)
            return inchi
        inchi = Converter._chemical_name_to_inchi_pubchem(chem_name)
        if inchi is None:
            raise ConversionError("Cannot fetch InChI from Chemical Name", chem_name)
        _cache.set(key, inchi)
        return inchi

    @staticmethod
    def _chemical_name_to_smiles_cactus(chem_name):
        """From chemical name to SMILES."""
        try:
            url = (
                "https://cactus.nci.nih.gov/chemical/"
                + "structure/%s/smiles" % chem_name
            )
            smiles = _urlopen_retry(url).read().rstrip().decode()
            return smiles
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name " "to SMILES (cactus.nci): %s" % chem_name
            )
            return None

    @staticmethod
    def _chemical_name_to_inchi_cactus(chem_name):
        """From chemical name to InChI."""
        try:
            url = (
                "https://cactus.nci.nih.gov/chemical/"
                + "structure/%s/stdinchi" % chem_name
            )
            inchi = _urlopen_retry(url).read().rstrip().decode()
            return inchi
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name " "to InChI (cactus.nci): %s" % chem_name
            )
            return None

    @staticmethod
    def _chemical_name_to_smiles_pubchem(chem_name):
        """From chemical name to SMILES."""
        try:
            import pubchempy as pcp
            # Use the property endpoint rather than the record-based
            # ``isomeric_smiles``: PubChem renamed its SMILES properties so the
            # old attribute is always empty. Request CanonicalSMILES (the
            # non-stereo connectivity string, now returned under the
            # ConnectivitySMILES key) to stay consistent with ctd_to_smiles,
            # which keys E4 on the same non-stereo SMILES.
            props = pcp.get_properties("CanonicalSMILES", chem_name, "name")
            if not props:
                Converter.__log.warning(
                    "Cannot convert Chemical Name "
                    "to SMILES (pubchem): %s" % chem_name
                )
                return None
            if len(props) > 1:
                Converter.__log.warning(
                    "Multiple CIDs found, using first: %s" % str(props)
                )
            for key, value in props[0].items():
                if "SMILES" in key and value:
                    return value
            return None
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name " "to SMILES (pubchem): %s" % chem_name
            )
            return None

    @staticmethod
    def _chemical_name_to_inchi_pubchem(chem_name):
        """From chemical name to InChI."""
        try:
            import pubchempy as pcp
            props = pcp.get_properties("InChI", chem_name, "name")
            if not props:
                Converter.__log.warning(
                    "Cannot convert Chemical Name " "to InChI (pubchem): %s" % chem_name
                )
                return None
            if len(props) > 1:
                Converter.__log.warning(
                    "Multiple CIDs found, using first: %s" % str(props)
                )
            return props[0].get("InChI")
        except Exception as ex:
            Converter.__log.warning(
                "Cannot convert Chemical Name " "to InChI (pubchem): %s" % chem_name
            )
            return None

    @staticmethod
    def _resove_inchikey_unichem(inchikey):
        try:
            inchikey = quote(inchikey)
            url = "https://www.ebi.ac.uk/unichem/rest/inchi/%s" % inchikey
            res = json.loads(urlopen(url).read().rstrip().decode())
        except Exception as ex:
            # Converter.__log.warning(str(ex))
            raise ConversionError("No response from unichem: %s" % url, inchikey)

        if isinstance(res, dict):
            err_msg = "; ".join(["%s: %s" % (k, v) for k, v in res.items()])
            raise ConversionError(err_msg, inchikey)
        elif isinstance(res, list):
            if len(res) != 1:
                raise ConversionError(
                    "No results from unichem: %s" % str(res), inchikey
                )
            if "standardinchi" not in res[0]:
                raise ConversionError(
                    "No results from unichem: %s" % str(res), inchikey
                )
            inchi = res[0]["standardinchi"]
            return inchi

    @staticmethod
    def _resove_inchikey_cactus(inchikey):
        try:
            inchikey = quote(inchikey)
            url = (
                "https://cactus.nci.nih.gov/"
                "chemical/structure/%s/stdinchi" % inchikey
            )
            res = urlopen(url).read().rstrip().decode()
            return res
        except Exception as ex:
            # Converter.__log.warning(str(ex))
            raise ConversionError("No response from cactus: %s" % url, inchikey)

    @staticmethod
    def _resove_inchikey_pubchem(inchikey):
        try:
            cpds = Converter().pcp.get_compounds(inchikey, "inchikey")
            if len(cpds) == 0:
                raise ConversionError("No results from pubchem", inchikey)
            if len(cpds) > 1:
                pass
                # Converter.__log.debug(
                #    "Multiple CIDs found, using first: %s" % str(cpds))
            return cpds[0].inchi
        except Exception as ex:
            Converter.__log.warning(str(ex))
            raise ConversionError("No response from pubchem: %s" % url, inchikey)

    @staticmethod
    def inchikey_to_inchi(inchikey, local_db=True, save_local=True, mapping_dict=None):
        """From InChIKey to InChI.

        Precedence is given to the local db that will be the fastest option.
        If it is not found locally several provider are contacted, and we
        possibly want to add the it to the Molecule table.
        """
        if local_db:
            from chemicalchecker.database import Molecule

            res = Molecule.get_inchikey_inchi_mapping([inchikey])
            if res[inchikey] is not None:
                return res[inchikey]

        if mapping_dict is not None:
            try:
                return mapping_dict[inchikey]
            except:
                Converter.__log.debug(
                    "InChIKey %s not found in dictionary, searching in external DBs..."
                    % inchikey
                )
                
        resolve_fns = {
            "unichem": Converter._resove_inchikey_unichem,
            "cactus": Converter._resove_inchikey_cactus,
            "pubchem": Converter._resove_inchikey_pubchem,
        }
        inchi = None
        for provider, func in resolve_fns.items():
            print(provider)
            try:
                inchi = func(inchikey)
                break
            except:
                Converter.__log.debug(
                    "InChIKey %s not found via %s" % (inchikey, provider)
                )
                continue
        if inchi is None:
            raise ConversionError("Unable to resolve", inchikey)
        if save_local:
            from chemicalchecker.database import Molecule

            Molecule.add_bulk([[inchikey, inchi]])
        return inchi
