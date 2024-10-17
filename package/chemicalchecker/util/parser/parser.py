"""Container for static parsing methods.

Each parsing function here is iterating on a raw input file.
Each input line is a molecule which is loaded, standardised and converted
to InChI and InChIKeys.
The raw features are yielded in chunks as dictionaries.
These methods are used to populate the :mod:`~chemicalchecker.database.molrepo`
database table.
"""
import os
import csv
import pandas as pd
import xml.etree.ElementTree as ET

from .converter import Converter
from chemicalchecker.util import logged
from chemicalchecker.util import psql


@logged
class Parser():
    """Parser class."""

    @staticmethod
    def parse_fn(function):
        """Serve a parse function."""
        try:
            return eval('Parser.' + function)
        except Exception as ex:
            Parser.__log.error("Cannot find parsing function %s", function)
            raise ex

    @staticmethod
    def bindingdb(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = map_paths[molrepo_name]
        fh = open(os.path.join(file_path), "r")
        # skip header
        header = fh.readline()
        header_rows = 1
        header = header.rstrip("\n").split("\t")
        # get indexes
        bdlig_idx = header.index("Ligand InChI Key")
        smiles_idx = header.index("Ligand SMILES")
        done = set()
        chunk = list()
        for idx, line in enumerate(fh):
            idx = idx + header_rows
            line = line.rstrip("\n").split("\t")
            src_id = line[bdlig_idx]
            smiles = line[smiles_idx]
            # skip repeated entries
            if src_id in done:
                # Parser.__log.debug("skipping line %s: repeated.", idx)
                continue
            done.add(src_id)
            if not smiles:
                # Parser.__log.debug("skipping line %s: missing smiles.", idx)
                continue
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def chebi(map_paths, molrepo_name, chunks=1000):
        try:
            import rdkit.Chem as Chem
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        converter = Converter()

        file_path = map_paths["chebi_lite"]
        suppl = Chem.SDMolSupplier(file_path)
        chunk = list()
        for idx, line in enumerate(suppl):
            if not line:
                continue
            src_id = line.GetPropsAsDict()['ChEBI ID']
            smiles = Chem.MolToSmiles(line, isomericSmiles=True)
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def ctd(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = map_paths["CTD_chemicals_diseases"]
        fh = open(os.path.join(file_path), "r")
        done = set()
        chunk = list()
        for idx, line in enumerate(fh):
            # skip header
            if line.startswith("#"):
                continue
            line = line.rstrip("\n").split("\t")
            # skip those without DirectEvidence
            if not line[5]:
                continue
            chemicalname = line[0]
            chemicalid = line[1]
            src_id = chemicalid
            # skip repeated entries
            if src_id in done:
                # Parser.__log.debug("skipping line %s: repeated.", idx)
                continue
            done.add(src_id)
            # try to conert CTD id to SMILES
            smiles = None
            try:
                smiles = converter.ctd_to_smiles(chemicalid)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
            # if that did't work we can still try with the chamical name
            if not smiles:
                try:
                    smiles = converter.chemical_name_to_smiles(chemicalname)
                except Exception as ex:
                    Parser.__log.warning("line %s: %s", idx, str(ex))
                    continue
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def chembl(map_paths, molrepo_name, chunks=1000):
        converter = Converter()
        # no file to parse here, but querying the chembl database
        query = "SELECT md.chembl_id, cs.canonical_smiles " +\
            "FROM molecule_dictionary md, compound_structures cs " +\
            "WHERE md.molregno = cs.molregno " +\
            "AND cs.canonical_smiles IS NOT NULL"
        cur = psql.qstring_cur(query, molrepo_name)
        chunk = list()
        for idx, row in enumerate(cur):
            src_id = row[0]
            smiles = row[1]
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def drugbank(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = map_paths[molrepo_name]
        if( os.path.isdir(file_path) ):
            fxml = ''
            for fs in os.listdir(file_path) :
                if( fs.endswith('.xml') ):
                    fxml = fs
            file_path = os.path.join(file_path, fxml)
        
        # parse XML
        prefix = "{http://www.drugbank.ca}"
        tree = ET.parse(file_path)
        root = tree.getroot()
        chunk = list()
        for idx, drug in enumerate(root):
            # Drugbank ID
            src_id = None
            for child in drug.findall(prefix + "drugbank-id"):
                if "primary" in child.attrib:
                    if child.attrib["primary"] == "true":
                        src_id = child.text
            if not src_id:
                Parser.__log.warning("line %s: %s", idx, "no drugbank-id")
                continue
            # SMILES
            smiles = None
            for props in drug.findall(prefix + "calculated-properties"):
                for prop in props:
                    if prop.find(prefix + "kind").text == "SMILES":
                        smiles = prop.find(prefix + "value").text
            if not smiles:
                Parser.__log.warning("line %s: %s", idx, "no SMILES")
                continue
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def kegg(map_paths, molrepo_name, chunks=1000):
        try:
            from openbabel import pybel
        except ImportError:
            raise ImportError("requires pybel " +
                              "http://openbabel.org")
        try:
            import wget
        except ImportError:
            raise ImportError("requires wget " +
                              "http://bitbucket.org/techtonik/python-wget/src")
        converter = Converter()

        file_path = map_paths["kegg_br"]
        fh = open(os.path.join(file_path), "r")
        # kegg molecules will be downloaded to following dir
        kegg_download = os.path.join(os.path.dirname(file_path), 'mols')
        if not os.path.isdir(kegg_download):
            os.mkdir(kegg_download)
        done = set()
        chunk = list()
        for idx, line in enumerate(fh):
            if not line.startswith("F"):
                continue
            src_id = line.split()[1]
            # skip repeated entries
            if src_id in done:
                # Parser.__log.debug("skipping line %s: repeated.", idx)
                continue
            done.add(src_id)
            # download mol if not available
            mol_path = os.path.join(kegg_download, '%s.mol' % src_id)
            if not os.path.isfile(mol_path):
                url = "http://rest.kegg.jp/get/" + src_id + "/mol"
                try:
                    wget.download(url, mol_path)
                except Exception:
                    Parser.__log.error('Cannot download: %s', url)
                    continue
            mol = pybel.readfile("mol", mol_path)
            for m in mol:
                smiles = m.write("smi").rstrip("\n").rstrip("\t")
                if not smiles:
                    Parser.__log.warning("line %s: %s", idx, "no SMILES")
                # the following is always the same
                try:
                    inchikey, inchi = converter.smiles_to_inchi(smiles)
                except Exception as ex:
                    Parser.__log.warning("line %s: %s", idx, str(ex))
                    inchikey, inchi = None, None
                id_text = molrepo_name + "_" + src_id
                if inchikey is not None:
                    id_text += ("_" + inchikey)
                result = {
                    "id": id_text,
                    "molrepo_name": molrepo_name,
                    "src_id": src_id,
                    "smiles": smiles,
                    "inchikey": inchikey,
                    "inchi": inchi
                }
                chunk.append(result)
            if len(chunk) >= chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def lincs(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = map_paths["compoundinfo_beta"]
        df = pd.read_csv(file_path, sep='\t')
        df = df[['pert_id', 'canonical_smiles', 'inchi_key']]
        df = df[df['canonical_smiles'] != 'restricted']
        df = df.dropna(subset=['canonical_smiles'])
        df = df.sort_values('pert_id')
        df = df.drop_duplicates(subset=['canonical_smiles'])
        df = df.reset_index(drop=True)

        chunk = list()
        for idx, line in df.iterrows():
            src_id = line['pert_id']
            smiles = line['canonical_smiles']
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def mosaic(map_paths, molrepo_name, chunks=1000):
        try:
            from openbabel import pybel
        except ImportError:
            raise ImportError("requires pybel " +
                              "http://openbabel.org")
        converter = Converter()
        # FIXME find source (hint:/aloy/home/mduran/myscripts/mosaic/D/D3/data)
        # eventually add All_collection to local
        # check input size

        file_path = map_paths["mosaic_all_collections"]
        chunk = list()
        for mol in pybel.readfile("sdf", file_path):
            if not mol:
                continue
            smi, src_id = mol.write("can").rstrip("\n").split("\t")
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("Mosaic ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def morphlincs(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = os.path.join(
            map_paths["morphlincs_LDS-1195"], "LDS-1195/Metadata/Small_Molecule_Metadata.txt")
        g = open(file_path, "r")
        g.readline()
        chunk = list()
        for l in csv.reader(g, delimiter="\t"):
            if not l[6]:
                continue
            src_id = l[8]
            smi = l[6]
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("Morphlincs ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def nci60(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = os.path.join(
            map_paths["DTP_NCI60_ZSCORE"], "output/DTP_NCI60_ZSCORE.xlsx")
        Parser.__log.info("Converting Zscore xls file to csv")
        data_xls = pd.read_excel(file_path, index_col=0)
        csv_path = file_path[:-4] + ".csv"
        data_xls.to_csv(csv_path, encoding='utf-8')
        f = open(csv_path, "r")
        f.readline()
        chunk = list()
        for l in csv.reader(f):
            src_id, smi = l[0], l[5]
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("NCI60 ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def pdb(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = map_paths["pdb_components"]
        chunk = list()
        f = open(file_path, "r")
        for l in f:
            l = l.rstrip("\n").split("\t")
            if len(l) < 2:
                continue
            src_id = l[1]
            smi = l[0]
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("PDB ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def sider(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        sider_file = ""
        stitch_file = ""
        chunk = list()
        for file in map_paths.values():
            if "meddra_all_se" in file:
                sider_file = file
                continue
            if "chemicals" in file:
                stitch_file = file

        if sider_file == "" or stitch_file == "":
            raise Exception("Missing expected input files")

        with open(sider_file, "r") as f:
            S = set()
            for l in f:
                l = l.split("\t")
                S.update([l[1]])

        with open(stitch_file, "r") as f:
            stitch = {}
            f.readline()
            for r in csv.reader(f, delimiter="\t"):
                if r[0] not in S:
                    continue
                stitch[r[0]] = r[-1]

        for s in list(S):
            src_id = s
            smi = stitch[s]
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("SIDER ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def smpdb(map_paths, molrepo_name, chunks=1000):
        try:
            from openbabel import pybel
        except ImportError:
            raise ImportError("requires pybel " +
                              "http://openbabel.org")
        converter = Converter()

        file_path = os.path.join(
            map_paths["smpdb_structures"], "smpdb_structures")
        S = set()
        L = os.listdir(file_path)
        chunk = list()
        for l in L:
            for mol in pybel.readfile("sdf", file_path + "/" + l):
                if not mol:
                    continue
                smi, Id = mol.write("can").rstrip("\n").split("\t")
                S.update([(Id, smi)])

        for s in sorted(S):
            src_id = s[0]
            smi = s[1]
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("SMPDB ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def biur_real(map_paths, molrepo_name, chunks=1000):
        try:
            import rdkit.Chem as Chem
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        converter = Converter()

        file_path = map_paths[molrepo_name]
        chunk = list()
        suppl = Chem.SDMolSupplier(file_path)
        for mol in suppl:
            if not mol:
                continue
            src_id = mol.GetProp("_Name")
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("biur_real ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def biur_virtual(map_paths, molrepo_name, chunks=1000):
        try:
            import rdkit.Chem as Chem
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        converter = Converter()

        file_path = os.path.join(
            str(map_paths[molrepo_name]), "VIRTUAL_BIUR_POR_MW")
        chunk = list()
        sdf_files = [f for f in os.listdir(file_path) if f[-4:] == ".sdf"]
        for sdf_file in sdf_files:
            suppl = Chem.SDMolSupplier(file_path + "/" + sdf_file)
            for mol in suppl:

                src_id = mol.GetProp("_Name")
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)

                try:
                    inchikey, inchi = converter.smiles_to_inchi(smi)
                except Exception as ex:
                    Parser.__log.warning(
                        "biur_virtual ID %s: %s", src_id, str(ex))
                    inchikey, inchi = None, None
                id_text = molrepo_name + "_" + src_id
                if inchikey is not None:
                    id_text += ("_" + inchikey)

                result = {
                    "id": id_text,
                    "molrepo_name": molrepo_name,
                    "src_id": src_id,
                    "smiles": smi,
                    "inchikey": inchikey,
                    "inchi": inchi
                }
                chunk.append(result)
                if len(chunk) == chunks:
                    yield chunk
                    chunk = list()
        yield chunk

    @staticmethod
    def cmaup(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = map_paths[molrepo_name]
        chunk = list()
        f = open(file_path, "r")
        for l in f:
            l = l.rstrip("\n").split("\t")
            if len(l) < 2:
                continue
            src_id = l[0]
            smi = l[-1]
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("CMAUP ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def repohub(map_paths, molrepo_name, chunks=1000):
        converter = Converter()

        file_path = map_paths[molrepo_name]
        chunk = list()
        f = open(file_path, "r")
        for l in f:
            l = l.rstrip("\n").split("\t")
            if len(l) < 2:
                continue
            src_ids = l[7].split(", ")
            smis = l[8].split(", ")
            for (src_id, smi) in zip(src_ids, smis):
                try:
                    inchikey, inchi = converter.smiles_to_inchi(smi)
                except Exception as ex:
                    Parser.__log.warning("RepoHub ID %s: %s", src_id, str(ex))
                    inchikey, inchi = None, None
                id_text = molrepo_name + "_" + src_id
                if inchikey is not None:
                    id_text += ("_" + inchikey)
                result = {
                    "id": id_text,
                    "molrepo_name": molrepo_name,
                    "src_id": src_id,
                    "smiles": smi,
                    "inchikey": inchikey,
                    "inchi": inchi
                }
                chunk.append(result)
                if len(chunk) == chunks:
                    yield chunk
                    chunk = list()
        yield chunk

    @staticmethod
    def hmdb(map_paths, molrepo_name, chunks=1000):
        from lxml import etree as ET
        converter = Converter()
        # Functions

        def fast_iter(context, func):
            for event, elem in context:
                yield func(elem)
                elem.clear()
                for ancestor in elem.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]
            del context

        def process_elem(elem):
            src_id = elem.find(ns + "accession")
            smiles = elem.find(ns + "smiles")
            if src_id is None or smiles is None:
                return None, None
            return src_id.text, smiles.text

        file_path = map_paths["hmdb_metabolites"]
        ns = "{http://www.hmdb.ca}"
        chunk = list()
        idx = 0
        # parse XML
        context = ET.iterparse(file_path, events=(
            "end", ), tag=ns + "metabolite")
        for src_id, smiles in fast_iter(context, process_elem):
            if src_id is None or smiles is None:
                continue
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            idx += 1
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def pharmacodb(file_path, molrepo_name, chunks=1000):
        from pubchempy import Compound
        converter = Converter()
        # no file to parse here, but querying the chembl database
        query = "SELECT drug_id, smiles, pubchem FROM drug_annots WHERE smiles IS NOT NULL or pubchem IS NOT NULL"
        # new query from chembl
        #query = "select cr.MOLREGNO as drug_id, cr.SRC_COMPOUND_ID as pubchem, cs.CANONICAL_SMILES as smiles from COMPOUND_RECORDS as cr, COMPOUND_STRUCTURES as cs where cr.MOLREGNO=cs.MOLREGNO and cr.SRC_ID=(select src_id from source where SRC_SHORT_NAME like '%PUBCHEM%')"
        cur = psql.qstring_cur(query, molrepo_name)
        chunk = list()
        for idx, row in enumerate(cur):
            src_id = "pharmacodb_%d" % row[0]
            smiles = row[1]
            pubchem = row[2]
            if (smiles is None or smiles == "-666") and pubchem is not None:
                try:
                    smiles = Compound.from_cid(pubchem).isomeric_smiles
                except:
                    continue
            if smiles is None or smiles == "-666":
                continue
            # the following is always the same
            try:
                inchikey, inchi = converter.smiles_to_inchi(smiles)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def touchstone(map_paths, molrepo_name, chunks=1000):
        converter = Converter()
        file_path = map_paths["GSE92742_Broad_LINCS_pert_info"]
        chunk = list()
        f = open(file_path, "r")
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        istouch_idx = header.index("is_touchstone")
        pertid_idx = header.index("pert_id")
        pertype_idx = header.index("pert_type")
        smiles_idx = header.index("canonical_smiles")
        for r in reader:
            if r[istouch_idx] != "1":
                continue
            if r[pertype_idx] != "trt_cp":
                continue
            src_id = r[pertid_idx]
            smi = r[smiles_idx]
            if smi == "-666":
                continue
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("Touchstone ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def zinc(map_paths, molrepo_name, chunks=1000):
        converter = Converter()
        file_path = map_paths[molrepo_name]
        f = open(file_path, "r")
        delimiter = '\t'
        index_smi = 0
        index_id = 1
        min_items = 2
        if molrepo_name == 'tool':
            delimiter = ' '
            index_smi = 0
            index_id = 2
            min_items = 3
            f.readline()

        chunk = list()

        for l in f:
            l = l.rstrip("\n").split(delimiter)
            if len(l) < min_items:
                continue
            src_id = l[index_id]
            smi = l[index_smi]
            try:
                inchikey, inchi = converter.smiles_to_inchi(smi)
            except Exception as ex:
                Parser.__log.warning("ZINC ID %s: %s", src_id, str(ex))
                inchikey, inchi = None, None
            id_text = molrepo_name + "_" + src_id
            if inchikey is not None:
                id_text += ("_" + inchikey)
            result = {
                "id": id_text,
                "molrepo_name": molrepo_name,
                "src_id": src_id,
                "smiles": smi,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk
