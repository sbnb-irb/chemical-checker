import os
import csv
import wget
import pandas as pd
import xml.etree.ElementTree as ET

from .converter import Converter
from chemicalchecker.util import logged
from chemicalchecker.util import psql


@logged
class Parser():
    """Container for static parsing methods.

    A parsing function here is iterating on an input file. It has to define
    on each input line the source id and the smile of  a molecule. Then the
    smile is converted to inchi and inchikey. The lines are appended as
    dictionaies and yielded in chunks.
    """

    @staticmethod
    def parse_fn(function):
        try:
            return eval('Parser.' + function)
        except Exception as ex:
            Parser.__log.error("Cannot find parsing function %s", function)
            raise ex

    @staticmethod
    def bindingdb(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
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
                inchikey, inchi = Converter.smiles_to_inchi(smiles)
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
    def chebi(file_path, molrepo_name, chunks=1000):
        try:
            import rdkit.Chem as Chem
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
        suppl = Chem.SDMolSupplier(file_path)
        chunk = list()
        for idx, line in enumerate(suppl):
            if not line:
                continue
            src_id = line.GetPropsAsDict()['ChEBI ID']
            smiles = Chem.MolToSmiles(line)
            # the following is always the same
            try:
                inchikey, inchi = Converter.smiles_to_inchi(smiles)
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
    def ctd(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
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
                smiles = Converter.ctd_to_smiles(chemicalid)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
            # if that did't work we can still try with the chamical name
            if not smiles:
                try:
                    smiles = Converter.chemical_name_to_smiles(chemicalname)
                except Exception as ex:
                    Parser.__log.warning("line %s: %s", idx, str(ex))
                    continue
            # the following is always the same
            try:
                inchikey, inchi = Converter.smiles_to_inchi(smiles)
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
    def chembl(file_path, molrepo_name, chunks=1000):
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
                inchikey, inchi = Converter.smiles_to_inchi(smiles)
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
    def drugbank(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
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
                inchikey, inchi = Converter.smiles_to_inchi(smiles)
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
    def kegg(file_path, molrepo_name, chunks=1000):
        try:
            import pybel
        except ImportError:
            raise ImportError("requires pybel " +
                              "http://openbabel.org")
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
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
            mol = pybel.readfile("mol", mol_path)
            for m in mol:
                smiles = m.write("smi").rstrip("\n").rstrip("\t")
                if not smiles:
                    Parser.__log.warning("line %s: %s", idx, "no SMILES")
                # the following is always the same
                try:
                    inchikey, inchi = Converter.smiles_to_inchi(smiles)
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
    def lincs(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 2:
            raise Exception("This parser expect 2 input files.")
        # skip header
        chunk = list()

        for file in file_path:
            col = -1
            if "GSE92742" in file:
                col = 6
            if "GSE70138" in file:
                col = 1

            if col < 0:
                raise Exception("Missing expected input files")
            fh = open(file, "r")
            fh.readline()
            for idx, line in enumerate(csv.reader(fh, delimiter="\t")):
                if not line[col] or line[col] == "-666":
                    continue
                src_id = line[0]
                smiles = line[col]
                # the following is always the same
                try:
                    inchikey, inchi = Converter.smiles_to_inchi(smiles)
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
    def mosaic(file_path, molrepo_name, chunks=1000):
        try:
            import pybel
        except ImportError:
            raise ImportError("requires pybel " +
                              "http://openbabel.org")
        # FIXME find source (hint:/aloy/home/mduran/myscripts/mosaic/D/D3/data)
        # eventually add All_collection to local
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
        chunk = list()
        for mol in pybel.readfile("sdf", file_path):
            if not mol:
                continue
            smi, src_id = mol.write("can").rstrip("\n").split("\t")
            try:
                inchikey, inchi = Converter.smiles_to_inchi(smi)
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
    def morphlincs(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
        g = open(file_path, "r")
        g.readline()
        chunk = list()
        for l in csv.reader(g, delimiter="\t"):
            if not l[6]:
                continue
            src_id = l[8]
            smi = l[6]
            try:
                inchikey, inchi = Converter.smiles_to_inchi(smi)
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
    def nci60(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
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
                inchikey, inchi = Converter.smiles_to_inchi(smi)
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
    def pdb(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
        chunk = list()
        f = open(file_path, "r")
        for l in f:
            l = l.rstrip("\n").split("\t")
            if len(l) < 2:
                continue
            src_id = l[1]
            smi = l[0]
            try:
                inchikey, inchi = Converter.smiles_to_inchi(smi)
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
    def sider(file_path, molrepo_name, chunks=1000):
        # check input size
        if len(file_path) != 2:
            raise Exception("This parser expect 2 input files.")
        sider_file = ""
        stitch_file = ""
        chunk = list()
        for file in file_path:
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
                inchikey, inchi = Converter.smiles_to_inchi(smi)
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
    def smpdb(file_path, molrepo_name, chunks=1000):
        try:
            import pybel
        except ImportError:
            raise ImportError("requires pybel " +
                              "http://openbabel.org")
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
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
                inchikey, inchi = Converter.smiles_to_inchi(smi)
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
    def biur_real(file_path, molrepo_name, chunks=1000):
        try:
            import rdkit.Chem as Chem
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
        chunk = list()
        suppl = Chem.SDMolSupplier(file_path)
        for mol in suppl:
            if not mol:
                continue
            src_id = mol.GetProp("_Name")
            smi = Chem.MolToSmiles(mol)
            try:
                inchikey, inchi = Converter.smiles_to_inchi(smi)
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
    def biur_virtual(file_path, molrepo_name, chunks=1000):
        try:
            import rdkit.Chem as Chem
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        # check input size
        if len(file_path) != 1:
            raise Exception("This parser expect a single input file.")
        file_path = file_path[0]
        chunk = list()
        sdf_files = [f for f in os.listdir(file_path) if f[-4:] == ".sdf"]
        for sdf_file in sdf_files:
            suppl = Chem.SDMolSupplier(file_path + "/" + sdf_file)
            for mol in suppl:

                src_id = mol.GetProp("_Name")
                smi = Chem.MolToSmiles(mol)

                try:
                    inchikey, inchi = Converter.smiles_to_inchi(smi)
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
