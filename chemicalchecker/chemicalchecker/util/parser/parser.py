import os
from .convert import Converter
from chemicalchecker.util import logged
import rdkit.Chem as Chem


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
    def bindingdb(file_path, src_name, chunks):
        fh = open(os.path.join(file_path), "r")
        # skip header
        header = fh.next()
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
            smile = line[smiles_idx]
            # skip repeated entries
            if src_id in done:
                # Parser.__log.debug("skipping line %s: repeated.", idx)
                continue
            done.add(src_id)
            if not smile:
                # Parser.__log.debug("skipping line %s: missing smile.", idx)
                continue
            # the following is always the same
            try:
                inchikey, inchi = Converter.smile_to_inchi(smile)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = "", ""
            result = {
                "src_name": src_name,
                "src_id": src_id,
                "smile": smile,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def chebi(file_path, src_name, chunks):
        suppl = Chem.SDMolSupplier(file_path)
        chunk = list()
        for idx, line in enumerate(suppl):
            if not line:
                continue
            src_id = line.GetPropsAsDict()['ChEBI ID']
            smile = Chem.MolToSmiles(line)
            # the following is always the same
            try:
                inchikey, inchi = Converter.smile_to_inchi(smile)
            except Exception as ex:
                Parser.__log.warning("line %s: %s", idx, str(ex))
                inchikey, inchi = "", ""
            result = {
                "src_name": src_name,
                "src_id": src_id,
                "smile": smile,
                "inchikey": inchikey,
                "inchi": inchi
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

"""
def chembl():
    # FIXME include loadChemblDB.py
    with open(moldir + "/chembl.tsv", "w") as f:

        query = "SELECT md.chembl_id, cs.canonical_smiles FROM molecule_dictionary md, compound_structures cs WHERE md.molregno = cs.molregno AND cs.canonical_smiles IS NOT NULL"

        con = Psql.connect(checkerconfig.chembl)
        con.set_isolation_level(0)
        cur = con.cursor()
        cur.execute(query)
        for r in cur:
            Id = r[0]
            smi = r[1]
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]

            f.write("%s\t%s\t%s\t%s\n" %
                    (Id, smi.replace("\n", "\\n"), inchikey, inchi))


def ctd():
    # FIXME include createChemicalsList.py
    f = open(os.path.join(downloadsdir, checkerconfig.ctd_molecules_download), "r")
    g = open(moldir + "/ctd.tsv", "w")
    for l in csv.reader(f, delimiter="\t"):
        if len(l) < 2:
            continue
        Id = l[0]
        smi = l[1]
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
    g.close()
    f.close()


def drugbank():

    # Parse Drugbank and convert to inchikeys.

    import xml.etree.ElementTree as ET

    xmlfile = os.path.join(downloadsdir, checkerconfig.drugbank_download)

    prefix = "{http://www.drugbank.ca}"

    tree = ET.parse(xmlfile)

    root = tree.getroot()

    with open(moldir + "/drugbank.tsv", "w") as f:

        for drug in root:

            # Drugbank ID

            db_id = None
            for child in drug.findall(prefix + "drugbank-id"):
                if "primary" in child.attrib:
                    if child.attrib["primary"] == "true":
                        db_id = child.text

            if not db_id:
                continue

            # Smiles

            smiles = None
            for props in drug.findall(prefix + "calculated-properties"):
                for prop in props:
                    if prop.find(prefix + "kind").text == "SMILES":
                        smiles = prop.find(prefix + "value").text
            if not smiles:
                continue

            smi = smiles
            Id = db_id

            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def kegg():
    # FIXME include last lines from downloadFiles.py
    with open(moldir + "/kegg.tsv", "w") as f:
        L = os.listdir(os.path.join(
            downloadsdir, checkerconfig.kegg_mol_folder_download))
        for l in L:
            mol = pybel.readfile("mol", os.path.join(
                downloadsdir, checkerconfig.kegg_mol_folder_download) + "/" + l)
            for m in mol:
                smi = m.write("smi").rstrip("\n").rstrip("\t")
                if ".mol" not in l:
                    continue
                Id = l.split(".")[0]
                if not smi:
                    continue
                mol = hts.apply(smi)
                if not mol:
                    inchikey = ""
                    inchi = ""
                else:
                    inchikey = mol[0]
                    inchi = mol[1]
                f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def lincs():
    # FIXME split in two
    S = set()

    with open(os.path.join(downloadsdir, checkerconfig.lincs_GSE70138_pert_info_download), "r") as f:
        f.next()
        for r in csv.reader(f, delimiter="\t"):
            if not r[1] or r[1] == "-666":
                continue
            S.update([(r[0], r[1])])

    with open(os.path.join(downloadsdir, checkerconfig.lincs_GSE92742_pert_info_download), "r") as f:
        f.next()
        for r in csv.reader(f, delimiter="\t"):
            if not r[6] or r[6] == "-666":
                continue
            S.update([(r[0], r[6])])

    with open(moldir + "/lincs.tsv", "w") as f:
        for s in sorted(S):
            Id = s[0]
            smi = s[1]
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def mosaic():
    #FIXME find source (hint: /aloy/home/mduran/myscripts/mosaic/D/D3/data) eventually add All_collection to local
    with open(moldir + "/mosaic.tsv", "w") as f:
        for mol in pybel.readfile("sdf", os.path.join(downloadsdir, checkerconfig.mosaic_all_collections_download)):
            if not mol:
                continue
            smi, Id = mol.write("can").rstrip("\n").split("\t")
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def morphlincs():
    f = open(moldir + "/morphlincs.tsv", "w")
    g = open(os.path.join(downloadsdir,
                          checkerconfig.morphlincs_molecules_download), "r")
    g.next()
    for l in csv.reader(g, delimiter="\t"):
        if not l[6]:
            continue
        Id = l[8]
        smi = l[6]
        if not smi:
            continue
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
    g.close()
    f.close()


def nci60():
    #FIXME the downloadfile is xls but in the pipeline is .csv, check where conversion happens
    f = open(os.path.join(downloadsdir, checkerconfig.nci60_download), "r")
    g = open(moldir + "/nci60.tsv", "w")
    f.next()
    for l in csv.reader(f):
        Id, smi = l[0], l[5]
        if not smi:
            continue
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
    g.close()
    f.close()


def pdb():

    ligand_inchikey = {}
    inchikey_inchi = {}
    f = open(os.path.join(downloadsdir,
                          checkerconfig.pdb_components_smiles_download), "r")
    g = open(moldir + "/pdb.tsv", "w")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if len(l) < 2:
            continue
        lig_id = l[1]
        mol = hts.apply(l[0])
        if not mol:
            g.write("%s\t%s\t\t\n" % (lig_id, l[0]))
            continue
        ligand_inchikey[lig_id] = mol[0]
        inchikey_inchi[mol[0]] = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (lig_id, l[0], mol[0], mol[1]))
    f.close()
    g.close()


def sider():

    with open(os.path.join(downloadsdir, checkerconfig.sider_download), "r") as f:
        S = set()
        for l in f:
            l = l.split("\t")
            S.update([l[1]])

    with open(os.path.join(downloadsdir, checkerconfig.stitch_molecules_download), "r") as f:
        stitch = {}
        f.next()
        for r in csv.reader(f, delimiter="\t"):
            if r[0] not in S:
                continue
            stitch[r[0]] = r[-1]

    with open(moldir + "/sider.tsv", "w") as f:
        for s in list(S):
            Id = s
            smi = stitch[s]
            if not smi:
                continue
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def smpdb():

    g = open(moldir + "/smpdb.tsv", "w")
    S = set()
    L = os.listdir(os.path.join(
        downloadsdir, checkerconfig.smpdb_folder_download))
    for l in L:
        for mol in pybel.readfile("sdf", os.path.join(downloadsdir, checkerconfig.smpdb_folder_download) + "/" + l):
            if not mol:
                continue
            smi, Id = mol.write("can").rstrip("\n").split("\t")
            S.update([(Id, smi)])

    for s in sorted(S):
        Id = s[0]
        smi = s[1]
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))

    g.close()
"""
