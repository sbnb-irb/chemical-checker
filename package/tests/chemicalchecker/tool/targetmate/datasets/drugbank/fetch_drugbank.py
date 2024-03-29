"""Get target data from a local xml DrugBank database.
Adapted from CC B1.001 preprocessing.
"""
import os
import pandas as pd
from chemicalchecker.tool.targetmate.utils.chemistry import read_smiles
from chemicalchecker.util import psql
from chemicalchecker.util import logged

from chemicalchecker.database import Molrepo
import xml.etree.ElementTree as ET
import collections
import numpy as np

class DrugBankDb():

    def __init__(self,
                  drugbank_xml="/aloy/scratch/ptorren/databases/drugbank/full_database.xml",
                  dirs = None):
        self.drugbank_xml = os.path.abspath(drugbank_xml)
        if dirs is None:
            self.dirs = {
            'Inhibitor': -1,
            'acetylation': -1,
            'activator': +1,
            'agonist': +1,
            'antagonist': -1,
            'binder': -1,
            'binding': -1,
            'blocker': -1,
            'cofactor': +1,
            'inducer': +1,
            'inhibitor': -1,
            'inhibitor, competitive': -1,
            'inhibitory allosteric modulator': -1,
            'intercalation': -1,
            'inverse agonist': +1,
            'ligand': -1,
            'negative modulator': -1,
            'partial agonist': +1,
            'partial antagonist': -1,
            'positive allosteric modulator': +1,
            'positive modulator': +1,
            'potentiator': +1,
            'stimulator': -1,
            'suppressor': -1}
        else:
            self.dirs = dirs
        self.prefix = "{http://www.drugbank.ca}"

        dbid_inchikey = {}
        inchikey_srcid = {}
        inchikey_smiles = {}
        molrepos = Molrepo.get_by_molrepo_name("drugbank")
        for molrepo in molrepos:
            if not molrepo.inchikey:
                continue
            dbid_inchikey[molrepo.src_id] = molrepo.inchikey
            inchikey_srcid[molrepo.inchikey] = molrepo.src_id
            inchikey_smiles[molrepo.inchikey] = molrepo.smiles
        self.dbid_inchikey = dbid_inchikey
        self.inchikey_srcid = inchikey_srcid
        self.inchikey_smiles = inchikey_smiles

    def decide(self, acts):
        if len(acts) == 0:
            return 0
        m = np.mean(acts)
        if m > 0:
            return 1
        else:
            return -1

    def get_inchikey(self, drug):
        db_id = None
        for child in drug.findall(self.prefix + "drugbank-id"):
            if "primary" in child.attrib:
                if child.attrib["primary"] == "true":
                    db_id = child.text

        if db_id not in self.dbid_inchikey:
            return None
        inchikey = self.dbid_inchikey[db_id]
        return inchikey

    def get_actions(self, targ):
        actions = []
        for action in targ.findall(self.prefix + "actions"):
            for child in action:
                actions += [child.text]
        return actions

    def get_uniprot(self, targ):
        uniprot_ac = None
        prot = targ.find(self.prefix + "polypeptide")
        if not prot:
            return None
        if "source" in prot.attrib:
            if prot.attrib["source"] == "Swiss-Prot":
                uniprot_ac = prot.attrib["id"]
        if not uniprot_ac:
            return None
        return uniprot_ac

    def get_targets(self, drug):
        targets = collections.defaultdict(list)
        uniprot_ac = None
        for targs in drug.findall(self.prefix + "targets"):
            for targ in targs.findall(self.prefix + "target"):
                actions = self.get_actions(targ)
                uniprot_ac = self.get_uniprot(targ)
                if not uniprot_ac: continue
                targets[uniprot_ac] = actions
        return targets

    def parse_drugbank(self):
        ACTS = collections.defaultdict(list)
        tree = ET.parse(self.drugbank_xml)
        root = tree.getroot()
        DB = {}
        for drug in root:
            inchikey = self.get_inchikey(drug) ## LOOK INTO THIS
            targets = self.get_targets(drug)
            if not inchikey: continue
            if not targets: continue
            DB[inchikey] = targets
        return DB

    def store_activities(self, explicit):
        ACTS = []
        DB = self.parse_drugbank()
        for inchikey, targs in DB.items():
            for uniprot_ac, actions in targs.items():
                d = []
                for action in actions:
                    if action in self.dirs:
                        d += [self.dirs[action]]
                if explicit:
                    if not d:
                        continue
                act = self.decide(d)
                ACTS.append((inchikey, uniprot_ac, self.inchikey_srcid[inchikey], act))
        return ACTS

class DrugBank(DrugBankDb):

    def __init__(self,
                 output_folder,
                 min_actives=10,
                uniprot_acs = None,
                 multiclass = True
                ):
        DrugBankDb.__init__(self)
        if output_folder:
            self.output_folder = os.path.abspath(output_folder)
            if not os.path.isdir(self.output_folder):
                os.mkdir(self.output_folder)
        self.min_actives = min_actives
        self.uniprot_acs = uniprot_acs
        self.multiclass = multiclass
        if self.multiclass:
            self.file_name = "multiclass"
        else:
            self.file_name = "general"

    def create_folder_hierarchy(self, uniprot_ac):
        path = os.path.join(self.output_folder, uniprot_ac)
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def get_activities(self, data):
        if self.multiclass:
            dat = [[act, self.inchikey_smiles[inchi], self.inchikey_srcid[inchi], inchi] for (inchi, act) in data]
        else:
            dat = [[1, self.inchikey_smiles[inchi], self.inchikey_srcid[inchi], inchi] for (inchi, act) in data]

        return pd.DataFrame(dat)

    def write_to_file(self, df, path):
        df.to_csv(os.path.join(path, "{:s}.tsv".format(self.file_name)), sep='\t', header=False, index=False)

    def parse_activities(self, explicit=True, store=True):
        molrepos = Molrepo.get_by_molrepo_name("drugbank")

        ACTS = super().store_activities(explicit)
        by_protein = collections.defaultdict(list)
        for (inchikey, uniprot_ac, srcid, act) in ACTS:
            if by_protein[uniprot_ac]:
                by_protein[uniprot_ac].append((inchikey, act))
            else:
                 by_protein[uniprot_ac] = [(inchikey, act)]

        if not store:
            return by_protein

        data = []
        for target in by_protein:
            if self.uniprot_acs is not None:
                if target not in self.uniprot_acs: continue
            if self.min_actives:
                if len(by_protein[target]) < self.min_actives: continue
            path = self.create_folder_hierarchy(target)
            df = self.get_activities(by_protein[target])
            self.write_to_file(df, path)
