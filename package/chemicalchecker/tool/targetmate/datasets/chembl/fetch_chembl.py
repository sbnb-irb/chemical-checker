"""Get target data from a local PostGreSQL ChEMBL database.
"""
import os
import pandas as pd
from ..universes import load_universe
from ..utils.chemistry import read_smiles

from chemicalchecker.util import psql


class ChemblDb:

    def __init__(self, dbname='chembl_25'):
        """Class to fetch data from a local ChEMBL database.

        Args:
            dbname (str): ChEMBL database name (default='chembl_25').
        """
        self.dbname = dbname

    def get_universe(self):
        query = '''
            SELECT s.canonical_smiles, m.chembl_id
                FROM compound_structures s, molecule_dictionary m
                WHERE
                    s.molregno = m.molregno
                    AND s.canonical_smiles IS NOT NULL
                    AND m.chembl_id IS NOT NULL;
        '''
        return psql.qstring(query, self.dbname)

    def get_targets(self):
        query = '''
            SELECT chembl_id, target_type, pref_name, tax_id, organism
                FROM target_dictionary;
        '''
        results = psql.qstring(query, self.dbname)
        col_names = ["target_id", "target_type",
                     "pref_name", "tax_id", "organism"]
        return pd.DataFrame(results, columns=col_names)

    def _get_activities(self, chembl_ids, entity, only_pchembl):
        query = '''
            SELECT
                a.chembl_id,
                a.assay_type,
                t.chembl_id,
                m.chembl_id,
                s.canonical_smiles,
                act.pchembl_value
            FROM
                molecule_dictionary m,
                compound_structures s,
                activities act,
                assays a,
                target_dictionary t
            WHERE
                m.molregno = s.molregno
                AND m.molregno = act.molregno
                AND act.assay_id = a.assay_id
                AND a.tid = t.tid
                AND s.canonical_smiles IS NOT NULL
        '''
        if only_pchembl:
            query += " AND act.pchembl_value IS NOT NULL"
        if type(chembl_ids) is str:
            query += " AND %s.chembl_id = '%s'" % (entity, chembl_ids)
        else:
            chembl_ids_list = ",".join(["'%s'" % t for t in chembl_ids])
            query += " AND %s.chembl_id IN (%s)" % (entity, chembl_ids_list)
        results = psql.qstring(query, self.dbname)
        col_names = ["assay_id", "assay_type", "target_id", "molecule_id",
                     "canonical_smiles", "pchembl_value"]
        return pd.DataFrame(results, columns=col_names)

    def get_molecule_activities(self, molecule_chembl_ids, only_pchembl=False):
        return self._get_activities(molecule_chembl_ids, entity="m",
                                    only_pchembl=only_pchembl)

    def get_target_activities(self, target_chembl_ids, only_pchembl=False):
        return self._get_activities(target_chembl_ids, entity="t",
                                    only_pchembl=only_pchembl)

    def get_assay_activities(self, assay_chembl_ids, only_pchembl=False):
        return self._get_activities(assay_chembl_ids, entity="a",
                                    only_pchembl=only_pchembl)


class Chembl(ChemblDb):

    def __init__(self, output_folder=".", universe_path=None,
                 min_actives=10, inactives_per_active=None,
                 pchembl_values=[5, 6, 7], only_pchembl=True,
                 standardize=True):
        """Query ChEMBL and produce a hierarchy of active/inactive data.

        Args:
            universe_path (str): Universe path (default=None)
            min_actives (int): Minimum number of actives for an training set to
                be considered (default=10).
            inactives_per_active (int): Number of inactives to sample for each
                active. If not specified, no sampling is done (default=None).
            pchembl_cuts (list): Chosen pchembl scores to divide actives and
                inactives (default=[5,6,7]).
            only_pchembl (bool): Keep values without a pchembl score and assume
                they are positives (default=True).
            standardize (bool): Standardize molecules (default=True).
        """
        ChemblDb.__init__(self)
        self.output_folder = output_folder
        if universe_path is None:
            self.universe = None
            if inactives_per_active is None:
                raise ValueError('`universe_path` must be specified when ' +
                                 '`inactives_per_active` is requested.')
        else:
            self.universe = load_universe(universe_path)
        self.min_actives = min_actives
        self.inactives_per_active = inactives_per_active
        self.only_pchembl = only_pchembl
        self.pchembl_values = sorted(
            set([round(x, 2) for x in pchembl_values]))
        if not self.only_pchembl:
            self.pchembl_values += [None]
        self.standardize = standardize
        self.activities = {}

    def _process_smiles(self, df):
        inchikeys = []
        smiles = []
        idxs = []
        for idx, smi in df["canonical_smiles"].items():
            smi = read_smiles(smi, standardize=self.standardize)
            if not smi:
                continue
            inchikeys += [smi[0]]
            smiles += [smi[1]]
            idxs += [idx]
        df = df.loc[idxs]
        df["inchikey"] = inchikeys
        df["smiles"] = smiles
        df = df[["assay_id", "assay_type", "target_id",
                 "molecule_id", "smiles", "inchikey", "pchembl_value"]]
        return df

    @staticmethod
    def _to_set(df):
        values = [tuple(x)
                  for x in df[["smiles", "molecule_id", "inchikey"]].values]
        d = {}
        for v in values:
            d[v[-1]] = v
        values = set([v for k, v in d.items()])
        return values

    def get_molecule_activities(self, molecule_chembl_ids):
        df = super().get_molecule_activities(
            molecule_chembl_ids, only_pchembl=self.only_pchembl)
        return self._process_smiles(df)

    def get_target_activities(self, target_chembl_ids):
        df = super().get_target_activities(
            target_chembl_ids, only_pchembl=self.only_pchembl)
        return self._process_smiles(df)

    def get_assay_activities(self, assay_chembl_id):
        df = super().get_assay_activities(
            assay_chembl_id, only_pchembl=self.only_pchembl)
        return self._process_smiles(df)

    def no_universe_predict(self, actives, inactives):
        common_iks = set([smi[-1] for smi in actives]
                         ).intersection([smi[-1] for smi in inactives])
        actives = set([smi for smi in actives if smi[-1] not in common_iks])
        if len(actives) < self.min_actives:
            return None
        inactives = set(
            [smi for smi in inactives if smi[-1] not in common_iks])
        return actives, inactives, set()

    def get_activities(self, df, pchembl_value=None):
        if pchembl_value:
            dfa = df[df["pchembl_value"] >= pchembl_value]
            dfi = df[df["pchembl_value"] < pchembl_value]
        else:
            dfa = df
            dfi = df[df["pchembl_value"] < -666]
        actives = self._to_set(dfa)
        inactives = self._to_set(dfi)
        if self.universe is None:
            results = self.no_universe_predict(actives, inactives)
        else:
            results = self.universe.predict(actives, inactives,
                                            self.inactives_per_active,
                                            self.min_actives)
        if not results:
            return None
        actives, inactives, putative_inactives = results
        R = []
        for r in actives:
            R += [[1, r[0], r[1], r[2]]]
        for r in inactives:
            R += [(-1, r[0], r[1], r[2])]
        for r in putative_inactives:
            R += [(0, r[0], r[1], r[2])]
        df_ = pd.DataFrame(R, columns=["activity", "smiles", "id", "inchikey"])
        df_ = df_.sample(frac=1).reset_index(drop=True)
        return df_

    @staticmethod
    def to_csv(df, file_name):
        file_name = os.path.abspath(file_name)
        df.to_csv(file_name, sep="\t", header=False, index=False)

    @staticmethod
    def pchembl_filename(pchembl_value):
        if not pchembl_value:
            return "pchembl_NA.csv"
        else:
            return "pchembl_%d.csv" % (pchembl_value * 100)

    def write_every_pchembl(self, df, folder):
        to_write = []
        done = []
        for pchembl_value in self.pchembl_values:
            df_ = self.get_activities(df, pchembl_value=pchembl_value)
            if df_ is None:
                continue
            to_write += [(df_, os.path.join(
                folder, self.pchembl_filename(pchembl_value)))]
            done += [pchembl_value]
        if not to_write:
            return done
        if not os.path.exists(folder):
            os.mkdir(folder)
        for tw in to_write:
            self.to_csv(tw[0], tw[1])
        return done

    @staticmethod
    def _summary_update(summary, sumr, sump):
        for sr in sumr:
            for sp in sump:
                summary += [sr + [sp]]
        return summary

    def write_folder_hierarchy(self):
        summary = []
        from tqdm import tqdm
        dft = self.get_targets()
        for idx, target_chembl_id in tqdm(dft["target_id"][:2].items()):
            df = self.get_target_activities(target_chembl_id)
            folder = os.path.join(self.output_folder, target_chembl_id)
            sumr = [[target_chembl_id, None, None]]
            sump = self.write_every_pchembl(df, folder)
            summary = self._summary_update(summary, sumr, sump)
            assay_types = pd.unique(df["assay_type"])
            for assay_type in assay_types:
                folder_a = os.path.join(folder, assay_type)
                df_a = df[df["assay_type"] == assay_type]
                sumr = [[target_chembl_id, assay_type, None]]
                sump = self.write_every_pchembl(df_a, folder_a)
                summary = self._summary_update(summary, sumr, sump)
                assay_ids = pd.unique(df_a["assay_id"])
                for assay_id in assay_ids:
                    folder_b = os.path.join(folder_a, assay_id)
                    df_b = df_a[df_a["assay_id"] == assay_id]
                    sumr = [[target_chembl_id, assay_type, assay_id]]
                    sump = self.write_every_pchembl(df_b, folder_b)
                    summary = self._summary_update(summary, sumr, sump)
        col_names = ["target_id", "assay_type", "assay_id", "pchembl_value"]
        summary = pd.DataFrame(summary, columns=col_names)
        summary.to_csv(os.path.join(self.output_folder, "summary.csv"),
                       sep="\t", na_rep="NA", header=True, index=False)


if __name__ == "__main__":
    chembl = Chembl()
    chembl.write_folder_hierarchy()
