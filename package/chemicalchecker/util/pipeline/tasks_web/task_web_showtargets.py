import os
import collections

from chemicalchecker.util import psql
from chemicalchecker.database import Dataset
from chemicalchecker.database import UniprotKB
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged

# We got these strings by doing: pg_dump -t 'pubchem' --schema-only mosaic
# -h aloy-dbsrv

DROP_TABLE = "DROP TABLE IF EXISTS public.showtargets"
DROP_TABLE_DESC = "DROP TABLE IF EXISTS public.showtargets_description"

CREATE_TABLE_DESC = """CREATE TABLE public.showtargets_description (
    uniprot_ac text PRIMARY KEY,
    genename text,
    fullname text,
    taxid text,
    organism text
);"""

CREATE_TABLE = """CREATE TABLE public.showtargets (
    inchikey text,
    uniprot_ac text,
    rank integer,
    display text
);"""

CREATE_INDEX_DESC = """
CREATE INDEX genename_showtargets_description_idx ON public.showtargets_description USING btree (genename);
CREATE INDEX taxid_showtargets_description_idx ON public.showtargets_description USING btree (taxid);
"""

CREATE_INDEX = """
CREATE INDEX inchikey_showtargets_idx ON showtargets(inchikey);
CREATE INDEX uniprot_ac_showtargets_idx ON showtargets(uniprot_ac);
CREATE INDEX rank_showtargets_idx ON showtargets(rank);
"""

INSERT = "INSERT INTO showtargets VALUES %s"

ref_spaces = ['B1.001', 'B2.001', 'B4.001', 'B5.001']


@logged
class ShowTargets(BaseTask):

    def __init__(self, name=None, **params):
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name
        BaseTask.__init__(self, name, **params)

        self.DB = params.get('DB', None)
        if self.DB is None:
            raise Exception('DB parameter is not set')
        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')
        self.uniprot_db_version = params.get('uniprot_db_version', None)
        if self.uniprot_db_version is None:
            raise Exception('uniprot_db_version parameter is not set')

    def run(self):
        """Run the show targets step."""
        database_name = self.DB
        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, database_name)
            psql.query(CREATE_TABLE, database_name)
            psql.query(DROP_TABLE_DESC, database_name)
            psql.query(CREATE_TABLE_DESC, database_name)
            # psql.query(CREATE_INDEX, database_name)
        except Exception as e:
            self.__log.error("Error while creating tables")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)
                return

        cc = ChemicalChecker(self.CC_ROOT)
        prots = set()
        for space in ref_spaces:
            s0 = cc.get_signature('sign0', 'full', space)
            features = s0.features
            prots.update([x.split('(')[0] for x in features if "Class:" not in x])
        prots = sorted(prots)

        self.__log.info("Querying UniprotKB...")
        ukb = UniprotKB(self.uniprot_db_version)
        showtarg_d = ukb.get_proteins(
            prots,
            limit_to_fields=["genename", "fullname", "taxid", "organism"])

        self.__log.info("Inserting proteins into database...")
        R = []
        for p in prots:
            if p in showtarg_d:
                d = showtarg_d[p]
                R += [(self.__pstr(p),
                       self.__pstr(d["genename"]),
                       self.__pstr(d["fullname"]),
                       self.__pstr(d["taxid"]),
                       self.__pstr(d["organism"]))]
            else:
                R += [("'%s'" % p, 'NULL', 'NULL', 'NULL', 'NULL')]
        R = ["(%s)" % ",".join(r) for r in R]

        try:
            for c in self.__chunker(R, 1000):
                psql.query("INSERT INTO showtargets_description VALUES %s" %
                           ",".join(c), database_name)
            psql.query(CREATE_INDEX_DESC, database_name)
            showtarg_d = {}
            for r in psql.qstring(
                    "SELECT uniprot_ac, genename, taxid FROM "
                    "showtargets_description", database_name):
                showtarg_d[r[0]] = [r[1], r[2]]
        except Exception as e:
            self.__log.error("Error while filling showtargets_description")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)
                return

        self.__log.info("Getting orthologs from MetaPhors")
        dataset = Dataset.get('C3.001')
        map_files = {}
        for ds in dataset.datasources:
            map_files[ds.datasource_name] = ds.data_path

        id_conversion = os.path.join(
            map_files["metaphors_id_conversion"], "id_conversion.txt")
        file_9606 = os.path.join(map_files["metaphors_9606"], "9606.txt")
        human_proteome = os.path.join(
            map_files["human_proteome"], "human_proteome.tab")

        metaphorsid_uniprot = collections.defaultdict(set)
        f = open(id_conversion, "r")
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[1] == "SwissProt" or l[1] == "TrEMBL":
                metaphorsid_uniprot[l[2]].update([l[0]])
        f.close()

        any_human = collections.defaultdict(set)
        f = open(file_9606, "r")
        f.readline()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[3] not in metaphorsid_uniprot:
                continue
            if l[1] not in metaphorsid_uniprot:
                continue
            for po in metaphorsid_uniprot[l[3]]:
                for ph in metaphorsid_uniprot[l[1]]:
                    any_human[po].update([ph])
                    any_human[ph].update([ph])
        f.close()

        f = open(human_proteome, "r")
        f.readline()
        for l in f:
            p = l.split("\t")[0]
            any_human[p].update([p])
        f.close()

        self.__log.info(
            "First is always MOA, in any species... "
            "(only sorted alphabetically by gene name)")
        seens = collections.defaultdict(set)
        showtargs = collections.defaultdict(list)
        s0 = cc.get_signature('sign0', 'full', ref_spaces[0])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features = features[sig > 0]
            key = keys[i]
            i += 1
            if len(collected_features) == 0:
                continue
            prots = [x.split('(')[0] for x in collected_features if "Class:" not in x]

            prots = self.__sort_alphabet(prots, showtarg_d)
            showtargs[key] += prots
            seens[key].update(prots)

        self.__log.info("Now it is human...")
        hp = set([r[0] for r in psql.qstring(
            "SELECT uniprot_ac FROM showtargets_description "
            "WHERE taxid = '9606'", database_name)])

        self.__log.info("...binding table... (sorted by potency)")
        s0 = cc.get_signature('sign0', 'full', ref_spaces[2])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features_1 = features[sig == 1]
            collected_features_2 = features[sig == 2]
            if len(collected_features_1) == 0 and len(collected_features_2) == 0:
                i += 1
                continue
            key = keys[i]
            i += 1
            if key in seens:
                s = seens[key]
            else:
                s = set()
            prots0 = hp.intersection(
                [x.split('(')[0] for x in collected_features_2 if "Class:" not in x]).difference(s)
            seens[key].update(prots0)
            prots0 = self.__sort_alphabet(list(prots0), showtarg_d)
            showtargs[key] += prots0
            prots1 = hp.intersection(
                [x.split('(')[0] for x in collected_features_1 if "Class:" not in x]).difference(s)
            seens[key].update(prots1)
            prots1 = self.__sort_alphabet(list(prots1), showtarg_d)
            showtargs[key] += prots1

        self.__log.info("...metabolic genes table...")
        s0 = cc.get_signature('sign0', 'full', ref_spaces[1])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features = features[sig > 0]
            if len(collected_features) == 0:
                i += 1
                continue
            key = keys[i]
            i += 1
            if key in seens:
                s = seens[key]
            else:
                s = set()
            prots = set(
                [x.split('(')[0] for x in collected_features if "Class:" not in x]).difference(s)
            seens[key].update(prots)
            prots = self.__sort_alphabet(prots, showtarg_d)
            showtargs[key] += prots

        self.__log.info("...HTS bioassays table...")
        s0 = cc.get_signature('sign0', 'full', ref_spaces[3])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features = features[sig > 0]
            if len(collected_features) == 0:
                i += 1
                continue
            key = keys[i]
            i += 1
            if key in seens:
                s = seens[key]
            else:
                s = set()
            prots = hp.intersection(
                [x.split('(')[0] for x in collected_features if "Class:" not in x]).difference(s)
            seens[key].update(prots)
            prots = self.__sort_alphabet(prots, showtarg_d)
            showtargs[key] += prots

        self.__log.info(
            "And then the rest of species "
            "(where no human orthologs are already known)")

        self.__log.info("...binding table... (sorted by potency)")
        s0 = cc.get_signature('sign0', 'full', ref_spaces[2])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features_1 = features[sig == 1]
            collected_features_2 = features[sig == 2]
            key = keys[i]
            i += 1
            if len(collected_features_1) == 0 and len(collected_features_2) == 0:
                continue
            if key in seens:
                s = seens[key]
            else:
                s = set()
            prots0 = set(
                [x.split('(')[0] for x in collected_features_2 if "Class:" not in x]).difference(s)
            ho = set()
            for p in prots0:
                if p in any_human:
                    ho.update(any_human[p])
            prots0 = prots0.difference(ho)
            seens[key].update(prots0)
            prots0 = self.__sort_alphabet(list(prots0), showtarg_d)
            showtargs[key] += prots0
            prots1 = set(
                [x.split('(')[0] for x in collected_features_1 if "Class:" not in x]).difference(s)
            ho = set()
            for p in prots1:
                if p in any_human:
                    ho.update(any_human[p])
            prots1 = prots1.difference(ho)
            seens[key].update(prots1)
            prots1 = self.__sort_alphabet(list(prots1), showtarg_d)
            showtargs[key] += prots1

        self.__log.info("...HTS bioassays table...")
        s0 = cc.get_signature('sign0', 'full', ref_spaces[3])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features = features[sig > 0]
            key = keys[i]
            i += 1
            if len(collected_features) == 0:
                continue

            if key in seens:
                s = seens[key]
            else:
                s = set()
            prots = set(
                [x.split('(')[0] for x in collected_features if "Class:" not in x]).difference(s)
            ho = set()
            for p in prots:
                if p in any_human:
                    ho.update(any_human[p])
            prots = prots.difference(ho)
            seens[key].update(prots)
            prots = self.__sort_alphabet(prots, showtarg_d)
            showtargs[key] += prots

        self.__log.info("And finally the rest of species...")

        self.__log.info("...binding table... (sorted by potency)")
        s0 = cc.get_signature('sign0', 'full', ref_spaces[2])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features_1 = features[sig == 1]
            collected_features_2 = features[sig == 2]
            key = keys[i]
            i += 1
            if len(collected_features_1) == 0 and len(collected_features_2) == 0:
                continue
            if key in seens:
                s = seens[key]
            else:
                s = set()
            prots0 = set(
                [x.split('(')[0] for x in collected_features_2 if "Class:" not in x]).difference(s)
            seens[key].update(prots0)
            prots0 = self.__sort_alphabet(list(prots0), showtarg_d)
            showtargs[key] += prots0
            prots1 = set(
                [x.split('(')[0] for x in collected_features_1 if "Class:" not in x]).difference(s)
            seens[key].update(prots1)
            prots1 = self.__sort_alphabet(list(prots1), showtarg_d)
            showtargs[key] += prots1

        self.__log.info("...HTS bioassays table...")
        s0 = cc.get_signature('sign0', 'full', ref_spaces[3])
        features = s0.features
        keys = s0.keys
        i = 0
        for sig in s0:
            collected_features = features[sig > 0]
            key = keys[i]
            i += 1
            if len(collected_features) == 0:
                continue
            if key in seens:
                s = seens[key]
            else:
                s = set()
            prots = set(
                [x.split('(')[0] for x in collected_features if "Class:" not in x]).difference(s)
            seens[key].update(prots)
            prots = self.__sort_alphabet(prots, showtarg_d)
            showtargs[key] += prots

        displays = {}
        for k, v in showtarg_d.items():
            if v[0] is None:
                displays[k] = k
            else:
                displays[k] = v[0]

        R = []
        for k, v in showtargs.items():
            already_disp = set()
            rank = 1
            for p in v:
                disp = displays[p]
                if disp.lower() in already_disp:
                    R += [("'%s'" % k, "'%s'" % p, 'NULL', 'NULL')]
                else:
                    R += [("'%s'" % k, "'%s'" % p, '%d' % rank, "'%s'" % disp)]
                    rank += 1
                    already_disp.update([disp.lower()])
        R = ["(%s)" % ",".join(r) for r in R]

        try:
            self.__log.info("Inserting into database...")

            for c in self.__chunker(R, 10000):
                psql.query(INSERT % ",".join(c), database_name)

            self.__log.info("Indexing table")
            psql.query(CREATE_INDEX, database_name)
            self.mark_ready()
        except Exception as e:

            self.__log.error("Error while saving tables")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)
                return

    def __pstr(self, s):
        if s == "":
            return 'NULL'
        else:
            return "'%s'" % s.replace("'", "")

    def __sort_alphabet(self, prots, showtarg_d):
        def nonesorter(a):
            if not a:
                return ""
            return a
        P0 = []
        P1 = []
        for p in prots:
            if p not in showtarg_d:
                P1 += [(p, p)]
            else:
                P0 += [(p, showtarg_d[p][0])]
        return [r[0] for r in sorted(P0, key=lambda tup: nonesorter(tup[1]))] + [r[0] for r in sorted(P1, key=lambda tup: nonesorter(tup[1]))]

    def __chunker(self, data, size=2000):
        for i in range(0, len(data), size):
            yield data[slice(i, i + size)]

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
