import os
import collections
import itertools as itt

from chemicalchecker.util import logged
from chemicalchecker.util.network import HotnetNetwork
from chemicalchecker.util import psql
from chemicalchecker.database import Dataset


@logged
class string_net():
    """Validation generator class.

    Creates a validation set
    """

    def __init__(self, net_dir, dbname, cpu=1):
        """Initialize the validation.

        Args:
            net_dir(str): The path to directory where to save network data.
            dbname(str): Name of the DB that needs to be used
        """
        self.dbname = "uniprotkb_" + dbname
        self.net_dir = net_dir
        self.cpu = cpu


# Functions

    def _read_string(self, string_file):

        self.__log.info("Reading %s..." % string_file)

        CS = 700

        string = [j for j in [i.rstrip().split() for i in open(string_file).readlines()[
            1:]] if int(j[2]) >= CS and j[0] != j[1]]

        return string

    def _select_uniprotkb(self, protein):

        R = psql.qstring(
            "SELECT source FROM uniprotkb_protein WHERE uniprot_ac = '" + protein + "'", self.dbname)

        if len(R):
            return R[0][0]
        else:
            return ''

    def _ENSP2Uniprot(self, string, map_file):

        ensp2Uniprot = collections.defaultdict(list)
        string_mapped = []
        ambiguous = []
        notmapped = []
        ints_amb = []

        self.__log.info('Reading %s...' % map_file)
        for l in open(map_file):
            l = l.split('\t')
            if len(l[20]) > 0:
                for p in l[20].split(';'):
                    ensp2Uniprot[p.replace(' ', '')].append(l[0])
        for i in string:
            p1 = i[0].replace('9606.', '')
            p2 = i[1].replace('9606.', '')
            if len(ensp2Uniprot[p1]) == 1 and len(ensp2Uniprot[p2]) == 1:
                string_mapped.append(
                    (ensp2Uniprot[p1][0], ensp2Uniprot[p2][0], {'weigth': i[2]}))
            else:
                if len(ensp2Uniprot[p1]) > 1:
                    ambiguous.append(p1)
                    ints_amb.append((p1, p2, {'weigth': i[2]}))
                elif len(ensp2Uniprot[p1]) == 0:
                    notmapped.append(p1)
                if len(ensp2Uniprot[p2]) > 1:
                    ints_amb.append((p1, p2, {'weigth': i[2]}))
                    ambiguous.append(p2)
                elif len(ensp2Uniprot[p2]) == 0:
                    notmapped.append(p2)
        self.__log.info('\tProteins not mapped: %s' %
                        len(list(set(notmapped))))
        sprot = []
        for p in list(set(ambiguous)):
            for i in ensp2Uniprot[p]:
                if self._select_uniprotkb(i) == 'sprot':
                    sprot.append(i)
        for i in ints_amb:
            if i[0] in ambiguous:
                uni1 = [p for p in ensp2Uniprot[i[0]] if p in sprot]
            else:
                uni1 = ensp2Uniprot[i[0]]
            if i[1] in ambiguous:
                uni2 = [p for p in ensp2Uniprot[i[1]] if p in sprot]
            else:
                uni2 = ensp2Uniprot[i[1]]

            if len(uni1) and len(uni2):
                for t in itt.product(uni1, uni2):
                    t += ({'weigth': i[2]},)
                    string_mapped.append(t)
        return string_mapped

    def _write_network(self, string):

        with open(self.net_dir + "/interactions.tsv", "w") as f:
            for s in string:
                if s[0] == s[1]:
                    continue
                s = sorted([s[0], s[1]])
                f.write("%s\t%s\n" % (s[0], s[1]))

    def run(self):
        """Run the network script."""

        dataset_code = 'C5.001'
        dataset = Dataset.get(dataset_code)
        map_files = {}
        for ds in dataset.datasources:
            map_files[ds.datasource_name] = ds.data_path

        string_file = os.path.join(
            map_files["string"], "9606.protein.links.txt")
        map_file = os.path.join(
            map_files["uniprot_HUMAN_9606"], "HUMAN_9606_idmapping_selected.tab")

        self.__log.info("Reading STRING")
        string = self._read_string(string_file)
        self.__log.info("Mapping to UniProtKB")
        string = self._ENSP2Uniprot(string, map_file)
        self.__log.info("Writing network")
        self._write_network(string)

        readyfile = "string_net.ready"

        HotnetNetwork.prepare(self.net_dir + "/interactions.tsv", self.net_dir, self.cpu)

        with open(os.path.join(self.net_dir, readyfile), "w") as f:
            f.write("")
