import os
import sys
import collections
from subprocess import call

from chemicalchecker.util import logged
from chemicalchecker.util import HotnetNetwork
from chemicalchecker.util import psql
from chemicalchecker.database import Dataset

DBS = set(['Reactome', 'KEGG', 'NetPath', 'PANTHER', 'WikiPathways'])


@logged
class pathwaycommons():
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

    def _execAndCheck(self, cmdStr):
        try:
            self.__log.debug(cmdStr)
            retcode = call(cmdStr, shell=True)
            self.__log.debug("FINISHED! " + cmdStr +
                             (" returned code %d" % retcode))
            if retcode != 0:
                if retcode > 0:
                    self.__log.error(
                        "ERROR return code %d, please check!" % retcode)
                elif retcode < 0:
                    self.__log.error(
                        "Command terminated by signal %d" % -retcode)
                sys.exit(1)
        except OSError as e:
            self.__log.critical("Execution failed: %s" % e)
            sys.exit(1)

    def _download_pathwaycommons(self, DATA):

            # Download Pathway Commons

        if not os.path.exists(os.path.join(self.net_dir, '1.txt')) or not os.path.exists(os.path.join(self.net_dir, '2.txt')):
                    # Part 1 contains the interactions and part 2 the gene
                    # mapping
            self.__log.info('Split file in part 1 and 2...')
            cmdStr = "sed '/^$/q' %s >%s/1.txt" % (DATA, self.net_dir)
            self._execAndCheck(cmdStr)
            cmdStr = "sed '1,/^$/d' %s >%s/2.txt" % (DATA, self.net_dir)
            self._execAndCheck(cmdStr)

    def _read_mapping(self):
        Map = collections.defaultdict(list)

        R = psql.qstring(
            "select genename,uniprot_ac From uniprotkb_protein where taxid = '9606' and genename != '' and complete = 'Complete proteome'", self.dbname)

        for l in R:
            Map[l[0]].append(l[1])
        return Map

    def _read_pathwaycommons(self, Map):
        notmap = []
        self.__log.info('Reading Pathway Commons...')
        self.__log.info('...from Part 1')
        pathwaycommon = []
        i = 0
        lines = open(os.path.join(self.net_dir, '1.txt')).readlines()[1:]
        length = len(lines)
        # print length
        for l in lines:
            l = l.split('\t')
            # print l
            i += 1
            if i % 1000 == 0:
                self.__log.info(
                    '...' + str(round(float(i) / length * 100, 2)) + '%')
            if len(l) > 2:
                dbs = l[3].split(';')
                # print dbs
                if len(set(dbs) & DBS) > 0:
                    if l[0] in Map and l[2] in Map:
                        pathwaycommon.append(
                            (Map[l[0]][0], Map[l[2]][0], {'database': l[3]}))
                    else:
                        if l[0] not in Map and l[0].find('CHEBI') < 0:
                            notmap.append(l[0])
                        if l[2] not in Map and l[2].find('CHEBI') < 0:
                            notmap.append(l[2])

        return pathwaycommon

    def _write_network(self, pathwaycommon):

        ppis = set()
        for k in pathwaycommon:
            ppis.update([tuple(sorted([k[0], k[1]]))])
        with open(self.net_dir + "/interactions.tsv", "w") as f:
            for ppi in ppis:
                f.write("%s\t%s\n" % (ppi[0], ppi[1]))

    def run(self):
        """Run the network script."""

        dataset_code = 'C5.001'
        dataset = Dataset.get(dataset_code)
        map_files = {}
        for ds in dataset.datasources:
            map_files[ds.name] = ds.data_path

        DATA = os.path.join(
            map_files["pathwaycommons_hgnc"], "PathwayCommons9.All.hgnc.txt")

        Map = self._read_mapping()
        self._download_pathwaycommons(DATA)
        pathwaycommon = self._read_pathwaycommons(Map)
        self._write_network(pathwaycommon)

        readyfile = "pathwaycommons.ready"

        HotnetNetwork.prepare(self.net_dir + "/interactions.tsv", self.net_dir, self.cpu)

        with open(os.path.join(self.net_dir, readyfile), "w") as f:
            f.write("")
