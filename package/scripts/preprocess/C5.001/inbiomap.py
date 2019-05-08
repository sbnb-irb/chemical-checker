import os
import csv
import glob
import numpy as np

from chemicalchecker.util import logged
from chemicalchecker.util.network import HotnetNetwork
from chemicalchecker.database import Dataset


@logged
class inbiomap():
    """Validation generator class.

    Creates a validation set
    """

    def __init__(self, net_dir, dbname, cpu=1):
        """Initialize the validation.

        Args:
            net_dir(str): The path to directory where to save network data.
            dbname(str): Name of the DB that needs to be used
        """
        self.net_dir = net_dir
        self.cpu = cpu


# Functions

    def _read_data(self, inweb):
        THRESHOLD = 0.2

        with open(inweb, "r") as f:
            E = set()
            for r in csv.reader(f, delimiter="\t"):
                if "uniprotkb:" not in r[0] or "uniprotkb:" not in r[1]:
                    continue
                if "taxid:9606" not in r[9] or "taxid:9606" not in r[10]:
                    continue
                score = np.mean([float(x)
                                 for x in r[-2].split("|") if x != "-"])
                if score < THRESHOLD:
                    continue
                ps0 = r[0].split("uniprotkb:")[1].split("|")
                ps1 = r[1].split("uniprotkb:")[1].split("|")
                for p0 in ps0:
                    for p1 in ps1:
                        E.update([tuple(sorted([p0, p1]))])
        return E

    def _write_network(self, E):

        with open(self.net_dir + "/interactions.tsv", "w") as f:
            for e in E:
                f.write("%s\t%s\n" % (e[0], e[1]))

    def run(self):
        """Run the network script."""

        dataset_code = 'C5.001'
        dataset = Dataset.get(dataset_code)
        map_files = {}
        for ds in dataset.datasources:
            map_files[ds.datasource_name] = ds.data_path

        inweb = glob.glob(map_files["inBio_Map_core"] +
                          '/InBio_Map_core_*/core.psimitab')[0]

        self.__log.info("Reading Inweb")
        E = self._read_data(inweb)
        self.__log.info("Writing network")
        self._write_network(E)

        readyfile = "inbiomap.ready"

        HotnetNetwork.prepare(self.net_dir + "/interactions.tsv", self.net_dir, self.cpu)

        with open(os.path.join(self.net_dir, readyfile), "w") as f:
            f.write("")
