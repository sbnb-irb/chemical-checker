import os

from chemicalchecker.util import logged
from chemicalchecker.util.network import HotnetNetwork
from chemicalchecker.util import psql


@logged
class ppidb():
    """Validation generator class.

    Creates a validation set
    """

    def __init__(self, net_dir, dbname, cpu=1):
        """Initialize the validation.

        Args:
            net_dir(str): The path to directory where to save network data.
            dbname(str): Name of the DB that needs to be used
        """
        self.dbname = "ppidb_" + dbname
        self.net_dir = net_dir
        self.cpu = cpu

# Functions

    def _fetch_data(self):

        PPIdbQuery = """
                        SELECT uniref_canonical1, uniref_canonical2
                        FROM PPIDB_INTERACTIONS
                        WHERE uniprot_taxid1 = '9606' AND
                        uniprot_taxid2 = '9606' AND
                        active_uniprot_proteins AND
                        NOT ambiguous_mapping AND
                        NOT duplicated_in_author_inferences AND
                        uniref_canonical1 != uniref_canonical2
                        AND (method_binary OR curation_binary);"""

        R = set()
        for r in psql.qstring(PPIdbQuery, self.dbname):
            if r[0] == r[1]:
                continue
            R.update([tuple(sorted([r[0], r[1]]))])

        return R

    def _write_network(self, ppi):

        with open(self.net_dir + "/interactions.tsv", "w") as f:
            for s in ppi:
                f.write("%s\t%s\n" % (s[0], s[1]))

    def run(self):
        """Run the network script."""

        ppi = self._fetch_data()
        self.__log.info("Writing network")
        self._write_network(ppi)

        readyfile = "ppidb.ready"

        HotnetNetwork.prepare(self.net_dir + "/interactions.tsv", self.net_dir, self.cpu)

        with open(os.path.join(self.net_dir, readyfile), "w") as f:
            f.write("")
