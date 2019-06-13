import h5py
import os
import numpy as np
from chemicalchecker.util import logged
from chemicalchecker.database import Molrepo
from chemicalchecker.util import BaseStep
from chemicalchecker.util import psql


@logged
class DB_universe(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the initial step."""

        molrepos = Molrepo.get_universe_molrepos()

        self.__log.info("Querying molrepos")

        inchikeys = set()

        for molrepo in molrepos:

            molrepo = str(molrepo[0])

            inchikeys.update(Molrepo.get_fields_by_molrepo_name(
                molrepo, ["inchikey"]))

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        with h5py.File(universe_file, "w") as h5:
            h5.create_dataset("keys", data=np.array(
                [str(i[0]) for i in inchikeys]))

        con = psql.get_connection(self.config.OLD_DB)

        con.autocommit = True

        cur = con.cursor()
        try:
            cur.execute('CREATE DATABASE {};'.format(self.config.DB))
            self.mark_ready()
        except Exception, e:
            self.__log.error(e)
        finally:
            con.close()
