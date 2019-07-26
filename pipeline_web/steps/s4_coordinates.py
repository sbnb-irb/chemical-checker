import os
import json
from chemicalchecker.util import logged
from chemicalchecker.util import BaseStep
from chemicalchecker.util import psql
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.database import Dataset

# We got these strings by doing: pg_dump -t 'pubchem' --schema-only mosaic
# -h aloy-dbsrv


DROP_TABLE = "DROP TABLE IF EXISTS public.coordinates"

DROP_TABLE_STATS = "DROP TABLE IF EXISTS public.coordinate_stats"

CREATE_TABLE = """CREATE TABLE public.coordinates (
    coord character varying(4) PRIMARY KEY,
    name text,
    description text
);
"""

CREATE_TABLE_STATS = """CREATE TABLE coordinate_stats (
    coord VARCHAR(2),
    mols INTEGER,
    xlim_l FLOAT,
    xlim_u FLOAT,
    ylim_l FLOAT,
    ylim_u FLOAT,
    PRIMARY KEY (coord)
);
"""

INSERT = "INSERT INTO coordinates (coord, name, description) VALUES ('%s', '%s', '%s')"

INSERT_STATS = "INSERT INTO coordinate_stats (coord, mols, xlim_u, xlim_l, ylim_u, ylim_l) VALUES ('%s', %d, %f, %f, %f, %f)"

COUNT_STATS = "SELECT COUNT(DISTINCT coord) FROM coordinate_stats"

COUNT = "SELECT COUNT(DISTINCT coord) FROM coordinates"


@logged
class Coordinates(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the coordinates step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.config.DB)
            psql.query(CREATE_TABLE, self.config.DB)
            psql.query(DROP_TABLE_STATS, self.config.DB)
            psql.query(CREATE_TABLE_STATS, self.config.DB)
            # psql.query(CREATE_INDEX, self.config.DB)

        except Exception, e:

            self.__log.error(e)
            raise Exception("Error while creating coordinates table")

        size_exemplary = 0
        self.__log.info("Filling coordinates table")

        for ds in all_datasets:
            if not ds.exemplary:
                continue
            size_exemplary += 1
            try:
                name = str(ds.name)
                desc = str(ds.description)
                psql.query(INSERT % (str(ds.coordinate), name.replace("'", "''"), desc.replace("'", "''")), self.config.DB)

            except Exception, e:

                self.__log.error(e)
                raise Exception("Error while filling coordinates table")

        self.__log.info("Filling coordinate_stats table")
        for ds in all_datasets:
            if not ds.exemplary:
                continue

            proj2 = cc.get_signature('proj2', 'reference', ds.dataset_code)
            proj2_full = cc.get_signature('proj2', 'full', ds.dataset_code)
            size = proj2_full.shape[0]
            try:
                coord = str(ds.coordinate)
                d = json.loads(
                    open(os.path.join(proj2.stats_path, "proj_stats.json")).read())
                psql.query(INSERT_STATS % (coord, size, d["xlim"][1], d[
                           "xlim"][0], d["ylim"][1], d["ylim"][0]), self.config.DB)

            except Exception, e:

                self.__log.error(e)
                raise Exception("Error while filling coordinate_stats table")

        try:
            self.__log.info("Checking tables")
            count = psql.qstring(COUNT, self.config.DB)
            if int(count[0][0]) != size_exemplary:
                self.__log.error(
                    "Not all exemplary datasets were added to coordinates (%d/%d)" % (int(count[0][0]), size_exemplary))
            else:
                count = psql.qstring(COUNT_STATS, self.config.DB)
                if int(count[0][0]) != size_exemplary:
                    self.__log.error(
                        "Not all exemplary datasets were added to coordinate_stats (%d/%d)" % (int(count[0][0]), size_exemplary))
                else:
                    self.mark_ready()
        except Exception, e:

            self.__log.error(e)
            raise Exception("Error while checking coordinates tables")
