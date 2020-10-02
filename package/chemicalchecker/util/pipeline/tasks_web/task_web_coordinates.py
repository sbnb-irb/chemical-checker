import os
import json

from chemicalchecker.util import psql
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask

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
class Coordinates(BaseTask):

    def __init__(self, name=None, **params):
        args = []
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

    def run(self):
        """Run the coordinates step."""
        all_datasets = Dataset.get()
        cc = ChemicalChecker(self.CC_ROOT)
        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.DB)
            psql.query(CREATE_TABLE, self.DB)
            psql.query(DROP_TABLE_STATS, self.DB)
            psql.query(CREATE_TABLE_STATS, self.DB)
            # psql.query(CREATE_INDEX, self.config.DB)
        except Exception as e:
            self.__log.error("Error while creating coordinates tables")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)
                return

        size_exemplary = 0
        self.__log.info("Filling coordinates table")

        for ds in all_datasets:
            if not ds.exemplary:
                continue
            size_exemplary += 1
            try:
                name = str(ds.name)
                desc = str(ds.description)
                psql.query(INSERT % (str(ds.coordinate), name.replace(
                    "'", "''"), desc.replace("'", "''")), self.DB)
            except Exception as e:
                self.__log.error("Error while filling coordinates table")
                if not self.custom_ready():
                    raise Exception(e)
                else:
                    self.__log.error(e)
                    return

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
                           "xlim"][0], d["ylim"][1], d["ylim"][0]), self.DB)
            except Exception as e:
                self.__log.error("Error while filling coordinate_stats table")
                if not self.custom_ready():
                    raise Exception(e)
                else:
                    self.__log.error(e)
                    return

        try:
            self.__log.info("Checking tables")
            count = psql.qstring(COUNT, self.DB)
            if int(count[0][0]) != size_exemplary:
                if not self.custom_ready():
                    raise Exception(
                        "Not all exemplary datasets were added to coordinates (%d/%d)" % (int(count[0][0]), size_exemplary))
                else:
                    self.__log.error(
                        "Not all exemplary datasets were added to coordinates (%d/%d)" % (int(count[0][0]), size_exemplary))
            else:
                count = psql.qstring(COUNT_STATS, self.DB)
                if int(count[0][0]) != size_exemplary:
                    if not self.custom_ready():
                        raise Exception(
                            "Not all exemplary datasets were added to coordinate_stats (%d/%d)" % (int(count[0][0]), size_exemplary))
                    else:
                        self.__log.error(
                            "Not all exemplary datasets were added to coordinate_stats (%d/%d)" % (int(count[0][0]), size_exemplary))
                else:
                    self.mark_ready()
        except Exception as e:

            self.__log.error("Error while checking coordinates tables")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']

        self.run()
