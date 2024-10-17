import os
import h5py

from chemicalchecker.util import psql
from chemicalchecker.database import Dataset
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged
# We got these strings by doing: pg_dump -t 'pubchem' --schema-only mosaic
# -h aloy-dbsrv


DROP_TABLE = "DROP TABLE IF EXISTS public.projections"

INSERT = "INSERT INTO projections VALUES %s"

COUNT = "SELECT COUNT(DISTINCT inchikey) FROM projections"


@logged
class Projections(BaseTask):

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

    def run(self):
        """Run the projections step."""
        all_datasets = Dataset.get()
        cc = ChemicalChecker(self.CC_ROOT)
        map_coord_code = {}
        db_name = self.DB
        for ds in all_datasets:
            if not ds.exemplary:
                continue
            map_coord_code[ds.coordinate] = ds.dataset_code
        spaces = sorted(map_coord_code.keys())

        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.DB)
            S = "CREATE TABLE projections (inchikey TEXT, "
            for coord in spaces:
                S += "%s_idx INTEGER, " % coord
                S += "%s_x FLOAT, " % coord
                S += "%s_y FLOAT, " % coord
            S += "PRIMARY KEY (inchikey) );\n"
            psql.query(S, db_name)
        except Exception as e:
            self.__log.error("Error while creating ptojections table")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)
                return

        NULL = ["NULL"] * 25 * 3

        ijk = {}
        c = 0
        for coord in spaces:
            ijk[coord] = (c, c + 1, c + 2)
            c += 3

        D = {}
        for coord in spaces:
            wh = ijk[coord]
            proj2_full = cc.get_signature(
                'proj2', 'full', map_coord_code[coord])
            if not os.path.exists(proj2_full.data_path):
                raise Exception("Projection point for dataset " +
                                map_coord_code[coord] + " is not available.")
            with h5py.File(proj2_full.data_path, "r") as hf:
                inchikeys = hf["keys"][:]
                V = hf["V"][:]
            for w in range(len(inchikeys)):
                ik = inchikeys[w].decode('utf-8')
                if ik not in D:
                    D[ik] = NULL[:]
                D[ik][wh[0]] = w
                D[ik][wh[1]] = V[w, 0]
                D[ik][wh[2]] = V[w, 1]

        self.__log.info("Filling projections table")
        d_keys = list(D.keys())
        
        for c in self.__chunker(d_keys, 10000):
            s = []
            for x in c:
                s += ["(" + ",".join(["'%s'" % x ] + ["%s" % y for y in D[x]]) + ")"]
            s = ",".join(s)
            psql.query(INSERT % s, db_name)

        try:
            self.__log.info("Creating indexes for table")
            S = ''
            for coord in spaces:
                coord = coord.lower()
                S += "CREATE UNIQUE INDEX %s_idx_projections_idx ON projections (%s_idx);\n" % (
                    coord, coord)
            psql.query(S, db_name)
            count = psql.qstring(COUNT, self.DB)
            if int(count[0][0]) != len(D):
                if not self.custom_ready():
                    raise Exception(
                        "Not all inchikeys were added to projections (%d/%d)" % (int(count[0][0]), len(D)))
                else:
                    self.__log.error(
                        "Not all inchikeys were added to projections (%d/%d)" % (int(count[0][0]), len(D)))
            else:
                self.mark_ready()
        except Exception as e:
            self.__log.error(
                "Error while checking & creating indexes in projections table")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)

    def __chunker(self, data, size=2000):
        for i in range(0, len(data), size):
            yield data[slice(i, i + size)]

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
