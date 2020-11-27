import os
import csv
import h5py
import shutil
import tempfile
import numpy as np
from scipy.stats import rankdata

from chemicalchecker.util import psql
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged, HPC


# We got these strings by doing: pg_dump -t 'scores' --schema-only mosaic
# -h aloy-dbsrv

DROP_TABLE = "DROP TABLE IF EXISTS public.molecular_info"

CREATE_TABLE = """CREATE TABLE public.molecular_info (
    inchikey text,
    formula text,
    popularity double precision,
    singularity double precision,
    mappability double precision,
    mw double precision,
    ro5 integer,
    qed double precision
);"""

CREATE_INDEX = """
CREATE INDEX scores_inchikey_idx ON public.molecular_info USING btree (inchikey);
"""

INSERT = "INSERT INTO molecular_info VALUES %s"

COUNT = "SELECT COUNT(*) FROM molecular_info"


@logged
class MolecularInfo(BaseTask):

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
        """Run the molecular info step."""
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/scores.py")

        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.DB)
            psql.query(CREATE_TABLE, self.DB)
        except Exception as e:
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)
                return

        universe_file = os.path.join(self.cachedir, "universe.h5")
        consensus_file = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "data/consensus.h5")
        with h5py.File(universe_file) as h5:
            keys = h5["keys"].asstr()[:]
        datasize = keys.shape[0]
        self.__log.info("Genretaing molecular info for " +
                        str(keys.shape[0]) + " molecules")
        keys.sort()

        job_path = tempfile.mkdtemp(
            prefix='jobs_molinfo_', dir=self.tmpdir)
        data_files_path = tempfile.mkdtemp(
            prefix='molinfo_data_', dir=self.tmpdir)

        params = {}
        params["num_jobs"] = datasize / 1000
        params["jobdir"] = job_path
        params["job_name"] = "CC_MOLINFO"
        params["elements"] = keys
        params["wait"] = True
        params["memory"] = 4
        params["cpu"] = 1
        # job command
        cc_config_path = self.config.config_path
        cc_package = os.path.join(self.config.PATH.CC_REPO, 'package')
        singularity_image = self.config.PATH.SINGULARITY_IMAGE
        command = "OMP_NUM_THREADS=1 SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, consensus_file, data_files_path, self.CC_ROOT)
        # submit jobs
        cluster = HPC.from_config(self.config)
        jobs = cluster.submitMultiJob(command, **params)

        del keys

        V = []
        iks = []
        formula = []
        for l in os.listdir(data_files_path):
            with open(os.path.join(data_files_path, l), "r") as f:
                for r in csv.reader(f, delimiter="\t"):
                    iks += [r[0]]
                    formula += [r[1]]
                    V += [[r[2], r[3], r[4], r[5], r[6], r[7]]]

        V = np.array(V).astype(np.float)

        if V.shape[0] != datasize:
            raise Exception(
                "Generated molecular info does not include all universe molecules (%d/%d)" % (V.shape[0], datasize))

        # Singularity
        V[:, 1] = rankdata(-V[:, 1]) / V.shape[0]

        # Mappability
        V[:, 2] = rankdata(V[:, 2]) / V.shape[0]
        index = range(0, datasize)
        for i in range(0, datasize, 1000):
            sl = slice(i, i + 1000)
            S = ["('%s', '%s', %.3f, %.3f, %.3f, %.3f, %.3f, %.3f)" %
                 (iks[i], formula[i], V[i, 0], V[i, 1], V[i, 2], V[i, 3], V[i, 4], V[i, 5]) for i in index[sl]]
            try:
                psql.query(INSERT % ",".join(S), self.DB)
            except Exception as e:

                print(e)

        try:
            self.__log.info("Checking table")
            count = psql.qstring(COUNT, self.DB)
            if int(count[0][0]) != datasize:
                if not self.custom_ready():
                    raise Exception(
                        "Not all universe keys were added to molecular_info (%d/%d)" % (int(count[0][0]), datasize))
                else:
                    self.__log.error(
                        "Not all universe keys were added to molecular_info (%d/%d)" % (int(count[0][0]), datasize))
            else:
                self.__log.info("Indexing table")
                psql.query(CREATE_INDEX, self.DB)
                shutil.rmtree(job_path, ignore_errors=True)
                shutil.rmtree(data_files_path, ignore_errors=True)
                self.mark_ready()
        except Exception as e:
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
