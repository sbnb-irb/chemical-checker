import tempfile
import h5py
import os
import csv
import numpy as np
from scipy.stats import rankdata
import shutil
from chemicalchecker.util import logged
from chemicalchecker.util import HPC
from chemicalchecker.util import BaseStep
from chemicalchecker.util import psql
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import Config
from chemicalchecker.database import Dataset

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
class MolecularInfo(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molecular info step."""

        config_cc = Config()

        all_datasets = Dataset.get()

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(
            config_cc.PATH.CC_REPO, "pipeline_web", "steps", "scripts", "scores.py")

        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.config.DB)
            psql.query(CREATE_TABLE, self.config.DB)

        except Exception, e:

            self.__log.error(e)

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        consensus_file = os.path.join(self.tmpdir, "consensus.h5")

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        matrix = list()
        coords = list()
        for ds in all_datasets:
            if not ds.exemplary:
                continue
            s3 = cc.get_signature('sign3', 'full', ds.dataset_code)
            coords.append(str(ds.coordinate))
            matrix.append(s3.get_h5_dataset('datasets_correlation'))
        matrix = np.vstack(matrix)

        with h5py.File(consensus_file, "w") as h5:
            h5.create_dataset("correlations", data=np.array(matrix))
            h5.create_dataset("coords", data=np.array(coords))

        with h5py.File(universe_file) as h5:
            keys = h5["keys"][:]

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
        singularity_image = config_cc.PATH.SINGULARITY_IMAGE
        command = "OMP_NUM_THREADS=1 SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, consensus_file, data_files_path)
        # submit jobs
        cluster = HPC(config_cc)
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

                psql.query(INSERT % ",".join(S), self.config.DB)
            except Exception, e:

                print(e)

        try:
            self.__log.info("Checking table")
            count = psql.qstring(COUNT, self.config.DB)
            if int(count[0][0]) != datasize:
                self.__log.error(
                    "Not all universe keys were added to molecular_info (%d/%d)" % (int(count[0][0]), datasize))
            else:
                self.__log.info("Indexing table")
                psql.query(CREATE_INDEX, self.config.DB)
                shutil.rmtree(job_path)
                shutil.rmtree(data_files_path)
                self.mark_ready()
        except Exception, e:

            self.__log.error(e)
