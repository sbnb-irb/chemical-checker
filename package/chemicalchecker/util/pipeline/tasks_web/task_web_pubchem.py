import tempfile
import h5py
import os
import pubchempy
import requests
import json
import shutil
from airflow.models import BaseOperator
from airflow import AirflowException
from chemicalchecker.util import logged
from chemicalchecker.util import HPC
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import psql
from chemicalchecker.util import Config

# We got these strings by doing: pg_dump -t 'pubchem' --schema-only mosaic
# -h aloy-dbsrv

DROP_TABLE = "DROP TABLE IF EXISTS public.pubchem"

CREATE_TABLE = """CREATE TABLE public.pubchem (
    cid integer,
    inchikey_pubchem text,
    inchikey text,
    name text,
    synonyms text,
    pubchem_name text,
    iupac_name text,
    direct_parent text,
    unique (cid, inchikey));
   """

CREATE_INDEX = """
CREATE INDEX cid_index ON public.pubchem USING btree (cid);
CREATE INDEX inchikey_index ON public.pubchem USING gin (to_tsvector('english'::regconfig, inchikey));
CREATE INDEX inchikey_pubchem_idx ON public.pubchem USING btree (inchikey);
CREATE INDEX inchikey_pubchem_pubchem_idx ON public.pubchem USING btree (inchikey_pubchem);
CREATE INDEX name_index ON public.pubchem USING gin (to_tsvector('english'::regconfig, name));
CREATE INDEX synonyms_index ON public.pubchem USING gin (to_tsvector('english'::regconfig, synonyms));
"""

INSERT = "INSERT INTO pubchem (cid, inchikey_pubchem, inchikey, name, synonyms, pubchem_name, iupac_name, direct_parent) VALUES %s"

SELECT = "SELECT cid, inchikey_pubchem, inchikey, name, synonyms, pubchem_name, iupac_name, direct_parent FROM pubchem WHERE inchikey IN (%s)"

SELECT_CHECK = "SELECT DISTINCT (inchikey) FROM pubchem WHERE inchikey IN (%s)"

COUNT = "SELECT COUNT(DISTINCT inchikey) FROM pubchem"


@logged
class Pubchem(BaseTask, BaseOperator):

    def __init__(self, config=None, name=None, **params):

        args = []

        task_id = params.get('task_id', None)

        if task_id is None:
            params['task_id'] = name

        BaseTask.__init__(self, config, name, **params)
        BaseOperator.__init__(self, *args, **params)

        self.DB = params.get('DB', None)
        self.OLD_DB = params.get('OLD_DB', None)
        if self.DB is None:
            raise Exception('DB parameter is not set')
        if self.OLD_DB is None:
            raise Exception('OLD_DB parameter is not set')

    def run(self):
        """Run the pubchem step."""

        config_cc = Config()

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/load_pubchem.py")

        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.DB)
            psql.query(CREATE_TABLE, self.DB)
            # psql.query(CREATE_INDEX, self.DB)

        except Exception as e:

            self.__log.error(e)

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        with h5py.File(universe_file, 'r') as h5:
            data_size = h5["keys"].shape[0]

        self.__log.info("Genretaing pubchem data for " +
                        str(data_size) + " molecules")
        chunk_size = 1000
        chunks = list()
        for i in range(0, data_size, chunk_size):
            chunks.append(slice(i, i + chunk_size))

        job_path = tempfile.mkdtemp(
            prefix='jobs_pubchem_', dir=self.tmpdir)

        params = {}
        params["num_jobs"] = 4
        params["jobdir"] = job_path
        params["job_name"] = "CC_PUBCH"
        params["elements"] = chunks
        params["wait"] = True
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {}"
        command = command.format(cc_package, cc_config_path, singularity_image,
                                 script_path, universe_file, self.OLD_DB, self.DB)
        # submit jobs
        cluster = HPC.from_config(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        try:
            self.__log.info("Checking table")
            count = psql.qstring(COUNT, self.DB)
            if int(count[0][0]) != data_size:
                if not self.custom_ready():
                    raise AirflowException(
                        "Not all universe keys were added to Pubchem (%d/%d)" % (int(count[0][0]), data_size))
                else:
                    self.__log.error(
                        "Not all universe keys were added to Pubchem (%d/%d)" % (int(count[0][0]), data_size))
            else:
                self.__log.info("Indexing table")
                psql.query(CREATE_INDEX, self.DB)
                shutil.rmtree(job_path)
                self.mark_ready()
        except Exception as e:

            self.__log.error(e)

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']

        self.run()
