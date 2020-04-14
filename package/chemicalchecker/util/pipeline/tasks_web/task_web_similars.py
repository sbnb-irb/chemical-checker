import tempfile
import os
import shutil
import h5py
import json
from tqdm import tqdm
from chemicalchecker.util import logged
from chemicalchecker.util import HPC
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import psql
from chemicalchecker.util import Config
from airflow.models import BaseOperator
from airflow import AirflowException

# We got these strings by doing: pg_dump -t 'scores' --schema-only mosaic
# -h aloy-dbsrv


@logged
class Similars(BaseTask, BaseOperator):

    def __init__(self, name=None, **params):

        args = []

        task_id = params.get('task_id', None)

        if task_id is None:
            params['task_id'] = name

        BaseTask.__init__(self, name, **params)
        BaseOperator.__init__(self, *args, **params)

        self.DB = params.get('DB', None)
        if self.DB is None:
            raise Exception('DB parameter is not set')
        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')
        self.MOLECULES_PATH = params.get('MOLECULES_PATH', None)
        if self.MOLECULES_PATH is None:
            raise Exception('MOLECULES_PATH parameter is not set')

    def run(self):
        """Run the molecular info step."""

        config_cc = Config()

        db_name = self.DB

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/similars.py")

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        names_map = {}

        with h5py.File(universe_file, 'r') as hf:
            universe_keys = hf["keys"][:]

        ik_names_file = os.path.join(self.tmpdir, "inchies_names.json")

        if not os.path.exists(ik_names_file):
            for input_data in self.__chunker(universe_keys):

                data = psql.qstring("select inchikey_pubchem as inchikey,name from pubchem INNER JOIN( VALUES " +
                                    ', '.join('(\'{0}\')'.format(w) for w in input_data) + ") vals(v) ON (inchikey_pubchem = v)", db_name)

                for i in range(0, len(data)):

                    inchi = data[i][0]
                    name = data[i][1]
                    if name is None:
                        name = inchi

                    names_map[inchi] = name

            if len(names_map) > 0:
                with open(ik_names_file, 'w') as outfile:
                    json.dump(names_map, outfile)
            else:
                if not self.custom_ready():
                    raise AirflowException(
                        "Inchikeys name JSON file was not created")
                else:
                    self.__log.error(
                        "Inchikeys name JSON file was not created")
                    return

        self.__log.info("Launching jobs to create json files for " +
                        str(len(universe_keys)) + " molecules")

        job_path = tempfile.mkdtemp(
            prefix='jobs_similars_', dir=self.tmpdir)

        version = self.DB.replace("cc_web_", '')
        mol_path = self.MOLECULES_PATH

        params = {}
        params["num_jobs"] = len(universe_keys) / 200
        params["jobdir"] = job_path
        params["job_name"] = "CC_JSONSIM"
        params["elements"] = universe_keys
        params["memory"] = 6
        params["wait"] = True
        # job command
        singularity_image = config_cc.PATH.SINGULARITY_IMAGE
        command = "OMP_NUM_THREADS=3 SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {} {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, ik_names_file, mol_path, self.DB, version, self.CC_ROOT)
        # submit jobs
        cluster = HPC.from_config(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        self.__log.info("Checking results")
        missing_keys = list()
        for i in tqdm(range(len(universe_keys))):
            inchikey = universe_keys[i]
            PATH = mol_path + "/%s/%s/%s/%s" % (
                inchikey[:2], inchikey[2:4], inchikey, 'explore_' + version + '.json')
            if not os.path.exists(PATH):
                missing_keys.append(inchikey)

        if len(missing_keys) != 0:
            if not self.custom_ready():
                raise AirflowException(
                    "Not all molecules have their json explore file (%d/%d)" % (len(missing_keys), len(universe_keys)))
            else:
                self.__log.error(
                    "Not all molecules have their json explore file (%d/%d)" % (len(missing_keys), len(universe_keys)))
        else:
            shutil.rmtree(job_path)
            self.mark_ready()

    def __chunker(self, data, size=2000):

        for i in range(0, len(data), size):
            yield data[slice(i, i + size)]

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']

        self.run()
