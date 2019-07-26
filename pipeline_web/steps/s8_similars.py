import tempfile
import os
import shutil
import h5py
import json
from chemicalchecker.util import logged
from chemicalchecker.util import HPC
from chemicalchecker.util import BaseStep
from chemicalchecker.util import psql
from chemicalchecker.util import Config

# We got these strings by doing: pg_dump -t 'scores' --schema-only mosaic
# -h aloy-dbsrv


@logged
class Similars(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molecular info step."""

        config_cc = Config()

        db_name = self.config.DB

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(
            config_cc.PATH.CC_REPO, "pipeline_web", "steps", "scripts", "libraries.py")

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        names_map = {}

        with h5py.File(universe_file, 'r') as hf:
            universe_keys = hf["keys"][:]

        for input_data in self.__chunker(universe_keys):

            data = psql.qstring("select inchikey_pubchem as inchikey,name from pubchem INNER JOIN( VALUES " +
                                ', '.join('(\'{0}\')'.format(w) for w in input_data) + ") vals(v) ON (inchikey_pubchem = v)", db_name)

            for i in range(0, len(data)):

                inchi = data[i][0]
                name = data[i][1]
                if name is None:
                    name = inchi

                names_map[inchi] = name

        ik_names_file = os.path.join(self.tmpdir, "inchies_names.json")
        with open(ik_names_file, 'w') as outfile:
            json.dump(names_map, outfile)

        self.__log.info("Launching jobs to create json files for " +
                        str(len(universe_keys)) + " molecules")

        job_path = tempfile.mkdtemp(
            prefix='jobs_libraries_', dir=self.tmpdir)

        libraries_path = os.path.join(self.tmpdir, "libraries_files")

        if not os.path.exists(libraries_path):
            original_umask = os.umask(0)
            os.makedirs(libraries_path, 0o775)
            os.umask(original_umask)

        version = self.config.DB.replace("cc_web_", '')
        mol_path = self.config.MOLECULES_PATH

        params = {}
        params["num_jobs"] = len(universe_keys) / 500
        params["jobdir"] = job_path
        params["job_name"] = "CC_JSONSIM"
        params["elements"] = universe_keys
        params["wait"] = True
        # job command
        singularity_image = config_cc.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, ik_names_file, mol_path, self.config.DB, version)
        # submit jobs
        cluster = HPC(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        self.__log.info("Checking results")
        missing_keys = list()
        for inchikey in universe_keys:
            PATH = mol_path + "/%s/%s/%s/%s" % (
                inchikey[:2], inchikey[2:4], inchikey, 'explore_' + version + '.json')
            if not os.path.exists(PATH):
                missing_keys.append(inchikey)

        if len(missing_keys) != 0:
            self.__log.error(
                "Not all molecules have their json explore file (%d/%d)" % (len(missing_keys), len(universe_keys)))
        else:
            shutil.rmtree(job_path)
            self.mark_ready()

    def __chunker(self, data, size=2000):

        for i in range(0, len(data), size):
            yield data[slice(i, i + size)]
