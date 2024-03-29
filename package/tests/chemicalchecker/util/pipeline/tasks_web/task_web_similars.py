import os
import h5py
import json
import shutil
import tempfile
from tqdm import tqdm

from chemicalchecker.util import psql
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged, HPC


# We got these strings by doing: pg_dump -t 'scores' --schema-only mosaic
# -h aloy-dbsrv


@logged
class Similars(BaseTask):

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
        self.MOLECULES_PATH = params.get('MOLECULES_PATH', None)
        if self.MOLECULES_PATH is None:
            raise Exception('MOLECULES_PATH parameter is not set')

    def run(self):
        """Run the molecular info step."""
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/similars.py")
        universe_file = os.path.join(self.cachedir, "universe.h5")

        with h5py.File(universe_file, 'r') as hf:
            universe_keys = hf["keys"][:]

        # get all bioactive compounds from libraries (with pubchem names)
        lib_bio_file = os.path.join(self.tmpdir, "lib_bio.json")
        if not os.path.exists(lib_bio_file):
            text_bio = "select  library_description.name as lib,lib.inchikey from libraries as lib INNER JOIN library_description on lib.lib = library_description.lib   where lib.is_bioactive = '1' order by  library_description.rank"
            lib_bio = psql.qstring(text_bio, self.DB)
            ref_bioactive = dict()
            for lib in lib_bio:
                if lib[0] not in ref_bioactive:
                    ref_bioactive[lib[0]] = set()
                ref_bioactive[lib[0]].add(lib[1])
            for lib in lib_bio:
                ref_bioactive[lib[0]] = list(ref_bioactive[lib[0]])
            with open(lib_bio_file, 'w') as outfile:
                json.dump(ref_bioactive, outfile)

        # save chunks of inchikey pubmed synonyms
        ik_names_file = os.path.join(self.tmpdir, "inchies_names.json")
        if not os.path.exists(ik_names_file):
            names_map = {}
            for input_data in self.__chunker(universe_keys):
                data = psql.qstring("select inchikey_pubchem as inchikey,name from pubchem INNER JOIN( VALUES " +
                                    ', '.join('(\'{0}\')'.format(w) for w in input_data) + ") vals(v) ON (inchikey_pubchem = v)", self.DB)
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
                    raise Exception(
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
        params["memory"] = 2
        params["wait"] = True
        # job command
        cc_config_path = self.config.config_path
        cc_package = os.path.join(self.config.PATH.CC_REPO, 'package')
        singularity_image = self.config.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {} {} {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, 
            ik_names_file, lib_bio_file, mol_path, self.DB, version, self.CC_ROOT)
        # submit jobs
        cluster = HPC.from_config(self.config)
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
                raise Exception(
                    "Not all molecules have their json explore file (%d/%d)" % (len(missing_keys), len(universe_keys)))
            else:
                self.__log.error(
                    "Not all molecules have their json explore file (%d/%d)" % (len(missing_keys), len(universe_keys)))
        else:
            shutil.rmtree(job_path, ignore_errors=True)
            self.mark_ready()

    def __chunker(self, data, size=2000):
        for i in range(0, len(data), size):
            yield data[slice(i, i + size)]

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
