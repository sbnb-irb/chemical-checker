import tempfile
import os
import shutil
from chemicalchecker.util import logged
from chemicalchecker.util import HPC
from chemicalchecker.util import BaseStep
from chemicalchecker.util import psql
from chemicalchecker.util import Config

# We got these strings by doing: pg_dump -t 'scores' --schema-only mosaic
# -h aloy-dbsrv

DROP_TABLE_DESC = "DROP TABLE IF EXISTS public.library_description"

DROP_TABLE = "DROP TABLE IF EXISTS public.libraries"

CREATE_TABLE = """CREATE TABLE public.libraries (
    inchikey text,
    lib text,
    is_bioactive smallint,
    is_landmark smallint
);"""

CREATE_TABLE_DESC = """CREATE TABLE public.library_description (
    lib text NOT NULL PRIMARY KEY,
    name text,
    description text,
    urls text,
    parser text,
    rank integer
);"""

CREATE_INDEX = """
CREATE INDEX inchikey_libraries_idx ON public.libraries USING btree (inchikey);
CREATE INDEX is_bioactive_libraries_idx ON public.libraries USING btree (is_bioactive);
CREATE INDEX is_landmark_libraries_idx ON public.libraries USING btree (is_landmark);
CREATE INDEX lib_libraries_idx ON public.libraries USING btree (lib);
ALTER TABLE ONLY public.libraries ADD CONSTRAINT libraries_lib_fkey FOREIGN KEY (lib) REFERENCES public.library_description(lib);
"""

CREATE_INDEX_DESC = """
CREATE INDEX lib_library_description_idx ON public.library_description USING btree (lib);
CREATE INDEX name_library_description_idx ON public.library_description USING btree (name);
CREATE INDEX rank_library_description_idx ON public.library_description USING btree (rank);
"""

INSERT_DESC = "INSERT INTO library_description VALUES %s"

CHECK = "select distinct(lib) from libraries"


@logged
class Libraries(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molecular info step."""

        config_cc = Config()

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(
            config_cc.PATH.CC_REPO, "pipeline_web", "steps", "scripts", "libraries.py")

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.config.DB)
            psql.query(CREATE_TABLE, self.config.DB)
            psql.query(DROP_TABLE_DESC, self.config.DB)
            psql.query(CREATE_TABLE_DESC, self.config.DB)

        except Exception, e:

            self.__log.error(e)

        libraries = self.config.STEPS[self.name].asdict()
        i = 1
        for idname, v in libraries.iteritems():
            s = "('%s',%s,%d)" % (idname, ",".join(["'%s'" % x for x in v]), i)
            psql.query(INSERT_DESC % s, self.config.DB)
            i += 1

        self.__log.info("Genretaing libraries for " +
                        str(len(libraries.keys())) + " libraries")

        job_path = tempfile.mkdtemp(
            prefix='jobs_libraries_', dir=self.tmpdir)

        libraries_path = os.path.join(self.tmpdir, "libraries_files")

        if not os.path.exists(libraries_path):
            original_umask = os.umask(0)
            os.makedirs(libraries_path, 0o775)
            os.umask(original_umask)

        params = {}
        params["num_jobs"] = len(libraries.keys())
        params["jobdir"] = job_path
        params["job_name"] = "CC_LIBRARIES"
        params["elements"] = libraries
        params["wait"] = True
        params["memory"] = 20
        params["cpu"] = 10
        # job command
        singularity_image = config_cc.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, universe_file, libraries_path, self.config.DB)
        # submit jobs
        cluster = HPC(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        try:
            self.__log.info("Checking table")
            libs = psql.qstring(CHECK, self.config.DB)
            if len(libs) != len(libraries.keys()):
                self.__log.error(
                    "Not all libs were added to libraries (%d/%d)" % (len(libs), len(libraries.keys())))
            else:
                self.__log.info("Indexing table")
                psql.query(CREATE_INDEX, self.config.DB)
                psql.query(CREATE_INDEX_DESC, self.config.DB)
                shutil.rmtree(job_path)
                self.mark_ready()
        except Exception, e:

            self.__log.error(e)
