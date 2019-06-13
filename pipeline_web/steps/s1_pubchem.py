import tempfile
import h5py
import os
import pubchempy
import requests
import json
import shutil
from chemicalchecker.util import logged
from chemicalchecker.util import HPC
from chemicalchecker.util import BaseStep
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
class Pubchem(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the pubchem step."""

        config_cc = Config()

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(
            config_cc.PATH.CC_REPO, "pipeline_web", "steps", "scripts", "load_pubchem.py")

        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.config.DB)
            psql.query(CREATE_TABLE, self.config.DB)
            # psql.query(CREATE_INDEX, self.config.DB)

        except Exception, e:

            self.__log.error(e)

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        with h5py.File(universe_file) as h5:
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
                                 script_path, universe_file, self.config.OLD_DB, self.config.DB)
        # submit jobs
        cluster = HPC(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        try:
            self.__log.info("Checking table")
            count = psql.qstring(COUNT, self.config.DB)
            if int(count[0][0]) != data_size:
                self.__log.error(
                    "Not all universe keys were added to Pubchem (%d/%d)" % (int(count[0][0]), data_size))
            else:
                self.__log.info("Indexing table")
                psql.query(CREATE_INDEX, self.config.DB)
                shutil.rmtree(job_path)
                self.mark_ready()
        except Exception, e:

            self.__log.error(e)

    def __formatting(self, text):

        new_text = list()
        for t in text:
            if t is None:
                new_text.append("''")
                continue
            if type(t) == int:
                if t == -1:
                    new_text.append('NULL')
                else:
                    new_text.append(str(t))
            else:
                if t is None:
                    t = ''
                new_text.append("'" + t.replace("'", "''") + "'")

        return "(" + ','.join(new_text) + ")"

    def __query_direct(self, ik):

        direct_parent = ''

        try:

            r = requests.get(
                'http://classyfire.wishartlab.com/entities/' + ik + '.json')

            if r.status_code == 200:
                djson = json.loads(r.text)
                direct_parent = djson["direct_parent"]["name"]
                if direct_parent is None:
                    direct_parent = ''
                # print direct_parent
        except Exception as e:
            self.__log.error(str(e))

        return direct_parent

    def __query_missing_data(self, missing_keys):

        input_data = pubchempy.get_compounds(missing_keys, 'inchikey')

        rows = list()

        items = set(missing_keys)

        for dt in input_data:

            data = dt.to_dict(
                properties=['synonyms', 'cid', 'iupac_name', 'inchikey'])

            ik = data["inchikey"]

            if ik not in items:
                continue

            name = ''
            pubchem_name = ''
            if len(data['synonyms']) > 0:
                name = data['synonyms'][0]
                pubchem_name = name

            if name == '' and data['iupac_name'] != '':
                name = data['iupac_name']

            direct_parent = self.__query_direct(ik)

            if name == '' and direct_parent != '':
                name = direct_parent

            new_data = (data['cid'], ik, ik, name, ';'.join(data['synonyms']), pubchem_name, data[
                        'iupac_name'], direct_parent)

            rows.append(new_data)

            items.remove(ik)

        print len(items), len(rows)

        if len(items) > 0:

            for ik in items:

                print len(rows), ik

                name = ''
                direct_parent = self.__query_direct(ik)

                if name == '' and direct_parent != '':
                    name = direct_parent

                new_data = (-1, '', ik, name, '', '', '', direct_parent)

                rows.append(new_data)

        if len(rows) < len(missing_keys):
            raise Exception("Not all universe is added to Pubchem table (%d/%d) " %
                            (len(rows), len(missing_keys)))

        return rows
