"""Load synonyms.

To avoid querying all the synonyms we reuse what we already stored in a
previous version of the DB and we only query missing compounds.
"""
import os
import h5py
import shutil
import tempfile

from chemicalchecker.util import psql
from chemicalchecker.util import Config
from chemicalchecker.util import logged, HPC
from chemicalchecker.util.pipeline import BaseTask

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
class Pubchem(BaseTask):

    def __init__(self, name=None, **params):
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name
        BaseTask.__init__(self, name, **params)

        self.DB = params.get('DB', None)
        self.OLD_DB = params.get('OLD_DB', None)
        if self.DB is None:
            raise Exception('DB parameter is not set')
        if self.OLD_DB is None:
            raise Exception('OLD_DB parameter is not set')
    
    def _dump_table( self, host_name,database_name,user_name,database_password,table_name, infile):

        command = 'PGPASSWORD={0} pg_dump -h {1} -d {2} -U {3} -p 5432 -t public.{4} -F plain -E "UTF-8" > {5}'\
        .format( database_password, host_name, database_name, user_name, table_name, infile)
        os.system( command )

    def _restore_table( self, host_name,database_name,user_name,database_password, outfile):

        if( os.path.isfile(outfile) ):
            command = 'PGPASSWORD={4} psql -h {0} -d {1} -U {2} -f {3}'\
                      .format(host_name, database_name, user_name, outfile, database_password)
            os.system( command )

    def _transform_sql( self, keys, table_new, infile, outfile ):
        found_record = False
        flag = False
        header = ''
        g = open( outfile, 'wb' )
        f = open( infile, 'rb' )
        for line in f:
            line = line.decode('UTF-8')
            if( flag and line.startswith('\\.') ):
                flag = False
                g.write( '\\.\n'.encode('UTF-8') )
                
            if(flag):
                if( header != ''):
                    g.write( header.encode('UTF-8') )
                    header = ''
                    
                lst = line.split('\t')
                if( len(lst) > 2 ):
                    key = lst[1]
                    if( key in keys ):
                        if( lst[3].lower() not in lst[4].lower() ):
                            lst[4] += lst[3] + '; '
                        line = '\t'.join(lst)
                        g.write( line.encode('UTF-8') )
                        found_record = True
                    
            if( line.startswith('COPY public.pubchem') ):
                header = f'''
    --
    -- PostgreSQL database dump
    --

    SET statement_timeout = 0;
    SET lock_timeout = 0;
    SET client_encoding = 'UTF8';
    SET standard_conforming_strings = on;
    SELECT pg_catalog.set_config('search_path', '', false);
    SET check_function_bodies = false;
    SET xmloption = content;
    SET client_min_messages = warning;
    SET row_security = off;

    SET default_tablespace = '';

    COPY public.{table_new} (cid, inchikey_pubchem, inchikey, name, synonyms, pubchem_name, iupac_name, direct_parent) FROM stdin;
                '''
                flag = True
                
        f.close()
        g.close()  
        
        if( not found_record ):
            os.remove( outfile )  

    def import_key_data_from_old_db( self, keys ):
        c = Config()
        host = c.DB.host
        user = c.DB.user
        passwd = c.DB.password
        
        db_old = self.OLD_DB
        table_old = 'pubchem'
        db_new = self.DB
        table_new = 'pubchem'
        tmpdir = self.tmpdir
        
        keys = set(keys)
        infile = os.path.join( tmpdir, 'pubchem_web.sql' )
        outfile = os.path.join( tmpdir, 'pubchem_web_final.sql' )
        
        print('Dumping')
        self.__log.info( 'Dumping' )
        self._dump_table( host, db_old, user, passwd, table_old, infile)
        
        print('Preparing molecule lines to copy')
        self.__log.info( 'Preparing molecule lines to copy' )
        self._transform_sql( keys, table_new, infile, outfile)
        
        print('Restoring bkp molecules')
        self.__log.info( 'Restoring bkp molecules' )
        self._restore_table( host, db_new, user, passwd, outfile)
        
        os.remove(infile)
        os.remove(outfile)

    def __chunker(self, data, size=2000):
        for i in range(0, len(data), size):
            yield data[slice(i, i + size)]
    
    def _prepare_inchikey_names(self, keys):
        # save chunks of inchikey pubmed synonyms
        ik_names_file = os.path.join(self.tmpdir, "inchies_names.json")
        if not os.path.exists(ik_names_file):
            names_map = {}
            for input_data in self.__chunker(keys):
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
    
    def run(self):
        """Run the pubchem step."""
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/load_pubchem.py")
        
        
        try:
            self.__log.info("Creating table")
            #psql.query(DROP_TABLE, self.DB)
            psql.query(CREATE_TABLE, self.DB)
            # psql.query(CREATE_INDEX, self.DB)
        except Exception as e:
            self.__log.error(e)
        
        universe_file = os.path.join(self.cachedir, "universe.h5")
        with h5py.File(universe_file, 'r') as h5:
            all_data_size = h5["keys"].shape[0]
            keys = list( h5['keys'][:] )
        universe = [ k.decode('utf8') for k in keys ]
        
        # Importing data from previous CC web db version
        self.import_key_data_from_old_db( universe )
        
        # query to see if there is some data filled in new db
        SELECT_CHECK = "SELECT DISTINCT (inchikey) FROM pubchem ;" 
        rows = psql.qstring( SELECT_CHECK, self.DB)
        done = set( [el[0] for el in rows] )
        keys = list( set(temp) - done )
        data_size = len(keys)
        
        self.__log.info("Generating pubchem data for %s molecules",
                        data_size)
        chunk_size = 1000
        chunks = list()
        """
        for i in range(0, data_size, chunk_size):
            chunks.append(slice(i, i + chunk_size))
        """
        for i in range(0, data_size, chunk_size):
            chunks.append( keys[i:(i + chunk_size)] )

        job_path = tempfile.mkdtemp(prefix='jobs_pubchem_', dir=self.tmpdir)
        if( data_size > 0 ):
            params = {}
            params["num_jobs"] = 4
            params["jobdir"] = job_path
            params["job_name"] = "CC_PUBCH"
            params["elements"] = chunks
            params["wait"] = True

            # job command
            cc_config_path = self.config.config_path
            cc_package = os.path.join(self.config.PATH.CC_REPO, 'package')
            singularity_image = self.config.PATH.SINGULARITY_IMAGE
            command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" \
            " singularity exec {} python {} <TASK_ID> <FILE> {} {} {}"
            command = command.format(
            cc_package, cc_config_path, singularity_image,
            script_path, universe_file, self.OLD_DB, self.DB)

            # submit jobs
            cluster = HPC.from_config(self.config)
            jobs = cluster.submitMultiJob(command, **params)

        try:
            self.__log.info("Checking table")
            count = psql.qstring(COUNT, self.DB)
            if int(count[0][0]) != all_data_size:
                if not self.custom_ready():
                    raise Exception(
                        "Not all universe keys were added to Pubchem (%d/%d)" %
                        (int(count[0][0]), all_data_size))
                else:
                    self.__log.error(
                        "Not all universe keys were added to Pubchem (%d/%d)" %
                        (int(count[0][0]), all_data_size))
            else:
                self.__log.info("Indexing table")
                try:
                    psql.query(CREATE_INDEX, self.DB)
                    
                except:
                    self.__log.info("Indexes already created")
                
                # save chunks of inchikey pubmed synonyms
                self._prepare_inchikey_names( universe )
                
                if( os.path.isdir(job_path) ):
                    shutil.rmtree(job_path, ignore_errors=True)
                self.mark_ready()
        except Exception as e:
            self.__log.error(e)

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
