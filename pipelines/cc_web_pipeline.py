import sys
import os
import numpy as np
import csv
import tempfile
import h5py
from chemicalchecker import ChemicalChecker
from chemicalchecker.database import Datasource
from chemicalchecker.util import HPC
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo
from chemicalchecker.util.pipeline import Pipeline, PythonCallable, Pubchem

os.environ['CC_CONFIG'] = '/aloy/home/oguitart/projects/source/chemical_checker/pipelines/cc_web_update.json'


CC_PATH = "/aloy/web_checker/package_cc/dream_ctd2/"

OLD_DB = 'mosaic'
DB = 'cc_web_2019_05'

data_calculators = ['morgan_fp_r2_2048', 'e3fp_3conf_1024', 'murcko_1024_cframe_1024',
                    'maccs_keys_166', 'general_physchem_properties', 'chembl_target_predictions_v23_10um']


pp = Pipeline(pipeline_path="/aloy/scratch/oguitart/web_cc")


def universe(tmpdir):

    molrepos = Molrepo.get_universe_molrepos()

    print("Querying molrepos")

    inchikeys = set()

    for molrepo in molrepos:

        molrepo = str(molrepo[0])

        inchikeys.update(Molrepo.get_fields_by_molrepo_name(
            molrepo, ["inchikey"]))

    universe_file = os.path.join(tmpdir, "universe.h5")

    universe_list = [str(i[0]) for i in inchikeys]

    universe_list.sort()

    with h5py.File(universe_file, "w") as h5:
        h5.create_dataset("keys", data=np.array(universe_list))

    con = psql.get_connection(OLD_DB)

    con.autocommit = True

    cur = con.cursor()
    try:
        cur.execute('CREATE DATABASE {};'.format(DB))
    except Exception, e:
        raise Exception(e)
    finally:
        con.close()


##### TASK: Create universe file and DB #######

universe_params = {}

universe_params['python_callable'] = universe
universe_params['op_args'] = [pp.tmpdir]

universe_task = PythonCallable(name="universe", **universe_params)

pp.add_task(universe_task)


# TASK: Calculate signatures 0
pbchem_params = {'DB': DB, 'OLD_DB': OLD_DB}
pbchem_task = Pubchem(name='pubchem', **pbchem_params)
pp.add_task(pbchem_task)


pp.run()
