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
from chemicalchecker.core import DataSignature
from chemicalchecker.database import Molrepo
from chemicalchecker.util.pipeline import Pipeline, PythonCallable, Pubchem, ShowTargets
from chemicalchecker.util.pipeline import Coordinates, Projections, Plots, MolecularInfo, SimilarsSign3

os.environ['CC_CONFIG'] = '/aloy/home/oguitart/projects/source/chemical_checker/pipelines/cc_web_update.json'


CC_PATH = "/aloy/web_checker/package_cc/newpipe/"
MOLECULES_PATH = '/aloy/web_checker/molecules/'

OLD_DB = 'mosaic'
DB = 'cc_web_2019_05'

uniprot_db_version = '2019_01'

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
        h5.create_dataset("keys", data=np.array(
            universe_list, DataSignature.string_dtype()))

    con = psql.get_connection(OLD_DB)

    con.autocommit = True

    success = False
    cur = con.cursor()
    try:
        cur.execute('CREATE DATABASE {};'.format(DB))
        success = True
    except Exception as e:
        print(e)
    finally:
        con.close()

    if not success:
        raise Exception("Universe file and DB not created successfully")

##### TASK: Create universe file and DB #######

universe_params = {}

universe_params['python_callable'] = universe
universe_params['op_args'] = [pp.tmpdir]

universe_task = PythonCallable(name="universe", **universe_params)

pp.add_task(universe_task)


# TASK: Fill Pubchem table
pbchem_params = {'DB': DB, 'OLD_DB': OLD_DB}
pbchem_task = Pubchem(name='pubchem', **pbchem_params)
pp.add_task(pbchem_task)

# TASK: Find targets
targets_params = {'DB': DB, 'CC_ROOT': CC_PATH,
                  'uniprot_db_version': uniprot_db_version}
targets_task = ShowTargets(name='showtargets', **targets_params)
pp.add_task(targets_task)

# TASK: Fill coordinates
coords_params = {'DB': DB, 'CC_ROOT': CC_PATH}
coords_task = Coordinates(name='coordinates', **coords_params)
pp.add_task(coords_task)

# TASK: Fill coordinates
projs_params = {'DB': DB, 'CC_ROOT': CC_PATH}
projs_task = Projections(name='projections', **projs_params)
pp.add_task(projs_task)

# TASK: Create all plots
plots_params = {'DB': DB, 'CC_ROOT': CC_PATH, 'MOLECULES_PATH': MOLECULES_PATH}
plots_task = Plots(name='plots', **plots_params)
pp.add_task(plots_task)

# TASK: Generate similars for sign3
sim3_params = {'CC_ROOT': CC_PATH}
sim3_task = SimilarsSign3(name='sim3', **sim3_params)
pp.add_task(sim3_task)

# TASK: Generate molecular info
minfo_params = {'DB': DB, 'CC_ROOT': CC_PATH}
minfo_task = MolecularInfo(name='molinfo', **minfo_params)
pp.add_task(minfo_task)

pp.run()
