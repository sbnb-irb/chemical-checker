"""Pipeline to update the CC web resource.

The steps (a.k.a. tasks) for CC web update are the following:

1. Generate a universe file (i.e. HDF5 with all inchikeys sorted)
2. Create a new web DB
3. Drop/Create/Fill the `pubchem` table (used e.g. for drug synonyms)
4. Drop/Create/Fill the `showtargets` and `showtargets_description` tables (known drug tagets)
5. Drop/Create/Fill the `coordinates` and `coordinate_stats` tables (spaces description and projection xy limits)
6. Drop/Create/Fill the `projections` table (xy for each proj2 molecule)
7. Create 2d svg molecule images for each molecule
8. Drop/Create/Fill the `molecular_info` table (popularity singularity mappability etc.)
9. Drop/Create/Fill the `libraries` and `library_description` tables (used to fetch 100 landmark molecules)
10. Generate explore.json file for each molecule (info for explore drug page)
"""
import os
import sys
import h5py
import logging
import argparse
import numpy as np

from chemicalchecker import ChemicalChecker
from chemicalchecker.core import DataSignature
from chemicalchecker.util import psql
from chemicalchecker.util import logged, Config
from chemicalchecker.util.pipeline import Pipeline, PythonCallable, Pubchem
from chemicalchecker.util.pipeline import ShowTargets, Libraries, Similars
from chemicalchecker.util.pipeline import Coordinates, Projections, Plots
from chemicalchecker.util.pipeline import MolecularInfo


def pipeline_parser():
    """Parse pipeline arguments."""
    description = 'Run the full CC update pipeline.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'cc_root', type=str,
        help='Directory of the CC instance the web will refere to '
        '(e.g. `/aloy/web_checker/package_cc/miniCC`)')
    parser.add_argument(
        'pipeline_dir', type=str,
        help='Directory where the pipeline will run '
        '(e.g. `/aloy/scratch/mbertoni/pipelines/miniCC_web`)')
    parser.add_argument(
        'molecule_path', type=str,
        help='Directory where the molecule images will be stored '
        '(e.g. `/aloy/web_checker/molecules`)')
    parser.add_argument(
        'old_web_db', type=str,
        help='Previous web database '
        '(e.g. `mosaic` or `cc_web_2019_05`)')
    parser.add_argument(
        'new_web_db', type=str,
        help='Web database to generate '
        '(e.g. `cc_web_2020_01`)')
    parser.add_argument(
        'uniprot_db', type=str,
        help='Uniprot db to use '
        '(e.g. `2019_01`)')
    parser.add_argument(
        '-t', '--only_tasks', type=str, nargs="+", default=[],
        required=False,
        help='Names of tasks that will `exclusively` run by the pipeline.')
    parser.add_argument(
        '-s', '--exclude_tasks', type=str, nargs="+", default=[],
        required=False,
        help='Names of tasks that will be skipped.')
    parser.add_argument(
        '-c', '--config', type=str, required=False,
        default=os.environ["CC_CONFIG"],
        help='Config file to be used. If not specified CC_CONFIG enviroment'
        ' variable is used.')
    parser.add_argument(
        '-d', '--dry_run', action='store_true',
        help='Execute pipeline script without running the pipeline.')
    return parser


@logged(logging.getLogger("[ PIPELINE %s ]" % os.path.basename(__file__)))
def main(args):
    # initialize Pipeline
    cfg = Config(args.config)
    pp = Pipeline(pipeline_path=args.pipeline_dir, keep_jobs=True,
                  config=cfg, only_tasks=args.only_tasks,
                  exclude_tasks=args.exclude_tasks)

    # print arguments
    for arg in vars(args):
        main._log.info('[ ARGS ] {:<25s}: {}'.format(arg, getattr(args, arg)))

    libraries = {
        "apd": [
            'Approved drugs',
            'Approved drug molecules from DrugBank',
            'http://zinc15.docking.org/catalogs/dbap/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "exd": [
            'Experimental drugs',
            'Experimental and investigational drug molecules from DrugBank',
            'http://zinc15.docking.org/catalogs/dbex/'
            'items.txt?count=all&output_fields=smiles%20zinc_id;'
            'http://zinc15.docking.org/catalogs/dbin/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "met": [
            'Human metabolites',
            'Endogenous metabolites from Human Metabolome Database (HMDb)',
            'http://zinc15.docking.org/catalogs/hmdbendo/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "tcm": [
            'Tradicional Chinese medicines',
            'Compounds extracted from traditional Chinese medicinal plants',
            'http://zinc15.docking.org/catalogs/tcmnp/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "lincs": [
            'LINCS compounds',
            'Collection of compounds of the LINCS initiative',
            'http://zinc15.docking.org/catalogs/lincs/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "pwck": [
            'Prestwick chemical library',
            'Prestwick commercial collection',
            'http://zinc15.docking.org/catalogs/prestwick/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "nihcc": [
            'NIH clinical collection',
            'NIH clinical collection',
            'http://zinc15.docking.org/catalogs/nihcc/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "ncidiv": [
            'NCI diversity collection',
            'NCI diversity collection',
            'http://zinc15.docking.org/catalogs/ncidiv/'
            'items.txt?count=all&output_fields=smiles%20zinc_id',
            'zinc'],
        "tool": [
            'Tool compounds',
            'Tool compounds',
            'http://zinc15.docking.org/toolcompounds.smi?count=all',
            'zinc']
    }

    # TASK: Create universe file (used by most of following steps)
    def create_uni_fn(cc_root, cachedir):
        cc = ChemicalChecker(args.cc_root)
        universe_list = cc.get_signature('sign3', 'full', 'A1.001').keys
        universe_file = os.path.join(cachedir, "universe.h5")
        with h5py.File(universe_file, "w") as h5:
            h5.create_dataset("keys", data=np.array(
                universe_list, DataSignature.string_dtype()))

    universe_task = PythonCallable(name="create_universe",
                                   python_callable=create_uni_fn,
                                   op_args=[args.cc_root, pp.cachedir])
    pp.add_task(universe_task)

    # TASK: Create DB
    def create_db_fn():
        con = psql.get_connection(args.old_web_db)
        con.autocommit = True
        success = False
        cur = con.cursor()
        try:
            cur.execute('CREATE DATABASE {};'.format(args.new_web_db))
            success = True
        except Exception as e:
            print(e)
        finally:
            con.close()
        if not success:
            raise Exception("Cannot create DB.")

    db_task = PythonCallable(name="create_db",  python_callable=create_db_fn)
    pp.add_task(db_task)

    # TASK: Fill Pubchem table
    pbchem_task = Pubchem(name='pubchem',
                          DB=args.new_web_db, OLD_DB=args.old_web_db)
    pp.add_task(pbchem_task)

    # TASK: Find targets on uniprot and fill local table
    targets_task = ShowTargets(name='showtargets',
                               DB=args.new_web_db, CC_ROOT=args.cc_root,
                               uniprot_db_version=args.uniprot_db)
    pp.add_task(targets_task)

    # TASK: Fill coordinates
    coords_task = Coordinates(name='coordinates',
                              DB=args.new_web_db, CC_ROOT=args.cc_root)
    pp.add_task(coords_task)

    # TASK: Fill projecions
    projs_task = Projections(name='projections',
                             DB=args.new_web_db, CC_ROOT=args.cc_root)
    pp.add_task(projs_task)

    # TASK: Create all plots
    plots_task = Plots(name='plots',
                       DB=args.new_web_db, CC_ROOT=args.cc_root,
                       MOLECULES_PATH=args.molecule_path)
    pp.add_task(plots_task)

    # TASK: Generate molecular info
    minfo_task = MolecularInfo(name='molinfo',
                               DB=args.new_web_db, CC_ROOT=args.cc_root)
    pp.add_task(minfo_task)

    # TASK: Generate libraries set
    libs_task = Libraries(name='libraries',
                          DB=args.new_web_db, CC_ROOT=args.cc_root,
                          libraries=libraries)
    pp.add_task(libs_task)

    # TASK: Create json of similar molecules for explore page
    similars_task = Similars(name='similars',
                             DB=args.new_web_db, CC_ROOT=args.cc_root,
                             MOLECULES_PATH=args.molecule_path)
    pp.add_task(similars_task)

    # RUN the pipeline!
    main._log.info('TASK SEQUENCE: %s' % ', '.join([t.name for t in pp.tasks]))
    if not args.dry_run:
        pp.run()


if __name__ == '__main__':
    # parse arguments
    args = pipeline_parser().parse_args(sys.argv[1:])
    main(args)
