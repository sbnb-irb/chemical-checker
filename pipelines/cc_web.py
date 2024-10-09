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
11. Link/copy generated files to webpage repository (mosaic)
12. Export signature 3 to ftp directory

FINAL MANUAL STEPS:

1. Update mosaic/app/shared/data/local_parameters*.json with newly generated NEWDB on aloy-dbsrv
2. Test the website
3. Copy NEWDB to aloy-dbwebsrv
$ pg_dump -h aloy-dbsrv -U 'sbnb-adm' NEWDB | gzip -c > NEWDB.sql.gz
$ createdb -h aloy-dbwebsrv -U 'sbnb-adm' NEWDB
$ gunzip -c NEWDB.sql.gz | psql -h aloy-dbwebsrv -U 'sbnb-adm' NEWDB
4. Update db host in mosaic/app/shared/data/local_parameters*.json to aloy-dbwebsrv

"""
import os
import sys
import h5py
import json
import shutil
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
from chemicalchecker.util.pipeline import MolecularInfo, StatsPlots


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
        'web_repo_path', type=str,
        help='Path to the mosaic web repository '
        '(e.g. `/aloy/home/mbertoni/code/mosaic`)')
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
            'file:///aloy/web_checker/repo_data/zinc_libs/met/met_1.smi',
            'zinc'],
        "tcm": [
            'Tradicional Chinese medicines',
            'Compounds extracted from traditional Chinese medicinal plants',
            'file:///aloy/web_checker/repo_data/zinc_libs/tcm/tcm_1.smi',
            'zinc'],
        "lincs": [
            'LINCS compounds',
            'Collection of compounds of the LINCS initiative',
            'file:///aloy/web_checker/repo_data/zinc_libs/lincs/lincs_1.smi',
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
            'http://zinc15.docking.org/catalogs/ncidiv/items.txt?count=all&output_fields=smiles%20zinc_id',
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
            h5.create_dataset("keys", data=DataSignature.h5_str(universe_list))
            # h5.create_dataset("keys", data=np.array(
            #     universe_list, DataSignature.string_dtype()))
        # also save as json (used by the web)
        bioactive_mol_set = os.path.join(cachedir, "bioactive_mol_set.json")
        json.dump(universe_list.tolist(), open(bioactive_mol_set, 'w'))

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
    #pp.add_task(plots_task)

    # TASK: Generate molecular info
    minfo_task = MolecularInfo(name='molinfo',
                               DB=args.new_web_db, CC_ROOT=args.cc_root)
    pp.add_task(minfo_task)

    # TASK: Generate libraries set
    libs_task = Libraries(name='libraries',
                          DB=args.new_web_db, CC_ROOT=args.cc_root,
                          libraries=libraries)
    pp.add_task(libs_task)

    # TASK: Create statistics plots
    stats_plots_task = StatsPlots(name='stats_plots',
                                  CC_ROOT=args.cc_root)
    pp.add_task(stats_plots_task)

    # TASK: Link/copy generated files to webpage repository (mosaic)
    def links_to_web_repo(cc_root, web_repo_path, tmpdir, cachedir):
        plots_web = os.path.join(cc_root, 'plots_web')
        if not os.path.isdir(plots_web):
            raise Exception(
                "%s not found! Did cc_web.py finish correctly?" %
                plots_web)
        
        # link plots dir
        #src_dir = os.path.join(plots_web, 'plots_home')
        src_dir = plots_web
        if not os.path.isdir(src_dir):
            raise Exception(
                "%s not found! Did cc_web.py finish correctly?" %
                src_dir)
        dst_dir = os.path.join(web_repo_path, 'app', 'images', 'plots')
        if os.path.isdir(dst_dir):
            os.unlink(dst_dir)
        os.symlink(src_dir, dst_dir)
        
        # link statistics dir
        src_dir = os.path.join(plots_web, 'plots_stats')
        if not os.path.isdir(src_dir):
            raise Exception(
                "%s not found! Did cc_web.py finish correctly?" %
                src_dir)
        dst_dir = os.path.join(web_repo_path, 'app', 'images', 'statistics')
        if os.path.isdir(dst_dir):
            os.unlink(dst_dir)
        os.symlink(src_dir, dst_dir)
        
        # link molecule dir
        src_dir = args.molecule_path
        if not os.path.isdir(src_dir):
            raise Exception(
                "%s not found! Did cc_web.py finish correctly?" %
                src_dir)
        dst_dir = os.path.join(web_repo_path, 'app', 'images', 'molecules')
        if os.path.isdir(dst_dir):
           os.unlink(dst_dir)
        os.symlink(src_dir, dst_dir)
        
        # copy bioactive_mol_set.json (aka cc universe)
        src_path = os.path.join(cachedir, 'bioactive_mol_set.json')
        dst_path = os.path.join(web_repo_path, 'app',
                                'shared', 'data', 'bioactive_mol_set.json')
        shutil.copyfile(src_path, dst_path)
        
        # copy inchies_names.json (aka molecule common names)
        src_path = os.path.join(tmpdir, 'inchies_names.json')
        dst_path = os.path.join(web_repo_path, 'app',
                                'shared', 'data', 'inchies_names.json')
        shutil.copyfile(src_path, dst_path)
        
        # generate all inchikeys per coordinate
        cc = ChemicalChecker(args.cc_root)
        ink_coord = {}
        for ds in cc.datasets_exemplary():
            s1 = cc.get_signature('sign1', 'full', ds)
            ink_coord[ds[:2]] = list(s1.keys)
        dst_path = os.path.join(web_repo_path, 'app',
                                'shared', 'data', 'iks_coord.json')
        json.dump(ink_coord, open(dst_path, 'w'))

    links_task = PythonCallable(
        name="links_to_web_repo",
        python_callable=links_to_web_repo,
        op_args=[args.cc_root, args.web_repo_path, pp.tmpdir, pp.cachedir])
    pp.add_task(links_task)

    # TASK: Export to ftp directory the minimum CC to be able 
    # to run a complete CC protocol: a zipped folder is created
    def export_cc_ftp(cc_root, ftp_path='/aloy/web_checker/ftp_data'): 
        cc = ChemicalChecker(cc_root) 
        cc.export_cc(ftp_path, cc.name)

    export_cc_task = PythonCallable(name="export_cc_ftp",
                                 python_callable=export_cc_ftp,
                                 op_args=[args.cc_root],
                                 op_kwargs={ 'ftp_path': '/aloy/web_checker/ftp_data' } )
    pp.add_task(export_cc_task)

    # TASK: Export signatures 3 to ftp directory 
    # leaving this task so if users want to download single sign3 instead of 
    # zipped CC, they can
    def export_sign3_ftp(cc_root, ftp_path='/aloy/web_checker/ftp_data'):
        cc = ChemicalChecker(args.cc_root)
        for ds in cc.datasets_exemplary():
            s3 = cc.get_signature('sign3', 'full', ds)
            dst = os.path.join(ftp_path, cc.name, '%s.h5' % ds[:2])
            cc.export(dst, s3, h5_filter=['keys', 'V', 'confidence', 'known'],
                      h5_names_map={'confidence': 'applicability'})

    export_task = PythonCallable(name="export_sign3_ftp",
                                 python_callable=export_sign3_ftp,
                                 op_args=[args.cc_root],
                                 op_kwargs={ 'ftp_path': '/aloy/web_checker/ftp_data' } )
    
    pp.add_task(export_task)
    
    def export_cc_sign012(cc_root, ftp_path='/aloy/web_checker/ftp_data'): 
        a = ['A','B','C','D','E']
        b = [1,2,3,4,5]
        c = [0,1,2]
        scr = f'{args.cc_root}/full/_sa_/_space_/_space_.001/_sign_/_sign_.h5'
        dest = f'{ftp_path}/2024_02/signature_s_/_space___sign_.h5'
        for i in a:
            for j in b:
                for k in c:
                    os.system( 'ln -sF '+( scr.replace('_sa_', str(i)).replace('_space_', f'{i}{j}').replace('_sign_', f'sign{k}') )+' '+( dest.replace('_s_', str(k)) ).replace('_space_', f'{i}{j}').replace('_sign_', f'sign{k}') )

    export_cc_s012_task = PythonCallable(name="export_cc_sign012",
                                 python_callable=export_cc_sign012,
                                 op_args=[args.cc_root])
    pp.add_task(export_cc_s012_task)
    
    def linkNew_cc_current(cc_root, new_version): 
        os.system( 'unlink /aloy/web_checker/signaturizers/current' )
        os.system( f'ln -s /aloy/web_checker/signaturizers/{ new_version } /aloy/web_checker/signaturizers/current' )
        
        os.system( 'unlink /aloy/web_checker/current' )
        os.system( f'ln -s /aloy/web_checker/package_cc/{ new_version } /aloy/web_checker/current' )

    link_cc_current_task = PythonCallable(name="linkNew_cc_current",
                                 python_callable=linkNew_cc_current,
                                 op_args= [ args.cc_root, args.new_web_db.replace('cc_web_', '') ] )
    pp.add_task( linkNew_cc_current_task )

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

