"""Pipeline to generate a full CC update.

The steps (a.k.a. tasks) for a full CC update are the following:

1. Download Datasources (raw input including any external DB)
2. Generate Molrepos (parse downloaded files and save molecules identifiers)
3. Generate Validation sets (MoA and ATC)
4. Compute Sign0
4.1. Sign0 for B,C,D,E levels (CC universe definition)
4.2. Calculate A level (chemistry properties)
4.3. Sign0 for A level
5. Compute Sign1 (and neig clus and proj)
6. Compute Sign2 (and neig clus and proj)
7. Generate stacked Sign2 for universe
8. Compute Sign3
9. Create symlinks
"""
import os
import sys
import logging
import argparse
import tempfile
from update_resources.generate_chembl_files import generate_chembl_files

from chemicalchecker import ChemicalChecker
from chemicalchecker.core import Validation
from chemicalchecker.core.sign3 import sign3
from chemicalchecker.util import Config, HPC, logged
from chemicalchecker.database import Dataset, Datasource
from chemicalchecker.database import Molrepo, Molecule, Calcdata
from chemicalchecker.util.pipeline import Pipeline, PythonCallable, CCFit


def pipeline_parser():
    """Parse pipeline arguments."""
    description = 'Run the full CC update pipeline.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'cc_root', type=str,
        help='Directory where the new CC instance will be generated '
        '(e.g. `/aloy/web_checker/package_cc/miniCC`)')
    parser.add_argument(
        'pipeline_dir', type=str,
        help='Directory where the pipeline will run '
        '(e.g. `/aloy/scratch/mbertoni/pipelines/miniCC`)')
    parser.add_argument(
        '-r', '--reference_cc', type=str, required=False,
        help='Root dir of the CC instance to use as reference '
        '(i.e. triplet sampling in sign0).')
    parser.add_argument(
        '-c', '--config', type=str, required=False,
        help='Config file to be used. If not specified CC_CONFIG enviroment'
        ' variable is used.')
    parser.add_argument(
        '-c', '--config', type=str, required=False,
        default=os.environ["CC_CONFIG"],
        help='Config file to be used. If not specified CC_CONFIG enviroment'
        ' variable is used.')
    parser.add_argument(
        '-d', '--dry_run', type=bool, required=False, default=False,
        help='Execute pipeline script without running the pipeline.')
    return parser


@logged(logging.getLogger("[ PIPELINE %s ]" % os.path.basename(__file__)))
def main(args):
    # print arguments
    for arg in vars(args):
        main._log.info('[ ARGS ] {:<25s}: {}'.format(arg, getattr(args, arg)))

    # initialize Pipeline
    pp = Pipeline(pipeline_path=args.pipeline_dir, keep_jobs=True,
                  config=Config(args.config))

    fit_order = ['sign0', 'sign1', 'clus1', 'proj1', 'neig1',
                 'sign2', 'clus2', 'proj2', 'neig2', 'sign3']

    data_calculators = [
        'morgan_fp_r2_2048',
        'e3fp_3conf_1024',
        'murcko_1024_cframe_1024',
        'maccs_keys_166',
        'general_physchem_properties',
        'chembl_target_predictions_v23_10um'
    ]

    validation_sets = ['moa', 'atc']

    # HPC parameters
    hpc_kwargs = {
        'sign0': {'memory': 44, 'cpu': 22},
        'sign1': {'memory': 40, 'cpu': 10},
        'sign2': {'memory': 20, 'cpu': 16},
        'sign3': {'memory': 2,  'cpu': 32},
        'neig1': {'memory': 30, 'cpu': 15},
        'neig2': {'memory': 30, 'cpu': 15},
        'neig3': {'memory': 30, 'cpu': 15},
        'clus1': {'memory': 20, 'cpu': 10},
        'clus2': {'memory': 20, 'cpu': 10},
        'clus3': {'memory': 20, 'cpu': 10},
        'proj1': {'memory': 20, 'cpu': 10},
        'proj2': {'memory': 20, 'cpu': 10},
        'proj3': {'memory': 20, 'cpu': 10}
    }

    # on which signature molset to call the fit?
    molset = {
        'sign0': 'full',
        'sign1': 'full',
        'sign2': 'reference',
        'sign3': 'full',
        'neig1': 'reference',
        'neig2': 'reference',
        'neig3': 'reference',
        'clus1': 'reference',
        'clus2': 'reference',
        'clus3': 'reference',
        'proj1': 'reference',
        'proj2': 'reference',
        'proj3': 'reference'
    }

    # dataset parameters
    datasets = [ds.code for ds in Dataset.get(exemplary=True)]
    data_file = os.path.join(args.preprocess_path, '%s.h5')
    sign_kwargs = {}
    fit_kwargs = {}
    for cctype in fit_order:
        fit_kwargs[cctype] = {}
        sign_kwargs[cctype] = {}
    # sign3 shared parameters
    cc = ChemicalChecker(args.cc_root)
    sign2_universe = os.path.join(pp.cache, "sign2_universe_stacked.h5")
    sign2_coverage = os.path.join(pp.cache, "sign2_universe_coverage.h5")
    sign2_list = [cc.get_signature('sign2', 'full', ds)
                  for ds in cc.datasets_exemplary()]
    mfp = cc.get_signature('sign0', 'full', 'A1.001').data_path
    for ds in datasets:
        fit_kwargs['sign0'][ds] = {
            'key_type': 'inchikey',
            'data_file': data_file % ds[:2],
            'do_triplets': False,
            'validations': False
        }
        fit_kwargs['sign1'][ds] = {
            'metric_learning': False,
        }
        fit_kwargs['sign2'][ds] = {
            'validations': False,
        }
        sign_kwargs['sign2'][ds] = {
            'node2vec': {'cpu': 4},
            'adanet': {'cpu': 16}
        }
        fit_kwargs['sign3'][ds] = {
            'sign2_list': sign2_list,
            'sign2_universe': sign2_universe,
            'sign2_coverage': sign2_coverage,
            'sign0': mfp,
        }
        sign_kwargs['sign3'][ds] = {
            'sign2': {'cpu': 32}
        }
        sign_kwargs['neig1'][ds] = {
            'cpu': 15
        }
        sign_kwargs['neig2'][ds] = {
            'cpu': 15
        }
        sign_kwargs['clus1'][ds] = {
            'cpu': 10,
            'general_params': {'balance': 1.5}
        }
        sign_kwargs['clus2'][ds] = {
            'cpu': 10
        }
        sign_kwargs['proj1'][ds] = {
            'cpu': 10
        }
        sign_kwargs['proj2'][ds] = {
            'cpu': 10
        }

    #############################################
    # TASK: Download all datasources
    def download_fn(tmpdir):
        # Generate the Chembl files drugtargets and drugindications via
        # Chembl Python API in /aloy/web_checker/repo_data
        generate_chembl_files()

        job_path = tempfile.mkdtemp(prefix='jobs_download_', dir=tmpdir)
        # start download jobs (one per Datasource)
        # job will wait until finished
        job = Datasource.download_hpc(job_path, only_essential=True)
        if job.status() == HPC.ERROR:
            main._log.error(
                "There are errors in some of the downloads jobs")

        # check if the downloads are really done
        if not Datasource.test_all_downloaded(only_essential=True):
            main._log.error(
                "Something went WRONG while DOWNLOADING, please retry")
            # print the faulty one
            missing_datasources = set()
            for ds in Datasource.get():
                for dset in ds.datasets:
                    if dset.essential:
                        missing_datasources.add(ds)
                        break
                for molrepo in ds.molrepos:
                    if molrepo.essential:
                        missing_datasources.add(ds)
                        break
            for ds in missing_datasources:
                if not ds.available:
                    main._log.error("Datasource %s not available" % ds)
            raise Exception('Not all datasources were downloaded correctly')

    downloads_task = PythonCallable(name="downloads",
                                    python_callable=download_fn,
                                    op_args=[pp.tmpdir])
    pp.add_task(downloads_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Generate Molrepos
    job_path = tempfile.mkdtemp(prefix='jobs_molrepos_', dir=pp.tmpdir)
    molrepos_task = PythonCallable(name="molrepos",
                                   python_callable=Molrepo.molrepo_hpc,
                                   op_args=[job_path],
                                   op_kwargs={'only_essential': True})
    pp.add_task(molrepos_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Generate validation sets
    def create_val_set_fn(set_name):
        print("Creating validation set for " + set_name)
        val = Validation(Config(args.config).PATH.validation_path, set_name)
        try:
            val.run()
        except Exception as ex:
            print(ex)
            raise Exception("Validation set '%s' not working" % set_name)

    for val_set in validation_sets:
        val_set_task = PythonCallable(
            name="val_set_" + val_set,
            python_callable=create_val_set_fn,
            op_args=[val_set])
        pp.add_task(val_set_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 0 for derived spaces only (i.e. BCDE levels)
    cctype = 'sign0'
    dss = [ds.code for ds in Dataset.get(exemplary=True) if ds.derived]
    task = CCFit(args.cc_root, cctype, molset[cctype],
                 datasets=dss,
                 fit_kwargs=fit_kwargs[cctype],
                 sign_kwargs=sign_kwargs[cctype],
                 hpc_kwargs=hpc_kwargs[cctype])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate data (defined for universe)
    def calculate_data_fn(type_data, tmpdir, inchikey_inchi):
        main._log.info("Calculating data for " + type_data)
        job_path = tempfile.mkdtemp(
            prefix='jobs_molprop_' + type_data + "_", dir=tmpdir)
        calculator = Calcdata(type_data)
        # This method sends the job and waits for the job to finish
        calculator.calcdata_hpc(job_path, list(inchikey_inchi))
        missing = len(calculator.get_missing_from_set(inchikey_inchi))
        if missing > 0:
            raise Exception(
                "Not all molecular properties were calculated. There are " +
                str(missing) + " missing out of " + str(len(inchikey_inchi)))

    # after running the first tranche of sign0 we known the CC universe
    universe = cc.universe
    main._log.info('CC Universe will include %s molecules.' % len(universe))
    inchikey_inchi = Molecule.get_inchikey_inchi_mapping(universe)
    for data_calc in data_calculators:
        calc_data_task = PythonCallable(
            name="calc_data_" + data_calc,
            python_callable=calculate_data_fn,
            op_args=[data_calc, pp.tmpdir, inchikey_inchi])
        pp.add_task(calc_data_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 0 for non derived spaces only (i.e. A level)
    cctype = 'sign0'
    dss = [ds.code for ds in Dataset.get(exemplary=True) if not ds.derived]
    task = CCFit(args.cc_root, cctype, molset[cctype],
                 datasets=dss,
                 fit_kwargs=fit_kwargs[cctype],
                 sign_kwargs=sign_kwargs[cctype],
                 hpc_kwargs=hpc_kwargs[cctype])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 1-2 also clus, proj and neig
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    for cctype in fit_order:
        task = CCFit(args.cc_root, cctype, molset[cctype],
                     datasets=dss,
                     fit_kwargs=fit_kwargs[cctype],
                     sign_kwargs=sign_kwargs[cctype],
                     hpc_kwargs=hpc_kwargs[cctype])
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Cache sign2 universe
    def sign2_universe_fn(sign2_list, sign2_universe, sign2_coverage):
        # FIXME this should be performed in a HPC task
        # generate sign2 universes (sign3 specific pre-calculations)
        if not os.path.isfile(sign2_universe):
            sign3.save_sign2_universe(sign2_list, sign2_universe)
        if not os.path.isfile(sign2_coverage):
            sign3.save_sign2_coverage(sign2_list, sign2_coverage)

    sign2_universe_task = PythonCallable(
        name="sign2_universe",
        python_callable=sign2_universe_fn,
        op_args=[sign2_list, sign2_universe, sign2_coverage])
    pp.add_task(sign2_universe_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 3
    cctype = 'sign3'
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    task = CCFit(args.cc_root, cctype, molset[cctype],
                 datasets=dss,
                 fit_kwargs=fit_kwargs[cctype],
                 sign_kwargs=sign_kwargs[cctype],
                 hpc_kwargs=hpc_kwargs[cctype])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Create sym links for exemplary plots
    def create_exemplary_links_fn(cc, sign_ref):
        target_path = os.path.join(args.cc_root, "exemplary")
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        for ds in Dataset.get(exemplary=True):
            signature_path = cc.get_signature_path(sign_ref, "full", ds.code)
            source_path = signature_path[:-6]
            target_dir = os.path.join(target_path, ds.code[:1])
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
            if not os.path.exists(os.path.join(target_dir, ds.code[:2])):
                os.symlink(source_path, os.path.join(target_dir, ds.code[:2]))

    links_task = PythonCallable(
        name="exemplary_links",
        python_callable=create_exemplary_links_fn,
        op_args=[cc, 'sign1'])
    pp.add_task(links_task)
    # END TASK
    #############################################

    #############################################
    # RUN the pipeline!
    if not args.dry_run:
        pp.run()
    # END PIPELINE
    #############################################


if __name__ == '__main__':
    # parse arguments
    args = pipeline_parser().parse_args(sys.argv[1:])
    main(args)
