"""Pipeline to generate a full CC update.
"""
import os
import sys
import logging
import argparse
import tempfile

from chemicalchecker.util import logged
from chemicalchecker import ChemicalChecker
from chemicalchecker.core import Validation
from chemicalchecker.database import Molrepo
from chemicalchecker.database import Dataset
from chemicalchecker.database import Calcdata
from chemicalchecker.database import Datasource
from chemicalchecker.util.pipeline import Pipeline, PythonCallable, CCFit
from chemicalchecker.util.pipeline import CCLongShort, CCSmileConverter
from chemicalchecker.util import Config
from chemicalchecker.util import HPC

from update_resources.generate_chembl_files import generate_chembl_files


def pipeline_parser():
    """Parse pipeline arguments."""
    description = 'Run the full CC update pipeline.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'cc_root', type=str, required=True,
        help='Directory where the new CC instance will be generated '
        '(e.g. `/aloy/web_checker/package_cc/miniCC`)')
    parser.add_argument(
        'pipeline_dir', type=str, required=True,
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
    return parser


@logged(logging.getLogger("[ PIPELINE %s ]" % os.path.basename(__file__)))
def main(args):
    # print arguments
    for arg in vars(args):
        main._log.info('[ ARGS ] {:<25s}: {}'.format(arg, getattr(args, arg)))
    return
    # initialize Pipeline
    pp = Pipeline(pipeline_path=args.pipeline_dir, keep_jobs=True)

    fit_order = ['sign0', 'sign1', 'neig1', 'sign2', 'sign3']

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
    sign2_universe = os.path.join(pp.cache, "universe_full")
    sign2_coverage = os.path.join(pp.cache, "coverage_full")
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
        sign_kwargs['neig3'][ds] = {
            'cpu': 15
        }
        sign_kwargs['clus1'][ds] = {
            'cpu': 10
        }
        sign_kwargs['clus2'][ds] = {
            'cpu': 10
        }
        sign_kwargs['clus3'][ds] = {
            'cpu': 10
        }
        sign_kwargs['proj1'][ds] = {
            'cpu': 10
        }
        sign_kwargs['proj2'][ds] = {
            'cpu': 10
        }
        sign_kwargs['proj3'][ds] = {
            'cpu': 10
        }

    # TASK: Download all datasources
    def download_fn(tmpdir):
        # Generate the Chembl files drugtargets and drugindications via
        # Chembl Python API in /aloy/web_checker/repo_data
        generate_chembl_files()

        job_path = tempfile.mkdtemp(prefix='jobs_download_', dir=tmpdir)
        # start download jobs (one per Datasource), job will wait until
        # finished
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

    # TASK: Parse molrepos
    job_path = tempfile.mkdtemp(prefix='jobs_molrepos_', dir=pp.tmpdir)
    molrepos_task = PythonCallable(name="molrepos",
                                   python_callable=Molrepo.molrepo_hpc,
                                   op_args=[job_path],
                                   op_kwargs={'only_essential': True})
    pp.add_task(molrepos_task)

    def calculate_data(type_data, tmpdir, iks):
        print("Calculating data for " + type_data)
        job_path = tempfile.mkdtemp(
            prefix='jobs_molprop_' + type_data + "_", dir=tmpdir)
        calculator = Calcdata(type_data)
        # This method sends the job and waits for the job to finish
        calculator.calcdata_hpc(job_path, list(final_ik_inchi))
        missing = len(calculator.get_missing_from_set(iks))
        if missing > 0:
            raise Exception(
                "Not all molecular properties were calculated. There are " +
                str(missing) + " missing out of " + str(len(iks)))

    def create_val_set(set_name):
        print("Creating validation set for " + set_name)
        val = Validation(Config().PATH.validation_path, set_name)
        try:
            val.run()
        except Exception as ex:
            print(ex)
            raise Exception("Validation set '%s' not working" % set_name)

    def create_exemplary_links(sign_ref):
        target_path = os.path.join(args.cc_root, "exemplary")
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        cc = ChemicalChecker(args.cc_root)
        for ds in Dataset.get(exemplary=True):
            signature_path = cc.get_signature_path(sign_ref, "full", ds.code)
            source_path = signature_path[:-6]
            target_dir = os.path.join(target_path, ds.code[:1])
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
            if not os.path.exists(os.path.join(target_dir, ds.code[:2])):
                os.symlink(source_path, os.path.join(target_dir, ds.code[:2]))

    # TASK: Get inchikey/inchi pairs and calculate data
    if not DEBUG:
        molrepos_names = {mr.molrepo_name for mr in Molrepo.get()}

        print("Fetching molecule repositories from the sql database:")
        final_ik_inchi = set()
        for molrepo in molrepos_names:
            print(molrepo)
            molrepo_ik_inchi = Molrepo.get_fields_by_molrepo_name(
                molrepo, ["inchikey", "inchi"])
            final_ik_inchi.update(molrepo_ik_inchi)
        iks_to_calc = {ik[0] for ik in final_ik_inchi}

        # TASK: Calculate data
        for data_calc in data_calculators:
            print("--> calc_data_" + data_calc)
            calc_data_params = {
                'python_callable': calculate_data,
                'op_args': [data_calc, pp.tmpdir, iks_to_calc]
            }
            calc_data_task = PythonCallable(
                name="calc_data_" + data_calc, **calc_data_params)
            pp.add_task(calc_data_task)

        # TASK: Generate validation sets
        for val_set in validation_sets:
            val_set_params = {
                'python_callable': create_val_set,
                'op_args': [val_set]
            }
            val_set_task = PythonCallable(
                name="val_set_" + val_set, **val_set_params)
            pp.add_task(val_set_task)

    # TASK: Calculate signatures 0
    s0_params = {'cc_ref_root': args.cc_ref}
    s0_task = CCFit(args.cc_root, 'sign0', **s0_params)
    pp.add_task(s0_task)

    if DEBUG:
        pp = Pipeline(pipeline_path="/aloy/scratch/sbnb-adm/package_cc")

    # TASK: Calculate signatures 1
    s1_params = {}
    s1_task = CCFit(args.cc_root, 'sign1', **s1_params)
    pp.add_task(s1_task)

    # TASK: Calculate clustering for signatures 1
    c1_params = {
        'general_params': {'balance': 1.5}
    }
    c1_task = CCFit(args.cc_root, 'clus1', **c1_params)
    pp.add_task(c1_task)

    # TASK: Calculate nearest neighbors for signatures 1
    n1_params = {}
    n1_task = CCFit(args.cc_root, 'neig1', **n1_params)
    pp.add_task(n1_task)

    # TASK: Calculate projections for signatures 1
    p1_params = {}
    p1_task = CCFit(args.cc_root, 'proj1', **p1_params)
    pp.add_task(p1_task)

    # TASK: Calculate signatures 2
    s2_params = {}
    s2_task = CCFit(args.cc_root, 'sign2', **s2_params)
    pp.add_task(s2_task)

    # TASK: Calculate nearest neighbors for signatures 2
    n2_params = {}
    n2_task = CCFit(args.cc_root, 'neig2', **n2_params)
    pp.add_task(n2_task)

    # TASK: Calculate projections for signatures 2
    p2_params = {}
    p2_task = CCFit(args.cc_root, 'proj2', **p2_params)
    pp.add_task(p2_task)

    pp.run()
    print("DONE, Calculate sign 0,1,2, nearest neighbours and projections")
    sys.exit(0)

    # TASK: Calculate signatures 3
    s3_params = {}
    s3_task = CCFit(args.cc_root, 'sign3', **s3_params)
    pp.add_task(s3_task)

    # TASK: Calculate consensus signature 3
    s3_short_params = {}
    s3_short_task = CCLongShort(cc_type='sign3', **s3_short_params)
    pp.add_task(s3_short_task)

    # TASK: Calculate smiles to signature 3
    s3_smile_params = {}
    s3_smile_task = CCSmileConverter(cc_type='sign3', **s3_smile_params)
    pp.add_task(s3_smile_task)

    # TASK: Create sym links for exemplary plots
    links_params = {
        'python_callable': create_exemplary_links,
        'op_args': ['sign1']
    }
    links_task = PythonCallable(name="exemplary_links", **links_params)
    pp.add_task(links_task)
    pp.run()


if __name__ == '__main__':
    # parse arguments
    args = pipeline_parser().parse_args(sys.argv[1:])
    main(args)
