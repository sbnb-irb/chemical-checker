"""Pipeline to generate a minimal CC of 10k compounds.

Compounds are chosen randomly (see script in package/test/data/
generate_preprocess.py). We skip the preprocessing step and start
directly fitting sign0 on a subset of the preprocessed data.
"""
import os
import sys
import logging
import argparse

from chemicalchecker.util import logged
from chemicalchecker import ChemicalChecker
from chemicalchecker.database import Dataset
from chemicalchecker.core.sign3 import sign3
from chemicalchecker.util.pipeline import PythonCallable
from chemicalchecker.util.pipeline import Pipeline, CCFit


def pipeline_parser():
    """Parse pipeline arguments."""
    description = 'Run the CC Mini pipeline.'
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
        'preprocess_path', type=str,
        help='Directory where the preprocessed H5 are stored '
        '(e.g. `/aloy/home/mbertoni/code/chemical_checker/package/tests/data/preprocess`)')
    parser.add_argument(
        '-r', '--reference_cc', type=str, required=False,
        help='Root dir of the CC instance to use as reference '
        '(i.e. triplet sampling in sign0).')
    parser.add_argument(
        '-c', '--config', type=str, required=False,
        help='Config file to be used. If not specified CC_CONFIG enviroment'
        ' variable is used.')
    return parser


# pipeline_dir = '/aloy/scratch/mbertoni/pipelines/miniCC'
# cc_root = '/aloy/web_checker/package_cc/miniCC'
# preprocess_path = '/aloy/home/mbertoni/code/chemical_checker/package/tests/data/preprocess/'

@logged(logging.getLogger("[ PIPELINE %s ]" % os.path.basename(__file__)))
def main(args):
    # print arguments
    for arg in vars(args):
        main._log.info('[ ARGS ] {:<25s}: {}'.format(arg, getattr(args, arg)))

    # initialize Pipeline
    pp = Pipeline(pipeline_path=args.pipeline_dir, keep_jobs=True)

    fit_order = ['sign0', 'sign1', 'neig1', 'sign2', 'sign3']

    # HPC parameters
    hpc_kwargs = {
        'sign0': {'cpu': 2},
        'sign1': {'cpu': 4},
        'sign2': {'cpu': 8},
        'sign3': {'cpu': 5},
        'neig1': {'cpu': 2},
    }

    # on which signature molset to call the fit?
    molset = {
        'sign0': 'full',
        'sign1': 'full',
        'sign2': 'reference',
        'sign3': 'full',
        'neig1': 'reference',
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
            'adanet': hpc_kwargs['sign2']
        }
        fit_kwargs['sign3'][ds] = {
            'sign2_list': sign2_list,
            'sign2_universe': sign2_universe,
            'sign2_coverage': sign2_coverage,
            'sign0': mfp,
        }
        sign_kwargs['sign3'][ds] = {
            'sign2': hpc_kwargs['sign3']
        }

    # special args
    # fitting sign0 of A1.001 requires no sanitize
    fit_kwargs['sign0']['A1.001'].update({'sanitize': False})

    # add all CCFit tasks to pipeline
    for cctype in fit_order:
        task = CCFit(args.cc_root, cctype, molset[cctype],
                     datasets=datasets,
                     fit_kwargs=fit_kwargs[cctype],
                     sign_kwargs=sign_kwargs[cctype],
                     hpc_kwargs=hpc_kwargs[cctype])
        pp.add_task(task)

    # define special task to cache sign2 universe
    def sign2_universe_fn(sign2_list, sign2_universe, sign2_coverage):
        # FIXME this should be performed in a HPC task
        # generate sign2 universes (sign3 specific pre-calculations)
        if not os.path.isfile(sign2_universe):
            sign3.save_sign2_universe(sign2_list, sign2_universe)
        if not os.path.isfile(sign2_coverage):
            sign3.save_sign2_coverage(sign2_list, sign2_coverage)

    # add function task just before sign3
    sign2_universe_task = PythonCallable(
        name="sign2_universe",
        python_callable=sign2_universe_fn,
        op_args=[sign2_list, sign2_universe, sign2_coverage])
    pp.insert_task(-1, sign2_universe_task)

    # run the pipeline
    pp.run()


if __name__ == '__main__':
    # parse arguments
    args = pipeline_parser().parse_args(sys.argv[1:])
    main(args)
