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

**N.B.**
Once the jobs for a task are completed we can evaluate task
performances (i.e. `cpu` and `memory` parameters were ok?) with:

`ssh pac qacct -j JOBID|egrep 'taskid|maxvmem|ru_wallclock|cpu'`

* 'taskid' the qacct give result aggregated, this help breaking down results.
* 'maxvmem' tell us about max RAM used (tweak `memory` parameter).
* 'ru_wallclock' gives the total time the job took.
* 'cpu' gives the cpu times consumed.

the ratio 'cpu/ru_wallclock' should be as close as possible to the
requested cpus (tweak `cpu` parameter).
"""
import os
import sys
import json
import logging
import argparse
import tempfile
from update_resources import generate_chembl_files

from chemicalchecker import ChemicalChecker
from chemicalchecker.core import Validation
from chemicalchecker.core.sign3 import sign3
from chemicalchecker.util import Config, HPC, logged
from chemicalchecker.database import Dataset, Datasource
from chemicalchecker.database import Molrepo, Calcdata
from chemicalchecker.util.pipeline import Pipeline, PythonCallable
from chemicalchecker.util.pipeline import CCFit, CCPredict
from update_resources.create_database import create_db_dataset
from chemicalchecker.core.diagnostics import Diagnosis

# this is needed to export signaturizers at the end
sys.path.insert(0, '/aloy/home/mbertoni/code/signaturizer')
from signaturizer.exporter import export_batch


def pipeline_parser():
    """Parse pipeline arguments."""
    description = 'Run the full CC update pipeline.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'cc_root', type=str,
        help='Directory where the new CC instance will be generated '
        '(e.g. `/aloy/web_checker/package_cc/2020_01`)')
    parser.add_argument(
        'pipeline_dir', type=str,
        help='Directory where the pipeline will run '
        '(e.g. `/aloy/scratch/mbertoni/pipelines/cc_update_2020_01`)')
    parser.add_argument(
        '-r', '--reference_cc', type=str, default="", required=False,
        help='Root dir of the CC instance to use as reference '
        '(i.e. triplet sampling in sign0).')
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

    fit_order = ['sign0', 'sign1', 'clus1', 'proj1', 'neig1',
                 'sign2', 'clus2', 'proj2', 'neig2', 'sign3',
                 'clus3', 'proj3', 'neig3']

    data_calculators = [
        'morgan_fp_r2_2048',
        'e3fp_3conf_1024',
        'murcko_1024_cframe_1024',
        'maccs_keys_166',
        'general_physchem_properties',
        'chembl_target_predictions_v23_10um'
    ]

    validation_sets = ['moa', 'atc']

    # HPC parameters and resources observations
    hpc_kwargs = {
        # sign0
        #  is using the most memory with ~GB
        #  is the one taking longer with ~s (h)
        'sign0': {'memory': 44, 'cpu': 22},
        # sign1 (w/o metric_learning) does not parallelize
        # B4 is using the most memory with ~12GB
        # C5 is the one taking longer with ~59000s (16.5h)
        'sign1': {'memory': 16, 'cpu': 4},
        # sign2 paralelize well and requires memory
        # A2 is using the most memory with ~42GB
        # A1 is the one taking longer with ~158000s (43h)
        'sign2': {'memory': 40, 'cpu': 16},
        # sign3
        # A1 is using the most memory with ~59GB
        # A1 is the one taking longer with ~186000s (52h)
        'sign3': {'mem_by_core': 7, 'memory': 56, 'cpu': 8},
        # predicting sign3 based on signaturizer
        'sign3_pred': {'cpu': 4},
        # neig1 paralelize very well and require very few memory
        # A2 is the one taking longer with ~9100s (2.5h)
        'neig1': {'memory': 3, 'cpu': 16},
        'neig2': {'memory': 3, 'cpu': 16},
        'neig3': {'memory': 3, 'cpu': 16},
        # clus1 does not paralelize very well and require memory
        # A1 is using the most memory ~28Gb
        # A1 is the one taking longer with ~10700s (3h)
        'clus1': {'memory': 30, 'cpu': 4},
        'clus2': {'memory': 30, 'cpu': 4},
        'clus3': {'memory': 30, 'cpu': 4},
        # proj1 paralelize very well and require few memory
        # A1 is using the most memory ~13Gb
        # A1 is the one taking longer with ~4500s (1.5h)
        'proj1': {'memory': 20, 'cpu': 16},
        'proj2': {'memory': 20, 'cpu': 16},
        'proj3': {'memory': 20, 'cpu': 16}
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

    # initialize parameter holders (two dict, one for init and one for fit)
    datasets = [ds.code for ds in Dataset.get(exemplary=True)]
    sign_kwargs = {}
    fit_kwargs = {}
    for cctype in fit_order:
        fit_kwargs[cctype] = {}
        sign_kwargs[cctype] = {}

    # GENERAL FIT/INIT PARAMETERS
    cc = ChemicalChecker(args.cc_root)
    sign2_universe = os.path.join(pp.cachedir, "sign2_universe_stacked.h5")
    sign2_coverage = os.path.join(pp.cachedir, "sign2_universe_coverage.h5")
    sign2_list = [cc.get_signature('sign2', 'full', ds)
                  for ds in cc.datasets_exemplary()]
    mfp = cc.get_signature('sign0', 'full', 'A1.001').data_path
    for ds in datasets:
        fit_kwargs['sign0'][ds] = {
            'key_type': 'inchikey',
            'do_triplets': False,
            'sanitize': True,
            'validations': True,
            'diagnostics': False,
            'sanitizer_kwargs': {'max_features': 10000,'chunk_size': 10000}
        }
        fit_kwargs['sign1'][ds] = {
            'metric_learning': False,
            'diagnostics': False
        }
        fit_kwargs['sign2'][ds] = {
            'validations': True,
            'diagnostics': False,
            'node2vec_kwargs': {'cpu': 4},
            'adanet_kwargs': {'cpu': hpc_kwargs['sign2']['cpu']}
        }
        fit_kwargs['sign3'][ds] = {
            'sign2_list': sign2_list,
            'sign2_universe': sign2_universe,
            'sign2_coverage': sign2_coverage,
            'sign0': mfp,
            'diagnostics': False
        }
        sign_kwargs['sign3'][ds] = {
            'sign2': {'cpu': hpc_kwargs['sign3']['cpu']}
        }
        sign_kwargs['neig1'][ds] = {
            'cpu': hpc_kwargs['neig1']['cpu']
        }
        sign_kwargs['neig2'][ds] = {
            'cpu': hpc_kwargs['neig2']['cpu']
        }
        sign_kwargs['clus1'][ds] = {
            'cpu': hpc_kwargs['clus1']['cpu'],
            'general_params': {'balance': 1.5}
        }
        sign_kwargs['clus2'][ds] = {
            'cpu': hpc_kwargs['clus2']['cpu']
        }
        sign_kwargs['proj1'][ds] = {
            'cpu': hpc_kwargs['proj1']['cpu']
        }
        sign_kwargs['proj2'][ds] = {
            'cpu': hpc_kwargs['proj2']['cpu']
        }

    # DATASET SPECIFIC FIT/INIT PARAMETERS
    # we want to keep exactly 2048 features (Morgan fingerprint) for A1
    fit_kwargs['sign0']['A1.001'] = {'sanitize': False}

    # TASK: Create new CC DB
    def create_db():
        # Create a new database for the current CC update
        create_db_dataset()

    creating_db_task = PythonCallable(name="creating_db",
                                    python_callable=create_db)
    pp.add_task(creating_db_task)
    # END TASK
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
    molrepos_task = PythonCallable(name="molrepos",
                                   python_callable=Molrepo.molrepo_hpc,
                                   op_args=[pp.tmpdir],
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
    # TASK: Calculate diagnostics plots of sign0 for derived spaces: the reference cc version is provided
    # by the 'reference_cc' input parameter (only for sign0 case, for other signatures 
    # the reference is args.cc_root itself)
    cctype = 'sign0'
    task = PythonCallable(name="diagnostics_sign0_BCDE",
                         python_callable=Diagnosis.diagnostics_hpc,
                         op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.reference_cc])
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
    for data_calc in data_calculators:
        calc_data_task = PythonCallable(
            name="calc_data_" + data_calc,
            python_callable=calculate_data_fn,
            op_args=[data_calc, pp.tmpdir, universe])
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
    # TASK: Calculate diagnostics plots of sign0 for A* spaces: the reference cc version is provided
    # by the 'reference_cc' input parameter (only for sign0 case, for other signatures 
    # the reference is args.cc_root itself)
    cctype = 'sign0'
    task = PythonCallable(name="diagnostics_sign0_A",
                         python_callable=Diagnosis.diagnostics_hpc,
                         op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.reference_cc])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 1-2 also clus, proj and neig
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    s0_idx = fit_order.index('sign0') + 1
    s3_idx = fit_order.index('sign3')
    for cctype in fit_order[s0_idx:s3_idx]:
        task = CCFit(args.cc_root, cctype, molset[cctype],
                     datasets=dss,
                     fit_kwargs=fit_kwargs[cctype],
                     sign_kwargs=sign_kwargs[cctype],
                     hpc_kwargs=hpc_kwargs[cctype])
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: diagonistc plots for sign1-sign2
    cctypes = ['sign1', 'sign2']
    for cctype in cctypes:
        task = PythonCallable(name="diagnostics_" + cctype,
                            python_callable=Diagnosis.diagnostics_hpc,
                            op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.cc_root])
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
    # TASK: Calculate diagnostics plots of sign3 for all spaces
    cctype = 'sign3'
    task = PythonCallable(name="diagnostics_" + cctype,
                         python_callable=Diagnosis.diagnostics_hpc,
                         op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.reference_cc])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate clus3, proj3 and neig3
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    s3_idx = fit_order.index('sign3') + 1
    for cctype in fit_order[s3_idx:]:
        task = CCFit(args.cc_root, cctype, molset[cctype],
                     datasets=dss,
                     fit_kwargs=fit_kwargs[cctype],
                     sign_kwargs=sign_kwargs[cctype],
                     hpc_kwargs=hpc_kwargs[cctype])
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Predict signatures 3 (signaturizer like)
    cctype = 'sign3'
    task_name = 'sign3_pred'
    # get inchies for cc universe
    uni_inchi_path = os.path.join(pp.cachedir, 'universe_inchi.json')
    if os.path.isfile(uni_inchi_path):
        universe_inchi = json.load(open(uni_inchi_path, 'r'))
    else:
        universe_inchi = cc.get_universe_inchi()
        json.dump(universe_inchi, open(uni_inchi_path, 'w'))
    predict_args = dict()
    predict_args[task_name] = dict()
    predict_kwargs = dict()
    predict_kwargs[task_name] = dict()
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    for ds in dss:
        s3_dest = cc.get_signature('sign3', 'predicted', ds)
        predict_args[task_name][ds] = [universe_inchi, s3_dest.data_path]
        predict_kwargs[task_name][ds] = {'keytype': 'INCHI',
                                         'keys': cc.universe}
    task = CCPredict(args.cc_root, cctype, molset[cctype],
                     name=task_name,
                     datasets=dss,
                     predict_fn='predict_from_string',
                     predict_args=predict_args[task_name],
                     predict_kwargs=predict_kwargs[task_name],
                     hpc_kwargs=hpc_kwargs[task_name])
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
    # TASK: Export signaturizers
    def export_signaturizers(cc_root, path='/aloy/web_checker/signaturizers/'):
        cc = ChemicalChecker(args.cc_root)
        sign_path = os.path.join(path, cc.name)
        if not os.path.isdir(sign_path):
            os.mkdir(sign_path)
        export_batch(cc, sign_path)

    export_task = PythonCallable(name="export_signaturizers",
                                 python_callable=export_signaturizers,
                                 op_args=[args.cc_root])
    pp.add_task(export_task)
    # END TASK
    #############################################

    #############################################
    # RUN the pipeline!
    main._log.info('TASK SEQUENCE: %s' % ', '.join([t.name for t in pp.tasks]))
    if not args.dry_run:
        pp.run()
    # END PIPELINE
    #############################################


if __name__ == '__main__':
    # parse arguments
    args = pipeline_parser().parse_args(sys.argv[1:])
    main(args)
