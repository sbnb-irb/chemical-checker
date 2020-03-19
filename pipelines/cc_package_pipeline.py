import sys
import os
import numpy as np
import csv
import tempfile
import h5py
from chemicalchecker import ChemicalChecker
from chemicalchecker.database import Datasource
from chemicalchecker.util import HPC
from chemicalchecker.database import Molrepo
from chemicalchecker.core import Validation
from chemicalchecker.database import Calcdata
from chemicalchecker.database import Dataset
from chemicalchecker.util.pipeline import Pipeline, PythonCallable, CCFit, CCLongShort, CCSmileConverter

os.environ['CC_CONFIG'] = '/aloy/home/oguitart/projects/source/chemical_checker/pipelines/dream_ctd2_config.json'


CC_PATH = "/aloy/web_checker/package_cc/dream_ctd2/"

data_calculators = ['morgan_fp_r2_2048', 'e3fp_3conf_1024', 'murcko_1024_cframe_1024',
                    'maccs_keys_166', 'general_physchem_properties', 'chembl_target_predictions_v23_10um']

validation_sets = ['moa', 'atc']

pp = Pipeline(pipeline_path="/aloy/scratch/oguitart/package_cc")


def downloads(tmpdir):

    job_path = tempfile.mkdtemp(
        prefix='jobs_download_', dir=tmpdir)
    # start download jobs (one per Datasource), job will wait until
    # finished
    job = Datasource.download_hpc(job_path, only_essential=True)

    if job.status() == HPC.ERROR:
        print(
            "There are errors in some of the downloads jobs")

    # check if the downloads are really done
    if not Datasource.test_all_downloaded(only_essential=True):
        print(
            "Something went WRONG while DOWNLOAD, should retry")
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
                print("ERROR: Datasource %s not available" % ds)

        raise Exception('Not all datasources were downloaded correctly')


def calculate_data(type_data, tmpdir, iks):

    print("Calculating data for " + type_data)

    job_path = tempfile.mkdtemp(
        prefix='jobs_molprop_' + type_data + "_", dir=tmpdir)

    calculator = Calcdata(type_data)

    # This method sends the job and waits for the job to finish
    calculator.calcdata_hpc(job_path, list(final_ik_inchi))
    missing = len(calculator.get_missing_from_set(iks))
    if missing > 0:
        raise Exception("Not all molecular properties were calculated. There are " +
                        str(missing) + " missing out of " + str(len(iks)))


def create_val_set(set_name):

    cc = ChemicalChecker(CC_PATH)

    val = Validation(cc.get_validation_path(), set_name)

    try:
        val.run()

    except Exception as ex:
        print(ex)
        raise Exception("Validation set %s not working" % set_name)


def create_exemplary_links(sign_ref):

    all_datasets = Dataset.get()

    cc = ChemicalChecker(CC_PATH)

    dataset_codes = list()
    for ds in all_datasets:
        if not ds.exemplary:
            continue

        dataset_codes.append(str(ds.dataset_code))

    target_path = os.path.join(CC_PATH, "exemplary")

    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    for code in dataset_codes:

        signature_path = cc.get_signature_path(sign_ref, "full", code)

        source_path = signature_path[:-6]

        target_dir = os.path.join(target_path, code[:1])

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        if not os.path.exists(os.path.join(target_dir, code[:2])):
            os.symlink(source_path, os.path.join(target_dir, code[:2]))


##### TASK: Download all datasources #######

downloads_params = {}

downloads_params['python_callable'] = downloads
downloads_params['op_args'] = [pp.tmpdir]

downloads_task = PythonCallable(name="downloads", **downloads_params)

pp.add_task(downloads_task)


##### TASK: Parse molrepos #######

molrepos_params = {}

job_path = tempfile.mkdtemp(
    prefix='jobs_molrepos_', dir=pp.tmpdir)

molrepos_params['python_callable'] = Molrepo.molrepo_hpc
molrepos_params['op_args'] = [job_path]
molrepos_params['op_kwargs'] = {'only_essential': True}

molrepos_task = PythonCallable(name="molrepos", **molrepos_params)

pp.add_task(molrepos_task)


##### TASK: Get inchikey/inchi pairs and calculate data #######

final_ik_inchi = set()
all_molrepos = Molrepo.get()
molrepos_names = set()
for molrepo in all_molrepos:
    molrepos_names.add(molrepo.molrepo_name)

for molrepo in molrepos_names:
    print(molrepo)
    molrepo_ik_inchi = Molrepo.get_fields_by_molrepo_name(
        molrepo, ["inchikey", "inchi"])
    final_ik_inchi.update(molrepo_ik_inchi)

iks_to_calc = set()

for ik in final_ik_inchi:
    iks_to_calc.add(ik[0])

for data_calc in data_calculators:
    calc_data_params = {}

    calc_data_params['python_callable'] = calculate_data
    calc_data_params['op_args'] = [data_calc, pp.tmpdir, iks_to_calc]

    calc_data_task = PythonCallable(
        name="calc_data_" + data_calc, **calc_data_params)

    pp.add_task(calc_data_task)

##### TASK: Generate validation sets #######

for val_set in validation_sets:
    val_set_params = {}

    val_set_params['python_callable'] = create_val_set
    val_set_params['op_args'] = [val_set]

    val_set_task = PythonCallable(name="val_set_" + val_set, **val_set_params)

    pp.add_task(val_set_task)


# TASK: Calculate signatures 0
s0_params = {'cc_old_path': '/aloy/web_checker/package_cc/paper'}
s0_task = CCFit(cc_type='sign0', **s0_params)
pp.add_task(s0_task)


# TASK: Calculate signatures 1
s1_params = {}
s1_params["ds_params"] = {"A1.001": {"num_topics": "1600", "max_freq": "0.8", "multipass": "True"},
                          "A2.001": {"num_topics": "1000", "max_freq": "0.8", "multipass": "True"},
                          "A3.001": {"num_topics": "1500", "max_freq": "0.8", "multipass": "True"},
                          "A4.001": {"num_topics": "70", "max_freq": "0.9", "multipass": "True"},
                          "A5.001": {"discrete": False},
                          "B1.001": {"num_topics": "200"},
                          "B2.001": {"num_topics": "200"},
                          "B3.001": {"num_topics": "500"},
                          "B4.001": {"num_topics": "800"},
                          "B5.001": {"num_topics": "800"},
                          "C1.001": {"num_topics": "600"},
                          "C2.001": {"num_topics": "500"},
                          "C3.001": {"num_topics": "200"},
                          "C4.001": {"num_topics": "500"},
                          "C5.001": {"num_topics": "500"},
                          "D1.001": {"num_topics": "4600"},
                          "D2.001": {"discrete": False},
                          "D3.001": {"num_topics": "800"},
                          "D4.001": {"discrete": False},
                          "D5.001": {"num_topics": "100"},
                          "E1.001": {"num_topics": "250"},
                          "E2.001": {"num_topics": "600"},
                          "E3.001": {"num_topics": "700"},
                          "E4.001": {"num_topics": "800"},
                          "E5.001": {"num_topics": "250"}}

s1_task = CCFit(cc_type='sign1', **s1_params)
pp.add_task(s1_task)

##### TASK: Calculate clustering for signatures 1 #######
c1_params = {}
c1_params['general_params'] = {'balance': 1.5}
c1_task = CCFit(cc_type='clus1', **c1_params)
pp.add_task(c1_task)

##### TASK: Calculate nearest neighbors for signatures 1 #######
n1_params = {}
n1_task = CCFit(cc_type='neig1', **n1_params)
pp.add_task(n1_task)

##### TASK: Calculate projections for signatures 1  ########
p1_params = {}
p1_task = CCFit(cc_type='proj1', **p1_params)
pp.add_task(p1_task)

# TASK: Calculate signatures 2
s2_params = {}
s2_task = CCFit(cc_type='sign2', **s2_params)
pp.add_task(s2_task)

##### TASK: Calculate nearest neighbors for signatures 2 #######
n2_params = {}
n2_task = CCFit(cc_type='neig2', **n2_params)
pp.add_task(n2_task)

##### TASK: Calculate projections for signatures 2  ########
p2_params = {}
p2_task = CCFit(cc_type='proj2', **p2_params)
pp.add_task(p1_task)

# TASK: Calculate signatures 3
s3_params = {}
s3_task = CCFit(cc_type='sign3', **s3_params)
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
links_params = {}
links_params['python_callable'] = create_exemplary_links
links_params['op_args'] = ['sign1']

links_task = PythonCallable(name="exemplary_links", **links_params)

pp.add_task(links_task)

pp.run()
