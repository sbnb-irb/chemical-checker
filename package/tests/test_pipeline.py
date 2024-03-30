import os
import pytest
import unittest
import functools

import sys
sys.path.insert(0, '/aloy/home/ymartins/Documents/cc_update/chemical_checker/package/' )
from chemicalchecker.util import Config


def skip_if_import_exception(function):
    """Assist in skipping tests failing because of missing dependencies."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as err:
            pytest.skip(str(err))
    return wrapper


class TestPipeline(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        #test_dir = '/aloy/home/oguitart/temps'
        self.data_dir = os.path.join(test_dir, 'data')
        self.pipeline_dir = os.path.join(self.data_dir, 'pipeline_test')
        if os.path.isdir(self.pipeline_dir):
            os.system("chmod -R 777 " + self.pipeline_dir)
            os.system("rm -rf " + self.pipeline_dir)
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    def tearDown(self):
        if os.path.isdir(self.pipeline_dir):
            os.system("chmod -R 777 " + self.pipeline_dir)
            os.system("rm -rf " + self.pipeline_dir)
        print('exited')

    @skip_if_import_exception
    def test_pipeline_fit_sign0(self):
        import faiss  # will be required by jobs
        from chemicalchecker.util.pipeline import Pipeline, CCFit

        pipeline_dir = self.pipeline_dir
        data_file = os.path.join(self.data_dir, 'E1_preprocess.h5')
        cc_root = os.path.join(pipeline_dir, 'cc')

        pp = Pipeline(pipeline_path=pipeline_dir, config=Config())
        self.assertTrue(os.path.isdir(pp.readydir))

        # SIGN 0
        s0_fit_kwargs = {
            "E1.001": {
                'key_type': 'inchikey',
                'data_file': data_file,
                'do_triplets': False,
                'validations': False
            }
        }
        s0_task = CCFit(cc_root, 'sign0', 'full',
                        datasets=['E1.001'], fit_kwargs=s0_fit_kwargs)
        pp.add_task(s0_task)
        pp.run()

        sign0_full_file = os.path.join(
            cc_root, 'full/E/E1/E1.001/sign0/sign0.h5')
        self.assertTrue(os.path.isfile(sign0_full_file))
        sign0_ref_file = os.path.join(
            cc_root, 'reference/E/E1/E1.001/sign0/sign0.h5')
        self.assertTrue(os.path.isfile(sign0_ref_file))

        # SIGN 1
        s1_fit_kwargs = {
            "E1.001": {
                'metric_learning': False,
                'validations': False
            }
        }
        s1_task = CCFit(cc_root, 'sign1', 'full',
                        datasets=['E1.001'], fit_kwargs=s1_fit_kwargs)
        pp.add_task(s1_task)
        pp.run()

        sign1_full_file = os.path.join(
            cc_root, 'full/E/E1/E1.001/sign1/sign1.h5')
        self.assertTrue(os.path.isfile(sign1_full_file))
        sign1_ref_file = os.path.join(
            cc_root, 'reference/E/E1/E1.001/sign1/sign1.h5')
        self.assertTrue(os.path.isfile(sign1_ref_file))

        # NEIG 1
        s1_neig_task = CCFit(cc_root, 'neig1', 'reference',
                             datasets=['E1.001'])
        pp.add_task(s1_neig_task)
        pp.run()
        neig1_ref_file = os.path.join(
            cc_root, 'reference/E/E1/E1.001/neig1/neig.h5')
        self.assertTrue(os.path.isfile(neig1_ref_file))

        s2_fit_kwargs = {
            "E1.001": {
                'validations': False,
                'oos_predictor': False
            }
        }
        s2_task = CCFit(cc_root, 'sign2', 'reference',
                        datasets=['E1.001'], fit_kwargs=s2_fit_kwargs)
        pp.add_task(s2_task)
        pp.run()

        # SIGN 2
        sign2_ref_file = os.path.join(
            cc_root, 'reference/E/E1/E1.001/sign2/sign2.h5')
        self.assertTrue(os.path.isfile(sign2_ref_file))

