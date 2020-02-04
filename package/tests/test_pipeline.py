import os
import shutil
import pytest
import unittest
import functools

from chemicalchecker.util.pipeline import Pipeline, CCPredict


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
        self.pp_dir = os.path.join(self.data_dir, 'package_test_predict')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    def tearDown(self):
        if os.path.isdir(self.pp_dir):
            os.system("rm -rf " + self.pp_dir)
            #shutil.rmtree(self.pp_dir)

    @skip_if_import_exception
    def test_pipeline(self):
        self.pp = Pipeline(pipeline_path=self.pp_dir)
        self.assertTrue(os.path.isdir(self.pp.readydir))
        output_path = self.pp_dir
        s0_params = {"output_path": output_path,
                     "datasets_input_files": {"B4.001": '/aloy/scratch/oguitart/tmp/entry_profile.tsv'}}

        s0_pred_task = CCPredict(cc_type='sign0', **s0_params)
        self.pp.add_task(s0_pred_task)

        s1_params = {"output_path": output_path,
                     "datasets_input_files": ["B4.001"]}

        s1_pred_task = CCPredict(cc_type='sign1', **s1_params)
        self.pp.add_task(s1_pred_task)

        s2_params = {"output_path": output_path,
                     "datasets_input_files": ["B4.001"]}

        s2_pred_task = CCPredict(cc_type='sign2', **s2_params)
        self.pp.add_task(s2_pred_task)

        s0_pred_task.mark_ready()
        self.assertTrue(s0_pred_task.is_ready())

        s1_pred_task.mark_ready()
        self.assertTrue(s1_pred_task.is_ready())

        s2_pred_task.mark_ready()
        self.assertTrue(s2_pred_task.is_ready())

        self.pp.run()

        s2_pred_task.clean()
        self.assertFalse(s2_pred_task.is_ready())
