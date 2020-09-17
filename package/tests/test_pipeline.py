import os
import pytest
import unittest
import functools


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
            # shutil.rmtree(self.pipeline_dir)

    @skip_if_import_exception
    def test_pipeline_fit_sign0(self):
        from chemicalchecker.util.pipeline import Pipeline, CCPredict, CCFit

        pipeline_dir = self.pipeline_dir
        data_file = os.path.join(self.data_dir, 'E1_preprocess.h5')
        cc_root = os.path.join(pipeline_dir, 'cc')

        pp = Pipeline(pipeline_path=pipeline_dir)
        self.assertTrue(os.path.isdir(pp.readydir))

        s0_params = {
            "output_path": pipeline_dir,
            'cc_old_path': '/aloy/web_checker/package_cc/paper',
            "datasets": ["E1.001"],
            "ds_params": {
                "E1.001": {
                    'key_type': 'inchikey',
                    'data_file': data_file,
                    'do_triplets': False,
                    'validations': False
                }
            }
        }

        s0_task = CCFit(cc_root, 'sign0', **s0_params)
        pp.add_task(s0_task)
        pp.run()

        sign0_file = os.path.join(cc_root, 'full/E/E1/E1.001/sign0/sign0.h5')
        self.assertTrue(os.path.isfile(sign0_file))
        """

        s0_pred_task = CCPredict(cc_type='sign0', **s0_params)
        pp.add_task(s0_pred_task)

        s0_pred_task = CCPredict(cc_type='sign0', **s0_params)
        pp.add_task(s0_pred_task)

        s1_params = {"output_path": pipeline_dir, 'CC_ROOT': CC_ROOT,
                     "datasets": ["E1.001"]}

        s1_pred_task = CCPredict(cc_type='sign1', **s1_params)
        pp.add_task(s1_pred_task)

        s2_params = {"output_path": pipeline_dir, 'CC_ROOT': CC_ROOT,
                     "datasets": ["E1.001"]}

        s2_pred_task = CCPredict(cc_type='sign2', **s2_params)
        pp.add_task(s2_pred_task)

        s0_pred_task.mark_ready()
        self.assertTrue(s0_pred_task.is_ready())

        s1_pred_task.mark_ready()
        self.assertTrue(s1_pred_task.is_ready())

        s2_pred_task.mark_ready()
        self.assertTrue(s2_pred_task.is_ready())
        """
        # This needs to be uncommented when the signatures predict work

        # s2_pred_task.clean()
        # self.assertFalse(s2_pred_task.is_ready())
