import os
import pytest
import shutil
import unittest
import functools

from chemicalchecker.core.sign2 import sign2
from chemicalchecker.core.sign3 import sign3, subsample


def skip_if_import_exception(function):
    """Assist in skipping tests failing because of missing dependencies."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as err:
            pytest.skip(str(err))
    return wrapper


class TestSign3(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')
        self.sign_dir = os.path.join(self.data_dir, 'sign3', 'sign3')
        if os.path.isdir(self.sign_dir):
            shutil.rmtree(self.sign_dir)

    def tearDown(self):
        if os.path.isdir(self.sign_dir):
            shutil.rmtree(self.sign_dir, ignore_errors=True)

    @skip_if_import_exception
    def test_sign3(self):
        sign2_list = list()
        for ds in ['E1.001', 'E2.001']:
            sign2_dir = os.path.join(self.data_dir, 'sign3', ds)
            sign2_list.append(sign2(sign2_dir, ds))
        s3 = sign3(self.sign_dir, 'E1.001')
        s3.params = dict()
        s3.params['adanet'] = {
            "eval": {
                'extension_step': 3,
                'epoch_per_iteration': 1,
                'adanet_iterations': 2,
                "augmentation": subsample,
            },
            "test": {
                'epoch_per_iteration': 1,
                'adanet_iterations': 1,
                "augmentation": subsample,
            },
            "sign0_eval": {
                'extension_step': 3,
                'epoch_per_iteration': 1,
                'adanet_iterations': 2,
                'augmentation': False,
            },
            "sign0_test": {
                'epoch_per_iteration': 1,
                'adanet_iterations': 1,
                'augmentation': False,
            }
        }
        s3.fit(sign2_list, sign2_list[0])

        self.assertTrue(os.path.isfile(s3.data_path))
        self.assertTrue(os.path.isdir(s3.model_path))
        eval_dir = os.path.join(s3.model_path, 'adanet_eval')
        self.assertTrue(os.path.isdir(eval_dir))
        test_dir = os.path.join(s3.model_path, 'adanet_test')
        self.assertTrue(os.path.isdir(test_dir))
        final_dir = os.path.join(s3.model_path, 'adanet_final')
        self.assertTrue(os.path.isdir(final_dir))
        self.assertEqual(s3.shape[0], 3563)
        self.assertEqual(s3.shape[1], 128)
        ds_corr = list(s3.get_h5_dataset('datasets_correlation'))
        real_ds_corr = [0.98518413, 0.5811032]
        self.assertAlmostEqual(ds_corr[0], real_ds_corr[0], 1)
        self.assertAlmostEqual(ds_corr[1], real_ds_corr[1], 1)
        info_h5 = {'V': (3563, 128),
                   'confidence': (3563,),
                   'consensus': (3563, 128),
                   'datasets_correlation': (2,),
                   'datasets': (2,),
                   'intensity': (3563,),
                   'intensity_norm': (3563,),
                   'keys': (3563,),
                   'pred_correlation': (3563, 3),
                   'stddev': (3563,),
                   'stddev_norm': (3563,),
                   'support': (3563,)}
        self.assertDictEqual(info_h5, s3.info_h5)
