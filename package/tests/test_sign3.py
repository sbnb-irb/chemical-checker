import os
import pytest
import shutil
import unittest
import functools

import sys
sys.path.insert(0, '/aloy/home/ymartins/Documents/cc_update/chemical_checker/package/' )

from chemicalchecker.core.sign3 import subsample
from chemicalchecker import ChemicalChecker
ChemicalChecker.set_verbosity('DEBUG')


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
        self.cc_local = os.path.join(self.data_dir, 'sign3', 'cc_local')
        if os.path.isdir(self.cc_local):
            shutil.rmtree(self.cc_local)

    def tearDown(self):
        if os.path.isdir(self.cc_local):
            shutil.rmtree(self.cc_local, ignore_errors=True)

    @skip_if_import_exception
    def test_sign3(self):
        data_path = os.path.join(self.data_dir, 'sign3')
        cc_local = ChemicalChecker(self.cc_local, custom_data_path=data_path)

        sign2_list = list()
        for ds in ['A1.001', 'E1.001', 'E2.001']:
            sign2_list.append(cc_local.signature(ds, 'sign2'))
        sign1_self = cc_local.signature('E1.001', 'sign1')
        sign2_self = cc_local.signature('E1.001', 'sign2')

        s3 = cc_local.signature('E1.001', 'sign3')
        siamese_args = {
            'epochs': 1,
            'cpu': 1,
            'layers': ['Dense', 'Dense'],
            'layers_sizes': [1024, 128],
            'activations': ['selu', 'tanh'],
            'dropouts': [0.2, None],
            'learning_rate': 1e-3,
            'batch_size': 128,
            'patience': 200,
            'loss_func': 'only_self_loss',
            'margin': 1.0,
            'alpha': 1.0,
            'num_triplets': 1000,
            't_per': 0.01,
            'onlyself_notself': True,
            'limit_mols': 1000,
            'augment_fn': subsample,
            'augment_kwargs': {
                'dataset': ['E1.001'],
                'p_self': 0.1
            },
        }
        s3.params['sign2'] = siamese_args
        s3.fit(sign2_list=sign2_list, sign2_self=sign2_self, 
               sign1_self=sign1_self, validations=False,
               complete_universe=False)
        # s3.fit(sign2_list=sign2_list, sign2_self=sign2_self, 
        #        sign1_self=sign1_self, validations=False,
        #        complete_universe='custom', calc_ds_idx=[0],
        #        calc_ds_names=['A1.001'])

        self.assertTrue(os.path.isfile(s3.data_path))
        self.assertTrue(os.path.isdir(s3.model_path))
        eval_dir = os.path.join(s3.model_path, 'siamese_eval')
        self.assertTrue(os.path.isdir(eval_dir))
        final_dir = os.path.join(s3.model_path, 'siamese_final')
        self.assertTrue(os.path.isdir(final_dir))
        self.assertEqual(s3.shape[0], 3543)
        self.assertEqual(s3.shape[1], 128)
