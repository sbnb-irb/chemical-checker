import os
import pytest
import shutil
import unittest
import functools

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


class TestSign4(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')
        self.cc_local = os.path.join(self.data_dir, 'sign4', 'cc_local')
        if os.path.isdir(self.cc_local):
            shutil.rmtree(self.cc_local)

    def tearDown(self):
        if os.path.isdir(self.cc_local):
            shutil.rmtree(self.cc_local, ignore_errors=True)
            pass

    @skip_if_import_exception
    def test_sign4(self):
        data_path = os.path.join(self.data_dir, 'sign4')
        cc_local = ChemicalChecker(self.cc_local, custom_data_path=data_path)

        sign0 = cc_local.signature('A1.001', 'sign0')
        sign3 = cc_local.signature('A1.001', 'sign3')

        sign0_params = {
            'epochs': 30,
            'cpu': 8,
            'learning_rate': 1e-3,
            'layers': ['Dense', 'Dense'],
            'layers_sizes': [256, 128],
            'activations': ['relu', 'tanh'],
            'dropouts': [0.1,  None],
        }
        s4 = cc_local.signature('A1.001', 'sign4', sign0_params=sign0_params)
        s4.fit(sign0, sign3)

        self.assertTrue(os.path.isfile(s4.data_path))
        self.assertTrue(os.path.isdir(s4.model_path))
        eval_dir = os.path.join(s4.model_path, 'smiles_eval')
        self.assertTrue(os.path.isdir(eval_dir))
        eval_dir = os.path.join(s4.model_path, 'smiles_applicability_eval')
        self.assertTrue(os.path.isdir(eval_dir))
        final_dir = os.path.join(s4.model_path, 'smiles_final')
        self.assertTrue(os.path.isdir(final_dir))
        final_dir = os.path.join(s4.model_path, 'smiles_applicability_final')
        self.assertTrue(os.path.isdir(final_dir))
        print(s4.shape)
        self.assertEqual(s4.shape[0], 1000)
        self.assertEqual(s4.shape[1], 128)
