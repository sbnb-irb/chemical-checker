import os
import pytest
import shutil
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


class TestAdanet(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        self.adanet_path = os.path.join(self.data_dir, 'adanet')
        if os.path.exists(self.adanet_path):
            shutil.rmtree(self.adanet_path, ignore_errors=True)

    def tearDown(self):
        if os.path.exists(self.adanet_path):
            shutil.rmtree(self.adanet_path, ignore_errors=True)

    @skip_if_import_exception
    def test_classifiation_binary(self):
        from chemicalchecker.tool.adanet import AdaNet
        file_path = os.path.join(self.data_dir, 'classifiation_binary.h5')
        self.assertTrue(os.path.isfile(file_path))
        # check adanet initialization
        ada = AdaNet(file_path,
                     prediction_task='classification',
                     adanet_iterations=1,
                     epoch_per_iteration=1,
                     model_dir=self.adanet_path)
        self.assertEqual(ada.prediction_task, 'classification')
        self.assertEqual(ada.n_classes, 2)
        self.assertEqual(ada.train_size, 80000)
        self.assertEqual(ada.input_dimension, 20)
        self.assertEqual(ada.label_dimension, 1)
        self.assertEqual(ada.adanet_iterations, 1)
        self.assertEqual(ada.epoch_per_iteration, 1)
        self.assertEqual(ada.model_dir, self.adanet_path)
        # check results
        _, (res, _) = ada.train_and_evaluate()
        self.assertAlmostEqual(res['accuracy'], 0.981, 2)
        self.assertAlmostEqual(res['auc'], 0.994, 3)
        self.assertAlmostEqual(res['precision'], 0.977, 3)
        self.assertAlmostEqual(res['recall'], 0.986, 3)
        # check persistency and predict
        predict_fn = AdaNet.predict_fn(ada.save_dir)
        y_pred, y_true = AdaNet.predict_online(file_path, 'test', predict_fn)
        self.assertEqual(y_pred.shape, (10000, ada.label_dimension))
        self.assertEqual(y_true.shape, (10000, ada.label_dimension))
        y_pred, y_true = AdaNet.predict_online(file_path, 'test', predict_fn,
                                               probs=True,
                                               n_classes=ada.n_classes)
        self.assertEqual(y_pred.shape, (10000, ada.n_classes))
        self.assertEqual(y_true.shape, (10000, ada.label_dimension))

    @skip_if_import_exception
    def test_classifiation_multi(self):
        from chemicalchecker.tool.adanet import AdaNet
        file_path = os.path.join(self.data_dir, 'classifiation_multi.h5')
        self.assertTrue(os.path.isfile(file_path))
        # check adanet initialization
        ada = AdaNet(file_path,
                     prediction_task='classification',
                     adanet_iterations=1,
                     epoch_per_iteration=1,
                     model_dir=self.adanet_path)
        self.assertEqual(ada.prediction_task, 'classification')
        self.assertEqual(ada.n_classes, 3)
        self.assertEqual(ada.train_size, 80000)
        self.assertEqual(ada.input_dimension, 20)
        self.assertEqual(ada.label_dimension, 1)
        self.assertEqual(ada.adanet_iterations, 1)
        self.assertEqual(ada.epoch_per_iteration, 1)
        self.assertEqual(ada.model_dir, self.adanet_path)
        # check results
        _, (res, _) = ada.train_and_evaluate()
        self.assertAlmostEqual(res['accuracy'], 0.965, 3)
        self.assertAlmostEqual(res['loss'], 0.133, 3)
        # check persistency
        predict_fn = AdaNet.predict_fn(ada.save_dir)
        y_pred, y_true = AdaNet.predict_online(file_path, 'test', predict_fn)
        self.assertEqual(y_pred.shape, (10000, ada.label_dimension))
        self.assertEqual(y_true.shape, (10000, ada.label_dimension))
        y_pred, y_true = AdaNet.predict_online(file_path, 'test', predict_fn,
                                               probs=True,
                                               n_classes=ada.n_classes)
        self.assertEqual(y_pred.shape, (10000, ada.n_classes))
        self.assertEqual(y_true.shape, (10000, ada.label_dimension))

    @skip_if_import_exception
    def test_regression_single(self):
        from chemicalchecker.tool.adanet import AdaNet
        file_path = os.path.join(self.data_dir, 'regression_single.h5')
        self.assertTrue(os.path.isfile(file_path))
        # check adanet initialization
        ada = AdaNet(file_path,
                     prediction_task='regression',
                     adanet_iterations=1,
                     epoch_per_iteration=1,
                     model_dir=self.adanet_path)
        self.assertEqual(ada.prediction_task, 'regression')
        self.assertEqual(ada.train_size, 80000)
        self.assertEqual(ada.input_dimension, 20)
        self.assertEqual(ada.label_dimension, 1)
        self.assertEqual(ada.adanet_iterations, 1)
        self.assertEqual(ada.epoch_per_iteration, 1)
        self.assertEqual(ada.model_dir, self.adanet_path)
        # check results
        _, (res, _) = ada.train_and_evaluate()
        self.assertAlmostEqual(res['loss'], 4.455, 3)
        # check persistency and predict
        predict_fn = AdaNet.predict_fn(ada.save_dir)
        y_pred, y_true = AdaNet.predict_online(file_path, 'test', predict_fn)
        self.assertEqual(y_pred.shape, (10000, ada.label_dimension))
        self.assertEqual(y_true.shape, (10000, ada.label_dimension))

    @skip_if_import_exception
    def test_regression_multi(self):
        from chemicalchecker.tool.adanet import AdaNet
        file_path = os.path.join(self.data_dir, 'regression_multi.h5')
        self.assertTrue(os.path.isfile(file_path))
        # check adanet initialization
        ada = AdaNet(file_path,
                     prediction_task='regression',
                     adanet_iterations=1,
                     epoch_per_iteration=1,
                     model_dir=self.adanet_path)
        self.assertEqual(ada.prediction_task, 'regression')
        self.assertEqual(ada.train_size, 80000)
        self.assertEqual(ada.input_dimension, 20)
        self.assertEqual(ada.label_dimension, 5)
        self.assertEqual(ada.adanet_iterations, 1)
        self.assertEqual(ada.epoch_per_iteration, 1)
        self.assertEqual(ada.model_dir, self.adanet_path)
        # check results
        _, (res, _) = ada.train_and_evaluate()
        self.assertAlmostEqual(res['loss'], 230.59328, 3)
        # check persistency and predict
        predict_fn = AdaNet.predict_fn(ada.save_dir)
        y_pred, y_true = AdaNet.predict_online(file_path, 'test', predict_fn)
        self.assertEqual(y_pred.shape, (10000, ada.label_dimension))
        self.assertEqual(y_true.shape, (10000, ada.label_dimension))
