"""Signature type 3.

Network embedding of observed *and* inferred similarity networks. Their added
value, compared to signatures type 2, is that they can be derived for
virtually *any* molecule in *any* dataset.
"""
import os
import numpy as np

from .signature_base import BaseSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged


@logged
class sign3(BaseSignature):
    """Signature type 2 class."""

    def __init__(self, signature_path, validation_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are 'graph', 'node2vec', and
                'adanet'.
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               validation_path, dataset, **params)
        self.validation_path = validation_path
        # generate needed paths
        self.data_path = os.path.join(signature_path, 'sign3.h5')
        self.model_path = os.path.join(signature_path, 'models')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.stats_path = os.path.join(signature_path, 'stats')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.stats_path)
        # assign dataset
        self.dataset = dataset
        # logging
        self.__log.debug('dataset: %s', dataset)
        self.__log.debug('data_path: %s', self.data_path)
        self.__log.debug('model_path: %s', self.model_path)
        self.__log.debug('stats_path: %s', self.stats_path)
        # get parameters or default values
        self.params = dict()
        self.params['graph'] = params.get('graph', None)
        self.params['node2vec'] = params.get('node2vec', None)
        self.params['adanet'] = params.get('adanet', None)

    def get_sign2_matrix(self, chemchecker):
        """Get combined matrix of stacked signature 2."""
        # get current space on which we'll train (using reference set to
        # limit redundancy)
        my_sign2 = chemchecker.get_signature(
            'sign2', 'full_map', self.dataset.code)
        # get other space signature 2 for molecule in current space (allow nan)
        other_spaces = list(chemchecker.datasets)
        other_spaces.remove(self.dataset.code)
        ref_dimension = my_sign2.shape[1]
        sign2_matrix = np.zeros(
            (my_sign2.shape[0], ref_dimension * len(other_spaces)),
            dtype=np.float32)
        for idx, ds in enumerate(other_spaces):
            sign2_ds = chemchecker.get_signature('sign2', 'full_map', ds)
            _, signs = sign2_ds.get_vectors(my_sign2.keys, include_nan=True)
            start_idx = ref_dimension * idx
            end_idx = ref_dimension * (idx + 1)
            sign2_matrix[:, start_idx:end_idx] = signs
            available = np.isin(my_sign2.keys, sign2_ds.keys)
            self.__log.info('%s shared molecules between %s and %s',
                            sum(available), self.dataset, ds)
        return sign2_matrix, my_sign2[:]

    def fit(self, chemchecker, reuse=True):
        """Learn a model."""
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
            from .sign2 import sign2
        except ImportError as err:
            raise err
        # adanet
        self.__log.debug('AdaNet fit %s based on other sign2', self.dataset)
        # get params and set folder
        adanet_params = self.params['adanet']
        adanet_path = os.path.join(self.model_path, 'adanet')
        if adanet_params:
            if 'model_dir' in adanet_params:
                adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # prepare train-test file
        traintest_file = os.path.join(self.model_path, 'traintest.h5')
        if adanet_params:
            traintest_file = adanet_params.pop(
                'traintest_file', traintest_file)
        if not reuse or not os.path.isfile(traintest_file):
            features, labels = self.get_sign2_matrix(chemchecker)
            Traintest.create(features, labels, traintest_file)
        # initialize adanet
        if adanet_params:
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file,
                         **adanet_params)
        else:
            ada = AdaNet(model_dir=adanet_path, traintest_file=traintest_file)
        # learn
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate()
        # save AdaNet performances and plots
        sign2_plot = Plot(self.dataset, adanet_path, self.validation_path)
        ada.save_performances(adanet_path, sign2_plot)
        self.__log.debug('model saved to %s' % adanet_path)

        self.mark_ready()

    def predict(self, sign1):
        """Use the learned model to predict the signature."""
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        pass

    @staticmethod
    def cross_fit(chemchecker, model_path, ds_from, ds_to, reuse=True):
        """Learn a predictor between sign_from and sign_to.

        Signatures should be `full` to maximize the intersaction, that's the
        training input.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
            from .sign2 import sign2
        except ImportError as err:
            raise err
        sign3.__log.debug('AdaNet cross fit signatures')
        # get params and set folder
        cross_pred_path = os.path.join(model_path, 'crosspred_%s' % ds_from)
        if not reuse or not os.path.isdir(cross_pred_path):
            os.makedirs(cross_pred_path)
        # create traintest file (simple intersection)
        ds_traintest_file = os.path.join(cross_pred_path, 'traintest.h5')
        sign_from = chemchecker.get_signature('sign2', 'full_map', ds_from)
        sign_to = chemchecker.get_signature('sign2', 'full_map', ds_to)
        if not reuse or not os.path.isfile(ds_traintest_file):
            _, X, Y = sign_from.get_intersection(sign_to)
            Traintest.create(X, Y, ds_traintest_file)
        # adanet
        ada_single_space = AdaNet(model_dir=cross_pred_path,
                                  traintest_file=ds_traintest_file)
        ada_single_space.train_and_evaluate()
        # add nearest neighbor predictor
        nearest_neighbor_pred = sign2.predict_nearest_neighbor(
            cross_pred_path, ds_traintest_file)
        extra_predictors = dict()
        extra_predictors['NearestNeighbor'] = nearest_neighbor_pred
        sign2_plot = Plot(ds_to, cross_pred_path, cross_pred_path)
        ada_single_space.save_performances(
            cross_pred_path, sign2_plot, extra_predictors)

    def average_cross_fit(self, chemchecker):
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
            from .sign2 import sign2
            from .neig1 import neig1
            from scipy.stats import pearsonr
            from sklearn.metrics import r2_score, mean_squared_error
            from sklearn.metrics import explained_variance_score
            import pandas as pd
            import pickle
            from sklearn.linear_model import LinearRegression
        except ImportError as err:
            raise err

        def _stats_row(y_true, y_pred, algo, dataset):
            row = dict()
            row['algo'] = algo
            row['dataset'] = dataset
            row['r2'] = r2_score(y_true, y_pred)
            pps = [pearsonr(y_true[:, x], y_pred[:, x])[0]
                   for x in range(y_true.shape[1])]
            row['pearson_avg'] = np.mean(pps)
            row['pearson_std'] = np.std(pps)
            row['mse'] = mean_squared_error(y_true, y_pred)
            row['explained_variance'] = explained_variance_score(
                y_true, y_pred)
            return row

        df = pd.DataFrame(columns=[
            'dataset', 'r2', 'pearson_avg', 'pearson_std', 'algo', 'mse',
            'explained_variance', 'time', 'architecture', 'nr_variables',
            'nn_layers', 'layer_size', 'architecture_history'])

        traintest_file = os.path.join(self.model_path, 'traintest.h5')
        linreg = dict()
        for part in ['train', 'test', 'validation']:
            # copy matrix from traintest_file
            traintest = Traintest(traintest_file, part)
            traintest.open()
            x_data = traintest.get_all_x()
            results = dict()
            results['AdaNet'] = traintest.get_all_x()
            results['NearestNeighbor'] = traintest.get_all_x()
            results['LinearRegression'] = traintest.get_all_x()
            y_data = traintest.get_all_y()
            traintest.close()
            # iterate on spaces
            other_spaces = list(chemchecker.datasets)
            other_spaces.remove(self.dataset.code)
            for idx, ds in enumerate(other_spaces):
                cross_pred_path = os.path.join(
                    self.model_path, 'crosspred_%s' % ds)
                ds_traintest_file = os.path.join(
                    cross_pred_path, 'traintest.h5')
                # train (on ds_traintest_file) or get algo
                # copy traintest_file columns
                col_slice = slice(idx * 128, (idx + 1) * 128)
                ds_x_data = x_data[:, col_slice]
                # not nan idxs
                notnan_idxs = ~np.isnan(ds_x_data).any(axis=1)
                # predict with adanet and append to x matrix
                save_dir = os.path.join(cross_pred_path, 'savedmodel')
                ds_pred = AdaNet.predict(save_dir, ds_x_data[notnan_idxs])
                results['AdaNet'][notnan_idxs, col_slice] = ds_pred
                # nn predictions
                nn_path = os.path.join(cross_pred_path, "nearest_neighbor")
                sign2_dest = os.path.join(nn_path, "sign2")
                nn_sign2 = sign2(sign2_dest, sign2_dest, ds)
                neig1_dest = os.path.join(nn_path, "neig1")
                nn_neig1 = neig1(neig1_dest, neig1_dest, "NN.001")
                nn_idxs = nn_neig1.get_kth_nearest(ds_x_data[notnan_idxs], 1)
                tmp = list()
                for idx in nn_idxs:
                    tmp.append(nn_sign2[idx])
                results['NearestNeighbor'][
                    notnan_idxs, col_slice] = np.vstack(tmp)
                # linear regression
                if part == 'train':
                    tt = Traintest(ds_traintest_file, part)
                    tt.open()
                    linreg[ds] = LinearRegression().fit(
                        tt.get_all_x(), tt.get_all_y())
                    tt.close()
                results['LinearRegression'][notnan_idxs, col_slice] = \
                    linreg[ds].predict(ds_x_data[notnan_idxs])

            for name, res in results.items():
                y_pred = np.zeros_like(y_data)
                for idx, row in enumerate(res):
                    mol = row.reshape((24, 128))
                    notnan_idxs = ~np.isnan(mol).any(axis=1)
                    y_pred[idx] = np.average(mol[notnan_idxs], axis=0)

                # Performances for AdaNet
                rows = _stats_row(y_data, y_pred, name, part)
                df.loc[len(df)] = pd.Series(rows)
                output_dir = os.path.join(self.model_path, 'crosspred_AVG')
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                output_pkl = os.path.join(output_dir, 'stats.pkl')
                with open(output_pkl, 'wb') as fh:
                    pickle.dump(df, fh)
                output_csv = os.path.join(output_dir, 'stats.csv')
                df.to_csv(output_csv)
