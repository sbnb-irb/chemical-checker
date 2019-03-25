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

    def get_probabilities(self, chemchecker):
        """Get the probabilities for space being present."""
        # get current space on which we'll train (using reference set to
        # limit redundancy)
        my_sign2 = chemchecker.get_signature(
            'sign2', 'reference', self.dataset.code)
        # get other space signature 2 for molecule in current space (presence)
        other_spaces = list(chemchecker.datasets)
        other_spaces.remove(self.dataset.code)
        available_mols = np.zeros((my_sign2.shape[0], len(other_spaces)),
                                  dtype=np.float32)
        for idx, ds in enumerate(other_spaces):
            sign2_ds = chemchecker.get_signature('sign2', 'full_map', ds)
            available = np.isin(my_sign2.keys, sign2_ds.keys)
            available_mols[:, idx] = available.T
            self.__log.info('%s shared molecules between %s and %s',
                            sum(available), self.dataset, ds)
        # get probabilities for a given space (column)
        p_space = np.sum(available_mols, axis=0) / np.sum(available_mols)
        # get probabilities for how many spaces for a molecule (row)
        p_count = dict()
        for row in available_mols:
            r = np.sum(row.astype(int))
            if r not in p_count:
                p_count[r] = 0.0
            p_count[r] += 1
        for k in p_count:
            p_count[k] /= len(available_mols)
        return p_space, p_count

    def _print_most_common_combinations(self, chemchecker):
        my_sign2 = chemchecker.get_signature(
            'sign2', 'reference', self.dataset.code)
        # get other space signature 2 for molecule in current space (presence)
        other_spaces = list(chemchecker.datasets)
        other_spaces.remove(self.dataset.code)
        available_mols = np.zeros((my_sign2.shape[0], len(other_spaces)),
                                  dtype=np.float32)
        for idx, ds in enumerate(other_spaces):
            sign2_ds = chemchecker.get_signature('sign2', 'full_map', ds)
            available = np.isin(my_sign2.keys, sign2_ds.keys)
            available_mols[:, idx] = available.T
            self.__log.info('%s shared molecules between %s and %s',
                            sum(available), self.dataset, ds)
        # most common combinations?
        freqs = dict()
        for row in available_mols:
            r = ''.join(row.astype(int).astype(str).tolist())
            if r not in freqs:
                freqs[r] = 0.0
            freqs[r] += 1
        for k in freqs:
            freqs[k] /= len(available_mols)
            freqs[k] *= 100
        top10 = sorted(list(tuple(freqs.items())),
                       key=lambda x: x[1], reverse=True)[:10]
        self.__log.info("Most frequent combinations:")
        for k, v in top10:
            self.__log.info("%s  %.2f%%", k, v)

    def get_sign2_matrix(self, chemchecker):
        """Get combined matrix of stacked signature 2."""
        # get current space on which we'll train (using reference set to
        # limit redundancy)
        my_sign2 = chemchecker.get_signature(
            'sign2', 'reference', self.dataset.code)
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
        adanet_path = os.path.join(self.model_path, 'adanet_augment')
        if adanet_params:
            if 'model_dir' in adanet_params:
                adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # get probabilities
        #probs = self.get_probabilities(chemchecker)
        # prepare train-test file
        traintest_file = os.path.join(adanet_path, 'traintest.h5')
        if adanet_params:
            traintest_file = adanet_params.get(
                'traintest_file', traintest_file)
            adanet_params.pop('traintest_file')

        if not reuse or not os.path.isfile(traintest_file):
            features, labels = self.get_sign2_matrix(chemchecker)
            Traintest.create(features, labels, traintest_file)
        if adanet_params:
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file,
                         **adanet_params)
        else:
            ada = AdaNet(model_dir=adanet_path, traintest_file=traintest_file)
        # learn NN with AdaNet
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate()
        # save AdaNet performances and plots
        sign2_plot = Plot(self.dataset, adanet_path, self.validation_path)
        # baseline comparison to cross predictors and neraest neighbors
        other_spaces = list(chemchecker.datasets)
        other_spaces.remove(self.dataset.code)
        extra_preditors = dict()
        for idx, ds in enumerate(other_spaces):
            col_idx = other_spaces.index(ds)
            cols = (col_idx * 128, (col_idx + 1) * 128)
            cross_pred_path = os.path.join(
                self.model_path, 'crosspred_%s' % ds)
            nearest_neighbor_pred = sign2.predict_nearest_neighbor(
                cross_pred_path, traintest_file, cols=cols)
            # we might skip dataset without intersections
            if nearest_neighbor_pred is None:
                continue
            extra_preditors['NearestNeighbor_%s' % ds] = nearest_neighbor_pred

        ada.save_performances(adanet_path, sign2_plot, extra_preditors)
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
    def cross_fit(sign_from, sign_to, model_path, params={'adanet': {}}, reuse=True):
        """Learn a predictor between sign_from and sign_to.

        Signatures should be `full` to maximize the intersaction, that's the
        training input.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
        except ImportError as err:
            raise err
        sign3.__log.debug('AdaNet cross fit signatures')
        # get params and set folder
        adanet_params = params['adanet']
        adanet_path = os.path.join(model_path, 'adanet')
        if adanet_params:
            if 'model_dir' in adanet_params:
                adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # prepare train-test file
        traintest_file = os.path.join(adanet_path, '../traintest.h5')
        if adanet_params:
            traintest_file = adanet_params.get(
                'traintest_file', traintest_file)
        if not reuse or not os.path.isfile(traintest_file):
            _, X, Y = sign_from.get_non_redundant_intersection(sign_to)
            Traintest.create(X, Y, traintest_file)
        if adanet_params:
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file, **adanet_params)
        else:
            ada = AdaNet(model_dir=adanet_path, traintest_file=traintest_file)
        # learn NN with AdaNet
        sign3.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate()
        # save AdaNet performances and plots
        sign2_plot = Plot(sign_from.dataset, adanet_path, adanet_path)
        ada.save_performances(adanet_path, sign2_plot)
        sign3.__log.debug('model saved to %s' % adanet_path)
