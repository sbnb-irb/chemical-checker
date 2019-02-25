"""Signature type 3.

Network embedding of observed *and* inferred similarity networks. Their added
value, compared to signatures type 2, is that they can be derived for
virtually *any* molecule in *any* dataset.

Signatures type 3 are the result of a two-step process:

1. Generate predictors that given sign2 in any dataset predict sign2 in
current dataset. NB: To have as much as possible training molecules we use full
sign2 of two datasets to find the intersection and after that remove redundant
molecules.

2. Train a meta-predictor able to handle missing data. The meta-predictor is
trained against molecules with available sign2 in current dataset (Y).
The data for training (X) is the collection of sign2 available in other spaces
for each molecule.

"""
import os
from pathlib2 import Path
from .signature_base import BaseSignature
import chemicalchecker
from chemicalchecker.util import HPC
from chemicalchecker.util import Plot
from chemicalchecker.util import logged
from chemicalchecker.util import Config
try:
    from chemicalchecker.tool import AdaNet, Traintest
except:
    pass


@logged
class sign3(BaseSignature):
    """Signature type 2 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are 'graph', 'node2vec', and
                'adanet'.
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path, dataset, **params)
        # generate needed paths
        self.data_path = os.path.join(signature_path, 'sign2.h5')
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
        self.params['adanet'] = params.get('adanet', None)

    def fit(self, sign2, reuse=True):
        """Learn a model.


        Args:
            sign2(list): Lst of Signature type 2.
        """
        #########
        # step 1:
        #########

        #########
        # step 2:
        #########
        pass

        Path(os.path.join(self.model_path, self.readyfile)).touch()

    def predict(self, sign1):
        """Use the learned model to predict the signature."""
        pass

    @staticmethod
    def cross_fit(sign_from, sign_to, model_path, params={'adanet': {}}, reuse=True):
        """Learn a predictor between sign_from and sign_to.

        Signatures should be `full` to maximaze the intersaction, that's the
        training input.
        """
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
