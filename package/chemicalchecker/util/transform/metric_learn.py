"""Semisupervised metric learning

We first determine the number of dimensions by a PCA.
"""
import os
from keras.layers import Dense

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import NeighborTripletTraintest
from chemicalchecker.tool.siamese import SiameseTriplets
from .base import BaseTransform

params = {
    'epochs': 3,
    'dropouts': [None],
    'layers_sizes': None,
    'learning_rate': "auto",
    'batch_size': 128,
    'activations': ["tanh"],
    'layer': [Dense],
    'loss_func': 'orthogonal_tloss',
    'margin': 1.0,
    'alpha': 1.0
}


@logged
class MetricLearn(BaseTransform):

    def __init__(self, sign1, tmp, name, max_keys, params):
        if tmp:
            raise Exception("Metric learn is not prepared to work with tmp")
        BaseTransform.__init__(self, sign1, name, max_keys, tmp)
        if params["layers_sizes"] is None:
            layer_size = sign1.info_h5["V_tmp"][1]
            params["layers_sizes"] = [layer_size]
        self.params = params

    def _fit(self, triplets):
        params = self.params
        # Get signatures
        V = self.sign_ref[:]
        dest_h5 = os.path.join(self.model_path, self.name+"_eval.h5")
        # generate traintest file
        NeighborTripletTraintest.precomputed_triplets(
            V, triplets, dest_h5,
            mean_center_x=True,  shuffle=True,
            split_names=['train_train', 'train_test', 'test_test'],
            split_fractions=[.8, .1, .1])
        # train siamese network
        model_dir = os.path.join(self.model_path, self.name+"_eval")
        params["traintest_file"] = dest_h5
        mod = SiameseTriplets(model_dir, evaluate=True, plot=False, **params)
        mod.fit()
        # generate traintest file for the final model
        dest_h5 = os.path.join(self.model_path, self.name+".h5")
        NeighborTripletTraintest.precomputed_triplets(
            V, triplets, dest_h5,
            mean_center_x=True,  shuffle=True,
            split_names=['train_train'],
            split_fractions=[1.])
        # train siamese network
        model_dir = os.path.join(self.model_path, self.name)
        params["traintest_file"] = dest_h5
        params["epochs"] = mod.last_epoch
        mod = SiameseTriplets(model_dir, evaluate=False, plot=False, **params)
        mod.fit()
        # Save and predict
        self.model_dir = model_dir
        self.predict(self.sign_ref)
        self.predict(self.sign)
        self.save()

    def _predict(self, sign1):
        mod = SiameseTriplets(self.model_dir)
        V = mod.predict(sign1[:])
        self.overwrite(sign1=sign1, V=V, keys=sign1.keys)


@logged
class UnsupervisedMetricLearn(MetricLearn):

    def __init__(self, sign1, tmp):
        MetricLearn.__init__(self, sign1, tmp, "unsupml", max_keys=None, params=params)

    def fit(self):
        self.sign_ref.neighbors(tmp=True)
        self.__log.debug("Now getting triplets")
        triplets = self.sign_ref.get_self_triplets(True)
        self.__log.debug("Doing unsupervised metric learning")
        self._fit(triplets)

    def predict(self, sign1):
        self._predict(sign1)
        

@logged
class SemiSupervisedMetricLearn(MetricLearn):

    def __init__(self, sign1, tmp):
        MetricLearn.__init__(self, sign1, "semiml", max_keys=None, params=params)

    def fit(self):
        self.__log.debug("Getting precalculated triplets throughout the CC")
        triplets = self.sign_ref.get_triplets()
        self._fit(triplets)

    def predict(self, sign1):
        self._predict(sign1)

