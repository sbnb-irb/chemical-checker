"""Metric learning.

We bias the sign1 to respect triplets sampled from the whole CC
(semi-supervised) or from the space itself (unsupervised).
"""
import os
import pickle
import numpy as np

from .base import BaseTransform

from chemicalchecker.util import logged
from chemicalchecker.tool.siamese import SiameseTriplets
from chemicalchecker.util.splitter import PrecomputedTripletSampler


@logged
class MetricLearn(BaseTransform):
    """MetricLearn class."""

    def __init__(self, sign1, tmp, name, max_keys):
        try:
            from tensorflow.keras.layers import Dense
        except ImportError:
            raise ImportError("requires tensorflow " +
                              "https://www.tensorflow.org/")
        params = {
            'epochs': 3,
            'dropouts': [None, None],
            'layers_sizes': None,
            'learning_rate': "auto",
            'batch_size': 128,
            'activations': ["tanh", "tanh"],
            'layers': [Dense, Dense],
            'loss_func': 'orthogonal_tloss',
            'margin': 1.0,
            'alpha': 1.0
        }
        if tmp:
            raise Exception("Metric learn is not prepared to work with tmp")
        BaseTransform.__init__(self, sign1, name, max_keys, tmp)
        if params["layers_sizes"] is None:
            input_size = self.sign_ref.shape[1]
            final_layer_size = sign1.info_h5["V_tmp"][1]
            params["layers_sizes"] = [int((input_size + final_layer_size) / 2),
                                      final_layer_size]
        self.params = params

    def _fit(self, triplets):
        params = self.params
        # Get signatures
        V = self.sign_ref[:]
        dest_h5 = os.path.join(self.model_path, self.name + "_eval.h5")
        # generate traintest file
        triplet_sampler = PrecomputedTripletSampler(
            None, self.sign_ref, dest_h5)
        triplet_sampler.generate_triplets(
            V, self.sign_ref.keys, triplets,
            mean_center_x=True,  shuffle=True,
            split_names=['train_train', 'train_test', 'test_test'],
            split_fractions=[.8, .1, .1])
        # train siamese network
        model_dir = os.path.join(self.model_path, self.name + "_eval")
        params["traintest_file"] = dest_h5
        mod = SiameseTriplets(model_dir, evaluate=True, plot=False, **params)
        mod.fit()
        # generate traintest file for the final model
        dest_h5 = os.path.join(self.model_path, self.name + ".h5")
        triplet_sampler = PrecomputedTripletSampler(
            None, self.sign_ref, dest_h5)
        NeighborTripletTraintest.precomputed_triplets(
            V, triplets, dest_h5,
            mean_center_x=True,  shuffle=True,
            split_names=['train_train'],
            split_fractions=[1.])
        # train siamese network
        model_dir = os.path.join(self.model_path, self.name)
        params["traintest_file"] = dest_h5
        params["learning_rate"] = mod.learning_rate
        mod = SiameseTriplets(model_dir, evaluate=False, plot=False, **params)
        mod.fit()
        # Save and predict
        self.model_dir = model_dir
        self._predict(self.sign_ref, find_scale=True)
        self._predict(self.sign, find_scale=False)
        self.save()

    def _predict(self, sign1, find_scale=False):
        mod = SiameseTriplets(self.model_dir)
        V = mod.predict(sign1[:])
        if find_scale:
            p25 = np.percentile(V.ravel(), 25)
            p75 = np.percentile(V.ravel(), 75)
            with open(os.path.join(self.model_dir, "ml_percs.pkl"), "wb") as f:
                pickle.dump((p25, p75), f)
        else:
            with open(os.path.join(self.model_dir, "ml_percs.pkl"), "rb") as f:
                p25, p75 = pickle.load(f)
        m = 2. / (p75 - p25)
        n = 1 - m * p75
        V = m * V + n
        self.overwrite(sign1=sign1, V=V, keys=sign1.keys)


@logged
class UnsupervisedMetricLearn(MetricLearn):

    def __init__(self, sign1, tmp):
        MetricLearn.__init__(self, sign1, tmp, "unsupml", max_keys=None)

    def fit(self):
        self.sign_ref.neighbors(tmp=True)
        self.__log.debug("Now getting triplets")
        triplets = self.sign_ref.get_self_triplets(True)
        self.__log.debug("Doing unsupervised metric learning")
        self._fit(triplets)

    def predict(self, sign1):
        self._predict(sign1, find_scale=False)


@logged
class SemiSupervisedMetricLearn(MetricLearn):

    def __init__(self, sign1, tmp):
        MetricLearn.__init__(self, sign1, tmp, "semiml", max_keys=None)

    def fit(self):
        self.__log.debug("Getting precalculated triplets throughout the CC")
        triplets = self.sign_ref.get_triplets(True)
        self._fit(triplets)

    def predict(self, sign1):
        self._predict(sign1, find_scale=False)
