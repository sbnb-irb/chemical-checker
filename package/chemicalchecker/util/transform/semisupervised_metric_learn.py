"""Semisupervised metric learning

We first determine the number of dimensions by a PCA.
"""
import os

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import NeighborTripletTraintest
from chemicalchecker.tool.siamese import SiameseTriplets
from .base import BaseTransform

params = {
    'epochs': 2,
    'dropout': 0.2,
    'layers': [128],
    'learning_rate': 1e-4,
    'batch_size': 128,
    'patience': 10,
    'loss_func': 'orthogonal_tloss',
    'margin': 1.0,
    'alpha': 1.0,
    'num_triplets': 100,
    'augment_fn': None,
    'augment_scale': 1
}

@logged
class SemiSupervisedMetricLearn(BaseTransform):

    def __init__(self, sign1):
        BaseTransform.__init__(self, sign1, "semiml", max_keys=None)

    def fit(self):
        # Get signatures
        V = self.sign_ref[:]
        triplets = self.sign_ref.get_triplets()
        dest_h5 = os.path.join(self.model_path, self.name+".h5")
        # generate traintest file
        NeighborTripletTraintest.precomputed_triplets(
            V, triplets, dest_h5,
            mean_center_x=True,  shuffle=True,
            split_names=['train_train', 'train_test', 'test_test'],
            split_fractions=[.8, .1, .1])
        # train siamese network
        model_dir = os.path.join(self.model_path, self.name) 
        mod = SiameseTriplets(model_dir, params, traintest_file=dest_h5)
        mod.fit()
        self.model_dir = model_dir
        self.save()

    def predict(self):
        mod = Siamese(self.model_dir)
        mod.predict(X)

