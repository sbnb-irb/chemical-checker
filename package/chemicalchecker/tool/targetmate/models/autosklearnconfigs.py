from autosklearn.classification import AutoSklearnClassifier as BaseAutoSklearnClassifier
from autosklearn.metrics import roc_auc
import numpy as np
import random


class AutoSklearnClassifier(BaseAutoSklearnClassifier):

    def __init__(self, **kwargs):
        BaseAutoSklearnClassifier.__init__(self, **kwargs)

    def prefit(self, X, y):
        super().fit(X.copy(), y.copy(), metric = roc_auc)

    def fit(self, X, y):
        super().refit(X.copy(), y.copy())
        

class AutoSklearnClassifierConfigs:

    def __init__(self, n_jobs, **kwargs):
        self.base_mod = AutoSklearnClassifier(
            time_left_for_this_task = 300,
            resampling_strategy = 'cv',
            resampling_strategy_arguments = {'folds': 5},
            n_jobs = n_jobs,
            delete_tmp_folder_after_terminate = False,
            delete_output_folder_after_terminate = False
            )

    def as_pipeline(self, X, y, **kwargs):
        shuff = np.array(range(len(y)))
        random.shuffle(shuff)
        mod = self.base_mod
        mod.prefit(X, y)
        mod.fit(X, y)
        return mod


