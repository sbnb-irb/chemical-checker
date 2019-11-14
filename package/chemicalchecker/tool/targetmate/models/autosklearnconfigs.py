from autosklearn.classification import AutoSklearnClassifier as BaseAutoSklearnClassifier
from autosklearn.metrics import roc_auc
from sklearn import model_selection
import numpy as np
import random
import uuid
import os
#os.environ["OMP_NUM_THREADS"] = "1"

class AutoSklearnClassifier(BaseAutoSklearnClassifier):

    def __init__(self, **kwargs):
        BaseAutoSklearnClassifier.__init__(self, **kwargs)

    def prefit(self, X, y):
        super().fit(X.copy(), y.copy(), metric = roc_auc)

    def fit(self, X, y):
        super().refit(X.copy(), y.copy())
        

class AutoSklearnClassifierConfigs:

    def __init__(self, n_jobs, tmp_path, train_timeout, resampling_strategy="cv", **kwargs):
        """Just set up paths"""
        auto_path = os.path.join(os.path.join(tmp_path, "autosklearn"))
        tmp_folder = os.path.join(auto_path, "tmp")
        output_folder = os.path.join(auto_path, "output")
        os.makedirs(tmp_folder, exist_ok = True)
        os.makedirs(output_folder, exist_ok = True)
        self.tmp_folder = tmp_folder
        self.output_folder = output_folder
        self.n_jobs = n_jobs
        self.train_timeout = train_timeout
        if resampling_strategy == "cv":
            self.resampling_strategy = model_selection.StratifiedKFold
            self.resampling_strategy_arguments = {"n_splits": 5, "shuffle": True}
        else:
            self.resampling_strategy = model_selection.StratifiedShuffleSplit
            self.resampling_strategy_arguments = {"train_size": 0.8, "n_splits": 1}
        
    def instantiate(self):
        """Set up a classifier"""
        tag = str(uuid.uuid4())
        tmp_folder = os.path.join(self.tmp_folder, tag)
        output_folder = os.path.join(self.output_folder, tag)
        # Instantiate classifier
        self.base_mod = AutoSklearnClassifier(
            time_left_for_this_task = self.train_timeout,
            resampling_strategy = self.resampling_strategy,
            resampling_strategy_arguments = self.resampling_strategy_arguments,
            tmp_folder = tmp_folder,
            output_folder = output_folder,
            delete_tmp_folder_after_terminate = False,
            delete_output_folder_after_terminate = False
            )

    def as_pipeline(self, X, y, **kwargs):
        """Prefit and refit"""
        self.instantiate()
        shuff = np.array(range(len(y)))
        random.shuffle(shuff)
        mod = self.base_mod
        mod.prefit(X, y)
        return mod


