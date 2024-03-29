from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from autosklearn.metrics import roc_auc
import numpy as np
from chemicalchecker.util import logged

search_configs = {
    
    "random_forest" : {
        "model": RandomForestClassifier,
        "params": {
             "class_weight": ["balanced"],
             "n_estimators": [100, 500, 1000],
             "max_depth" : [None, 5, 10],
             "min_samples_split": [2, 3, 10],
             "criterion": ["gini", "entropy"],
             "max_features": ["sqrt", "log2", None, 0.5],
             "bootstrap": [True]
        }
    }
}


@logged
class GridClassifierConfigs:

    def __init__(self, base_mod, n_jobs, n_iter, **kwargs):
        self.SEED = 42
        self.n_iter = n_iter
        self.base_mod = base_mod
        self.n_jobs = n_jobs

    def as_pipeline(self, X, y, **kwargs):
        base_mod = search_configs[self.base_mod]["model"]
        params = search_configs[self.base_mod]["params"]
        space = 1
        for k, v in params.items():
            space *= len(v)
        n_iter = int(np.min([space, self.n_iter]))
        mod = RandomizedSearchCV(estimator = base_mod(),
                                 param_distributions = params,
                                 n_iter = self.n_iter,
                                 n_jobs = self.n_jobs,
                                 iid = False,
                                 cv = 5,
                                 verbose = 0,
                                 scoring = "roc_auc",
                                 random_state = self.SEED
                                 )        
        mod.fit(X, y)
        self.__log.info("Best score: %.3f" % mod.best_score_)
        params = mod.best_params_
        params["n_jobs"] = self.n_jobs
        mod = base_mod()
        mod.set_params(**params)
        self.__log.info(mod)
        return mod