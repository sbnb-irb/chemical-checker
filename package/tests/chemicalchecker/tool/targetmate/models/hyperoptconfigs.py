"""See tutorial in https://docs.azuredatabricks.net/_static/notebooks/hyperopt-sklearn-model-selection.html"""

import numpy as np
import uuid
from chemicalchecker.util import logged
from sklearn import ensemble
from xgboost.sklearn import XGBClassifier
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import mlflow
from ..evaluation.score import validation_score

# Others

SEED = 42
MIN_TIMEOUT = 60

# Search configs

search_configs ={
    "random_forest": {
        "model": ensemble.RandomForestClassifier(class_weight="balanced", random_state=SEED),
        "params": {
            "n_estimators":[100, 500, 1000],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 3, 10],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"]
        }
    },
    "xgboost": {
        "model": XGBClassifier(random_state=SEED),
        "params": {
            "n_estimators": [100, 500, 1000],
            "max_depth": [1, 5, 10, 15],
            "colsample_bytree": [0.5, 1],
            "gamma": [0, 0.5, 1],
            "subsample": [0.5, 1],
            "learning_rate": [0.01, 0.1, 1],
            "min_child_weight": [1, 3, 6]
        }
    }
}


@logged
class HyperOpt(object):

    def __init__(self, base_mod, metric, n_jobs, n_iter, timeout,
                 is_cv,
                 is_classifier,
                 is_stratified,
                 n_splits,
                 test_size,
                 scaffold_split, **kwargs):

        self.params   = search_configs[base_mod]["params"]
        self.base_mod = search_configs[base_mod]["model"]

        self.metric = metric
        self.n_jobs  = n_jobs
        self.n_iter  = n_iter
        timeout = timeout / n_iter
        timeout = np.max([MIN_TIMEOUT, timeout])
        self.timeout = timeout

        self.is_cv          = is_cv
        self.is_classifier  = is_classifier
        self.is_stratified  = is_stratified
        self.scaffold_split = scaffold_split
        self.n_splits       = n_splits
        self.test_size      = test_size

        self.base_mod.set_params(n_jobs=n_jobs)

    def params2choices(self):
        params = {}
        for k,v in self.params.items():
            params[k] = hp.choice(k, v)
        return params

    def search(self, X, y, smiles):
        self.__log.info("Timeout: %d" % self.timeout)
        self.__log.info("CV: %s, Stratified: %s, Metric: %s" % (self.is_cv, self.is_stratified, self.metric))

        def objective(params):
            self.base_mod.set_params(**params)
            accuracy = validation_score(mod = self.base_mod,
                                        X = X, y = y,
                                        smiles = smiles,
                                        metric = self.metric,
                                        is_cv = self.is_cv,
                                        is_classifier = self.is_classifier,
                                        is_stratified = self.is_stratified,
                                        scaffold_split = self.scaffold_split,
                                        n_splits = self.n_splits,
                                        test_size = self.test_size,
                                        random_state = SEED
                                        )
            return {"loss": -accuracy, "status": STATUS_OK}

        params = self.params2choices()
        algo = tpe.suggest
        with mlflow.start_run():
            choice = fmin(
                fn=objective,
                space=params,
                algo=algo,
                max_evals=self.n_iter,
                rstate=np.random.RandomState(SEED))
        best_params = {}
        for k, idx in choice.items():
            best_params[k] = self.params[k][idx]
        return best_params


@logged
class HyperoptClassifierConfigs(HyperOpt):

    def __init__(self, base_mod, **kwargs):
        HyperOpt.__init__(self, base_mod, is_classifier=True, **kwargs)
        self.__log.info("Iters: %d, Timeout: %d" % (self.n_iter, self.timeout))

    def as_pipeline(self, X, y, smiles=None, **kwargs):
        """The smiles parameter is used for the chemistry-aware (e.g. scaffold) split"""
        best_params = self.search(X, y, smiles)
        mod = self.base_mod
        mod.set_params(**best_params)
        self.__log.info("Best model: %s" % mod)
        return mod
