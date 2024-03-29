from autosklearn.classification import AutoSklearnClassifier as BaseAutoSklearnClassifier
from autosklearn.metrics import roc_auc
from sklearn import model_selection
import numpy as np
import uuid
import os
import random
from chemicalchecker.util import logged
from ..utils.log import set_logging


def read_logging():
    import yaml
    path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(path, "logger/logging.yaml")
    with open(filename, "r") as f:
        logging_dict = yaml.load(f)
    return logging_dict


class AutoSklearnClassifier(BaseAutoSklearnClassifier):

    def __init__(self, log, **kwargs):
        BaseAutoSklearnClassifier.__init__(self, **kwargs)
        self.log = log
        # set_logging(self.log)

    def prefit(self, X, y):
        super().fit(X.copy(), y.copy())
        # set_logging(self.log)

    def fit(self, X, y):
        super().refit(X.copy(), y.copy())
        # set_logging(self.log)


@logged
class AutoSklearnClassifierConfigs:

    def __init__(self,
                 n_jobs,
                 tmp_path,
                 train_timeout,
                 resampling_strategy="cv",
                 log="INFO",
                 **kwargs):
        """Just set up paths and determine sampling strategy"""
        auto_path = os.path.join(os.path.join(tmp_path, "autosklearn"))
        tmp_folder = os.path.join(auto_path, "tmp")
        output_folder = os.path.join(auto_path, "output")
        os.makedirs(tmp_folder, exist_ok = True)
        os.makedirs(output_folder, exist_ok = True)
        self.log = log,
        self.tmp_folder = tmp_folder
        self.output_folder = output_folder
        self.n_jobs = n_jobs
        self.memory = 1024*2*self.n_jobs
        self.__log.info("Number of jobs: %d" % self.n_jobs)
        self.train_timeout = train_timeout
        if resampling_strategy == "cv":
            self.__log.info("Using crossvalidation")
            self.resampling_strategy = model_selection.StratifiedKFold
            self.resampling_strategy_arguments = {"n_splits": 5, "shuffle": True}
        else:
            self.__log.info("Using holdout")
            self.resampling_strategy = model_selection.StratifiedShuffleSplit
            self.resampling_strategy_arguments = {"train_size": 0.8, "n_splits": 1}

    def instantiate(self):
        """Set up a classifier"""
        tag = str(uuid.uuid4())
        tmp_folder = os.path.join(self.tmp_folder, tag)
        output_folder = os.path.join(self.output_folder, tag)
        seed = random.randint(1, 999)
        # Some settings
        self.__log.info("Time left for task: %d sec." % self.train_timeout)
        per_run_time_limit = int(np.max([self.train_timeout / 10, 60]))
        self.__log.info("Per run time limit: %d sec." % per_run_time_limit)
        # ensemble_memory_limit = int(self.memory/2)
        # self.__log.info("Ensemble memory limit: %d Mb" % ensemble_memory_limit)
        # ml_memory_limit = int(self.memory/2)
        # self.__log.info("ML memory limit: %d Mb" % ml_memory_limit)
        # Instantiate classifier
        base_mod = AutoSklearnClassifier(
            log = self.log,
            time_left_for_this_task = self.train_timeout,
            per_run_time_limit = per_run_time_limit,
            ensemble_nbest = 100,
            resampling_strategy = self.resampling_strategy,
            resampling_strategy_arguments = self.resampling_strategy_arguments,
            n_jobs = self.n_jobs,
            seed = seed,
            logging_config=None,
            tmp_folder = tmp_folder,
            output_folder = output_folder,
            delete_tmp_folder_after_terminate = False,
            delete_output_folder_after_terminate = False #,
            # ensemble_memory_limit=ensemble_memory_limit,
            # ml_memory_limit=ml_memory_limit
            )
        return base_mod

    def as_pipeline(self, X, y, **kwargs):
        """Prefit and refit"""
        mod = self.instantiate()
        mod.prefit(X, y)
        self.__log.info(mod.sprint_statistics())
        return mod
