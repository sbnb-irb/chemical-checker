import numpy as np
import pickle
from chemicalchecker.util import logged
from ..utils import metrics


def load_validation(destination_dir):
    with open(destination_dir, "rb") as f:
        return pickle.load(f)


class PrecomputedSplitter:

    def __init__(self, train_idx, test_idx):
        self.train_idx = train_idx
        self.test_idx = test_idx

    def split(X = None, y = None):
        yield self.train_idx, test_idx


@logged
class Validation:
    """Validation class."""

    def __init__(self, splitter=None, destination_dir=None):
        """Initialize validation class.

        Args:
            splitter(): If none specified, the corresponding TargetMate splitter is used (default=None).
            destination_dir(str): If non specified, the models path of the TargetMate instance is used (default=None).
        """
        self.destination_dir = destination_dir
        self.splitter   = splitter

    def _stack(self,)

    def compute(self, tm, data, train_idx, test_idx):
        # TO-DO Determine base model only once.
        self.__log.debug("Initializing all the results arrays")
        self.train_idx = None
        self.test_idx = None
        self.y_true = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.y_pred_ens_train = None
        self.y_pred_ens_test = None
        self.__log.debug("Computing validation")
        if train_idx is not None and test_idx is not None:
            kf = PrecomputedSplitter(train_idx, test_idx)
        else:
            if not self.splitter:
                kf = tm.kfolder()
            else:
                kf = self.splitter
        i = 0
        for train_idx, test_idx in kf.split(X=np.zeros(len(data.activity)), y=data.activity):
            self.__log.info("Fold %0d" % (i+1))
            data_train = data[train_idx]
            data_test  = data[test_idx]
            self.__log.debug("Fit with train set")
            tm.fit(data_train, is_tmp=True)
            self.__log.debug("Predict with train set")
            pred_train = tm.predict(data_train)
            self.__log.debug("Predict with test set")
            pred_test = tm.predict(data_test)
            self.__log.debug("Appending")
            self.datasets 
            i += 1

    def score(self):
        pass

    def save(self):
        self.__log.debug("Saving validation")
        with open(self.destination_dir, "wb") as f:
            pickle.dump(self, f)

    def validate(self, tm, data, train_idx=None, test_idx=None, save=True):
        """Validate a TargetMate classifier.

        Args:
            tm: TargetMate model.
            data: Data object.
            train_idx: Precomputed indices for the train set (default=None). 
            test_idx: Precomputed indices for the test set (default=None).
        """
        if self.destination_dir is None:
            destination_dir = os.path.join(tm.models_path, "validation.pkl")
        else:
            destination_dir = self.destination_dir
        self.datasets = tm.datasets
        self.compute(tm, data)
        self.score()
        if save:
            self.save()