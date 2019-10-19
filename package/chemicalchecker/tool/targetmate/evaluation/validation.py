import numpy as np
import pickle

from chemicalchecker.util import logged

from ..utils import metrics

@logged
class Validation:

    def __init__(self, splitter=None, destination_dir=None):
        """Initialize cross-validatin class

        Args:
            splitter(): If none specified, the corresponding TargetMate splitter is used (default=None).
            destination_dir(str): If non specified, the models path of the TargetMate instance is used (default=None).
        """
        self.destination_dir = destination_dir
        self.splitter   = splitter
        self.datasets   = []
        self.train_idx  = []
        self.test_idx   = []
        self.y_true     = []
        self.y_pred     = []
        self.y_pred_ens = []

    def compute(self, tm, data):
        """Cross-validate a targetmate classifier

        Args:

        """
        # TO-DO Determine base model only once.
        self.datasets = tm.datasets
        if not self.splitter:
            kf = tm.kfolder()
        else:
            kf = self.splitter
        i = 0
        for train_idx, test_idx in kf.split(X=np.zeros(len(data.activity)), y=data.activity):
            self.__log.info("Fold %0d" % (i+1))
            
            data_train = data[train_idx]
            data_test  = data[test_idx]
            tm.fit(data_train, is_tmp=True)
            pred = tm.predict(data_test)

            i += 1

    def score(self):
        pass

    def save(self):
        pass

    def validate(self, tm, data):
        self.compute(tm, data)
        self.score()

