"""TargetMate ML classes"""
import pickle
import os
from chemicalchecker.util import logged
from .base import StackedModel


def tm_from_disk(tm):
    if type(tm) != str:
        return tm
    with open(tm, "rb") as f:
        tm = pickle.load(f)
    return tm


@logged
class TargetMateStackedClassifier(StackedModel):
    """Stacked predictions"""

    def __init__(self, **kwargs):
        StackedModel.__init__(self, is_classifier=True, **kwargs)

    def on_disk_tmp(self):
        path = os.path.join(self.tmp_path, "tmp_tm.pkl")
        self.__log.info("Writing model to %s" % path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    def on_disk(self):
        path = os.path.join(self.models_path, "tm.pkl")
        self.__log.info("Writing model to %s" % path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    def on_disk_complete_model(self): # Added by Paula:
        path = os.path.join(self.models_path, "tm.pkl")
        self.__log.info("Writing model to %s" % path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path
