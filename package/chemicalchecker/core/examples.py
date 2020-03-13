"""This class fetches examples to be used in tutorials and notebooks"""

import os
import numpy as np
import csv
import pandas as pd

from chemicalchecker.util import logged

class BaseExample(object):

    def __init__(self):
        """Initialize"""
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

@logged
class Example(BaseExample):
    """The Example class contains some example data, accessible with the class methods."""
    def __init__(self):
        """Initialize the class"""
        BaseExample.__init__(self)

    def toy_matrix(self, key_type, fit_case):
        """A toy matrix, just to test the sign0"""
        self.__log.info("Getting a toy matrix dataset")
        with open(os.path.join(self.path, "toy_matrix.pkl"), "rb") as f:
            d = pickle.load(f)

    def toy_pairs(self):
        """Toy pairs, just to test sign0"""
        self.__log.info("Getting a toy pairs dataset")

    def drug_targets(self):
        """Drug targets from PharmacoDB"""
        self.__log.info("Getting drug targets (example of unweighted pairs)")
        pairs = []
        with open(os.path.join(self.path, "pharmacodb_targets.tsv"), "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                pairs += [(r[0], r[1])]
        return pairs

    def cell_sensitivity(self):
        """Cell sensitivity data from CTRP"""
        self.__log.info("Getting cell sensitivity data (example of continuous data)")
        df = pd.read_csv(os.path.join(self.path, "ctrp_auc.csv"), delimiter=",")
        keys = np.array(df[df.columns[0]])
        features = np.array(df.columns[1:])
        X = np.array(df[df.columns[1:]])
        return X, keys, features

    