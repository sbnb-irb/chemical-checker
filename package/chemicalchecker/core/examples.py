"""This class fetches examples to be used in tutorials and notebooks"""

import os
import numpy as np
import csv
import pandas as pd
import pickle

from chemicalchecker.util import logged

class BaseExample(object):

    def __init__(self):
        """Initialize"""
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

    def check_type(self, key_type):
        if key_type not in ["inchikey", "smiles", "src"]:
            raise Exception("key_type must be one of 'inchikey', 'smiles' and 'src'")


@logged
class Example(BaseExample):
    """The Example class contains some example data, accessible with the class methods."""
    def __init__(self):
        """Initialize the class"""
        BaseExample.__init__(self)

    def toy_matrix(self, key_type='inchikey', fit_case=True):
        """A toy matrix, just to test the sign0
           
            Args:
               key_type(str): One of 'inchikey', 'smiles', 'src' (source) (default='inchikey')
               fit_case(bool): Get a fit case or a predict/transform case (default=True)
            
            Returns:
               X, keys, features
        """
        self.check_type(key_type)
        self.__log.info("Getting a toy matrix dataset")
        with open(os.path.join(self.path, "toy_matrix.pkl"), "rb") as f:
            d = pickle.load(f)
        if fit_case:
            suf = 1
        else:
            suf = 2
        X = d["X%d" % suf]
        features = d["features%d" % suf]
        if key_type == "inchikey":
            keys = d["key_inks%d" % suf]
        if key_type == "smiles":
            keys = d["key_smis%d" % suf]
        if key_type == "src":
            keys = d["key_srcs%d" % suf]
        return X, keys, features

    def toy_pairs(self, key_type='inchikey', fit_case=True):
        """Toy pairs, just to test the sign0
           
            Args:
               key_type(str): One of 'inchikey', 'smiles', 'src' (source) (default='inchikey')
               fit_case(bool): Get a fit case or a predict/transform case (default=True)
            
            Returns:
               pairs
        """
        self.check_type(key_type)
        self.__log.info("Getting a toy pairs dataset")
        with open(os.path.join(self.path, "toy_pairs.pkl"), "rb") as f:
            d = pickle.load(f)
        if fit_case:
            suf = 1
        else:
            suf = 2
        if key_type == "inchikey":
            pairs = d["pairs_inks%d" % suf]
        if key_type == "smiles":
            pairs = d["pairs_smis%d" % suf]
        if key_type == "src":
            pairs = d["pairs_srcs%d" % suf]
        return pairs

    def drug_targets(self):
        """Drug targets from PharmacoDB

            Returns:
                pairs
        """
        self.__log.info("Getting drug targets (example of unweighted pairs)")
        pairs = []
        with open(os.path.join(self.path, "pharmacodb_targets.tsv"), "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                pairs += [(r[0], r[1])]
        return pairs

    def cell_sensitivity(self):
        """Cell sensitivity data from CTRP

            Returns:
                X, keys, features
        """
        self.__log.info("Getting cell sensitivity data (example of continuous data)")
        df = pd.read_csv(os.path.join(self.path, "ctrp_auc.csv"), delimiter=",")
        keys = np.array(df[df.columns[0]])
        features = np.array(df.columns[1:])
        X = np.array(df[df.columns[1:]])
        return X, keys, features

    