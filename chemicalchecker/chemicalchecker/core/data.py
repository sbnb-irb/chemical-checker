"""This class initialize and serve the different Chemical Checker data types.

It is a factory of different signatures (including along with clusters and
neighbors). Is the place where the classes implementing such data types are
imported and initialized.
"""
from .sign0 import sign0
#from .sign1 import sign1
#from .sign2 import sign2
#from .sign3 import sign3
#from .clst import clst

class DataFactory():

    def make_data(self, datatype, data_path):
        if datatype in globals():
            return eval(datatype)(data_path)
        else:
            raise Exception("Data type %s not available" % datatype)
