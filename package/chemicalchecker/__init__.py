"""**Welcome to the Chemical Checker Package documentation!**

 Goal
    The Chemical Checker Package aims to provide an interface to use all
    Chemical Checker tools. From the ingestion of new data (pre-processing)
    to generation of the different signatures types (signaturization)
    providing common methods for accessing available data and models.

 Organization
    The package is organized in four subpackages:

        * The :mod:`~chemicalchecker.core`  module holds the mains
          Chemical Checker functionalities. It holds the
          :class:`ChemicalChecker`
          entry-point, the Signature classes and ``HDF5`` i/o
          implementation.

        * The :mod:`~chemicalchecker.database` module include DB
          access and table definitions.

        * The :mod:`~chemicalchecker.tool` module gather
          experimental code and wrappers to external software.

        * The :mod:`~chemicalchecker.util` module pulls together
          general utilities.
"""
from .core import ChemicalChecker
from .util import Config


__author__ = """SBNB"""
__email__ = 'sbnb@irbbarcelona.org'
__version__ = '1.0.1'
