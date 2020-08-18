"""**Welcome to the Chemical Checker Package documentation!**

The Chemical Checker Package aims to provide an interface to use all Chemical
Checker tools. From the generation of new data (signatures, etc) to accessing
already produced data.

 Organization
    The package is organized in four subpackages:

        * :mod:`~chemicalchecker.core`: The ``core`` module holds the mains
          Chemical Checker functionalities. It holds the
          :class:`ChemicalChecker`
          entry-point, the Signature classes and ``HDF5`` i/o
          implementation.

        * :mod:`~chemicalchecker.database`: The ``database`` module include DB
          access and table definitions.

        * :mod:`~chemicalchecker.tool`: The ``tool`` modules gather
          experimental code and wrappers to external software.

        * :mod:`~chemicalchecker.util`: The ``util`` module pull together
          general utilities.
"""
from .core import ChemicalChecker


__author__ = """SBNB"""
__email__ = 'sbnb@irbbarcelona.org'
__version__ = '0.1.0'
