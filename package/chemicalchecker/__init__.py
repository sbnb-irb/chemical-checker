"""**Welcome to the Chemical Checker Package documentation!**

The chemical checker package aims to provide an interface to use all chemical
checker tools. From the generation of new data (signatures, etc) to accessing
already produced data.

 Organization
    The package is organized in four subpackages:

        * :mod:`package.core`: The ``core`` module holds the main Chemical
          Checker functionalities. It holds the ChemicalChecekr entry-point,
          the Signature definitions and ``HDF5`` implementation.

        * :mod:`package.database`: The ``database`` module include DB access
          and table definitions.

        * :mod:`package.tool`: The ``tool`` modules gather experimental code
          and wrappers to external software.

        * :mod:`package.util`: The ``util`` module pull together general
          utilities.
"""

from .core import ChemicalChecker


__author__ = """SBNB"""
__email__ = 'sbnb@irbbarcelona.org'
__version__ = '0.1.0'
