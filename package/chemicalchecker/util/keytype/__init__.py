"""Detect type of compound keys automatically.

Recognized types are:
   * InChI
   * InChIKey
   * SMILES

When checking the a list of keys homogeneity of key type across
the list is assumed.
"""
from .detect import KeyTypeDetector