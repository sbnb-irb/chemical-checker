"""Signature type 0 are basically raw features. Each bioactive space has
a peculiar format which might be categorial, discrete or continuous.
"""
import os
import imp
import h5py
import numpy as np
from tqdm import tqdm

from .signature_data import DataSignature
from .signature_base import BaseSignature

from chemicalchecker.util import logged
