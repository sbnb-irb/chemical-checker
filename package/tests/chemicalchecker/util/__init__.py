"""Utilities used throughout the code.

This module expose several utilities that are broadly used throughout the
Chemical Checker code (e.g. :mod:`~chemicalchecker.util.logging` for handling
output or :mod:`~chemicalchecker.util.hpc` to submit jobs to the HPC cluster).
"""
try:
    # FIXME(https://github.com/abseil/abseil-py/issues/99)
    # FIXME(https://github.com/abseil/abseil-py/issues/102)
    # Unfortunately, many libraries that include absl (including Tensorflow)
    # will get bitten by double-logging due to absl's incorrect use of
    # the python logging library:
    #   2019-07-19 23:47:38,829 my_logger   779 : test
    #   I0719 23:47:38.829330 139904865122112 foo.py:63] test
    #   2019-07-19 23:47:38,829 my_logger   779 : test
    #   I0719 23:47:38.829469 139904865122112 foo.py:63] test
    # The code below fixes this double-logging.  FMI see:
    #   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
    import logging
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    pass
from .logging.our_logging import *
from autologging import logged
from .config import Config
from .hpc import HPC
