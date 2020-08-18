"""Utility for submitting jobs to a HPC cluster.

A lot of processes in this package are very computationally intensive.
This class allows to send any task to any HPC environmment.
According to the config parameters this class will send the job
in the right format to the specified queueing technology.
The support queing system are:

  * **SGE (Sun Grid Engine)**: :mod:`~chemicalchecker.util.hpc.sge`
  * **SLURM**: :mod:`~chemicalchecker.util.hpc.slurm`
  * **local**: :mod:`~chemicalchecker.util.hpc.local`
"""
from .hpc import HPC
