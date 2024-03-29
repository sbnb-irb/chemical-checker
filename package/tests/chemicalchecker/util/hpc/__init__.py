"""Submit jobs to an HPC cluster.

A lot of processes in this package are very computationally intensive.
This class allows to send any task to any HPC environmment.
According to the config parameters this class will send the job
in the right format to the specified queueing technology.
The support queing system are:

  * **SGE**: :mod:`~chemicalchecker.util.hpc.sge` for Sun Grid Engine
  * **SLURM**: :mod:`~chemicalchecker.util.hpc.slurm` for Slurm Workload
  * **SLURM**: :mod:`~chemicalchecker.util.hpc.slurm_gpu` for Slurm Workload using GPU server
    Manager
  * **local**: :mod:`~chemicalchecker.util.hpc.local` for local processes.
"""
from .hpc import HPC
