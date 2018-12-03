"""Utility for sending tasks to and HPC cluster

A lot of processes in this package are very computationally intensive. 
this class allows to send any task to and HPC environmment. 
According to the config parameters this class will send the tasks 
in the right format to the specified queueing technology.
"""
import os

from .sge import sge

from chemicalchecker.util import logged


@logged
class HPC():
    """Send tasks to an HPC cluster."""

    def __init__(self, config):
        """Initialize the HPC object.

        """
        self.__log.debug('HPC system to use: %s', config.HPC.system)

        if config.HPC.system in globals():
            self.__log.debug("initializing object %s", config.HPC.system)
            self.hpc = eval(config.HPC.system)(config)
        else:
            raise Exception("HPC system %s not available" % config.HPC.system)

    def submitMultiJob(self, command, **kwargs):
        """Submit a multi job/task.

         Args:
            num_jobs:Number of jobs to run the command (default:1)
            cpu:Number of cores the job will use(default:1)
            wait:Wait for the job to finish (default:True)
            jobdir:Directotory where the job will run (default:'')
            job_name:Name of the job (default:10)
            elements:List of elements that will need to run on the command
            compress:Compress all generated files after job is done (default:True)
            check_error:Check for error message in output files(default:True)
            memory:Maximum memory the job can take kin Gigabytes(default: 2)
            time: Maximum time the job can run on the cluster(default:infinite)

        """

        self.hpc.submitMultiJob(command, **kwargs)
