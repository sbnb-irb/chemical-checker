"""Utility for sending tasks to a HPC cluster

A lot of processes in this package are very computationally intensive.
This class allows to send any task to any HPC environmment.
According to the config parameters this class will send the tasks
in the right format to the specified queueing technology.
"""
import os

from .sge import sge
from .slurm import slurm

from chemicalchecker.util import logged


@logged
class HPC():
    """Send tasks to an HPC cluster."""

    STARTED = "started"
    DONE = "done"
    READY = "ready"
    ERROR = "error"

    def __init__(self, **kwargs):
        """Initialize the HPC object.

        Args:
            system:Queuing HPC system (default:'')
            host:Name of the HPC host master(default:'')
            queue:Name of the queue (default:'')
            username:Username to connect to the host (default:'')
            password:Password to connect (default:'')
            error_finder: Method to search errors in HPC jobs log(default:None)
            dry_run:Only for test checks (default=False)

        """
        self.system = kwargs.get("system", '')

        self.__log.debug('HPC system to use: %s', self.system)
        self.job_id = None

        if self.system == '':
            raise Exception('HPC system not specified')

        if self.system in globals():
            self.__log.debug("initializing object %s", self.system)
            self.hpc = eval(self.system)(**kwargs)
        else:
            raise Exception("HPC system %s not available" % self.system)

    @classmethod
    def from_config(cls, config):
        if "HPC" in config.keys():
            return cls(**config.HPC.asdict())
        else:
            raise Exception("Config does not contain HPC fields")

    def submitMultiJob(self, command, **kwargs):
        """Submit a multi job/task.

         Args:
            command: The comand that will be executed in the cluster. It should contain a
                     <TASK_ID> string and a <FILE> string. This will be replaced but
                     the correponding task id and the pickle file with the elements that
                     the command will need.
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

        if self.job_id is None:
            self.job_id = self.hpc.submitMultiJob(command, **kwargs)
        else:
            raise Exception("HPC instance already in use")

    def check_errors(self):
        """Check for errors in the output logs of the jobs.

        If there are no errors and the status is "done", the status will change to "ready".

        Returns:
            errors(str): The lines in the output logs where the error is found.
                        The format of the errors is filename, line number and line text.
                        If there are no errors it returns None.

        """

        return self.hpc.check_errors()

    def compress(self):
        """Compress the output logs into a tar.gz file in the same job directory


        """

        self.hpc.compress()

    def status(self):
        """Gets the status of the job submission

           The status is None if there is no job submission.
           The status is also saved in a *.status file in the job directory.

        Returns:
            status(str): There are three possible status if there was a submission
                         "started": Job started but not finished
                         "done": Job finished
                         "ready": Job finished without errors
        """

        return self.hpc.status()
