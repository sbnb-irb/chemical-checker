"""HPC factory.

Allow the initialization of any of the defined queueing systems.
Provide a shared interface to get job status, check log for errors, and
compress log output.
"""
from .sge import sge
from .slurm import slurm
from .local import local

from chemicalchecker.util import logged


@logged
class HPC():
    """HPC factory class."""

    STARTED = "started"
    DONE = "done"
    READY = "ready"
    ERROR = "error"

    def __init__(self, **kwargs):
        """Initialize a HPC instance.

        Args:
            system (str): Queuing HPC system. (default: '')
            host (str): Name of the HPC host master. (default: '')
            queue (str): Name of the queue. (default: '')
            username (str): Username to connect to the host. (default: '')
            password (str): Password to connect. (default: '')
            error_finder (func): Method to search errors in HPC jobs log.
                (default: None)
            dry_run (bool): Only for test checks. (default=False)

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
        """Submit multiple job/task.

         Args:
            command (str): The comand that will be executed in the cluster.
                It should contain a <TASK_ID> string and a <FILE> string.
                This will be replaced but the correponding task id and the
                pickle file with the elements that the command will need.
            num_jobs (int): Number of jobs to run the command. (default: 1)
            cpu (int): Number of cores the job will use. (default: 1)
            wait (bool): Wait for the job to finish. (default: True)
            jobdir (str): Directotory where the job will run. (default: '')
            job_name (str): Name of the job. (default: 10)
            elements (list): List of elements that will need to run on the
                command.
            compress (bool): Compress all generated files after job is done.
                (default: True)
            check_error (bool): Check for error message in output files.
                (default: True)
            memory (int): Maximum memory the job can take kin Gigabytes.
                (default:  2)
            time (int): Maximum time the job can run on the cluster.
                (default: infinite)

        """
        if self.job_id is None:
            self.job_id = self.hpc.submitMultiJob(command, **kwargs)
        else:
            raise Exception("HPC instance already in use")

    def check_errors(self):
        """Check for errors in the output logs of the jobs.

        If there are no errors and the status is ``done``, the status will
        change to ``ready``.

        Returns:
            errors (str): Lines in the output logs where the error is found.
            The format of the errors is filename, line number and line text.
            If there are no errors it returns None.

        """
        return self.hpc.check_errors()

    def compress(self):
        """Compress the output logs.

        Compress the output logs into a tar.gz file in the same job directory.
        """
        self.hpc.compress()

    def status(self):
        """Gets the status of the job submission.

       The status is None if there is no job submission.
       The status is also saved in a *.status file in the job directory.

        Returns:
            status (str): There are three possible statuses for a submission:

               * ``started``: Job started but not finished
               * ``done``: Job finished
               * ``ready``: Job finished without errors
        """
        return self.hpc.status()
