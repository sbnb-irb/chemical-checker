"""SGE interace to send jobs to an HPC cluster

"""
import os
import paramiko


from chemicalchecker.util import logged


@logged
class sge():
    """Send tasks to an HPC cluster through SGE queueing system."""

    def __init__(self, config):
        """Initialize the SGE object.

        """
        self.host = config.HPC.host
        try:
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(self.host)
        except paramiko.SSHException as sshException:
            self.__log.warning(
                "Unable to establish SSH connection: %s" % sshException)
        finally:
            ssh.close()

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
            memory:Maximum memory the job can take in Gigabytes(default: 2)
            time: Maximum time the job can run on the cluster(default:infinite)

        """

        # get arguments or default values
        num_jobs = kwargs.get("num_jobs", 1)
        cpu = kwargs.get("cpu", 1)
        wait = kwargs.get("wait", True)
        jobdir = kwargs.get("jobdir", '')
        job_name = kwargs.get("job_name", 'hpc_cc_job')
        elements = kwargs.get("elements", [])
        compress = kwargs.get("compress", True)
        check_error = kwargs.get("check_error", True)
        memory = kwargs.get("memory", 2)
        time = kwargs.get("time", None)

        self.__log.debug("Job name is: " + job_name)
