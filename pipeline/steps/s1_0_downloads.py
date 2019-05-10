import tempfile

from chemicalchecker.util import logged
from chemicalchecker.database import Datasource
from chemicalchecker.util import HPC
from chemicalchecker.util import BaseStep


@logged
class Downloads(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the downloads step."""

        job_path = tempfile.mkdtemp(
            prefix='jobs_download_', dir=self.tmpdir)
        # start download jobs (one per Datasource), job will wait until
        # finished
        job = Datasource.download_hpc(job_path, only_essential=True)

        if job.status() == HPC.ERROR:
            self.__log.warning(
                "There are errors in some of the downloads jobs")

        # check if the downloads are really done
        if not Datasource.test_all_downloaded(only_essential=True):
            self.__log.warning(
                "Something went WRONG while DOWNLOAD, should retry")
            # print the faulty one
            missing_datasources = set()
            for ds in Datasource.get():
                for dset in ds.datasets:
                    if dset.essential:
                        missing_datasources.add(ds)
                        break
                for molrepo in ds.molrepos:
                    if molrepo.essential:
                        missing_datasources.add(ds)
                        break
            for ds in missing_datasources:
                if not ds.available:
                    self.__log.error("Datasource %s not available" % ds)

        else:
            self.mark_ready()
