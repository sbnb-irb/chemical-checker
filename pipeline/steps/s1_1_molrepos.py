import tempfile

from chemicalchecker.util import logged
from chemicalchecker.database import Molrepo
from chemicalchecker.util import BaseStep


@logged
class Molrepos(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molrepos step."""

        job_path = tempfile.mkdtemp(
            prefix='jobs_molrepos_', dir=self.tmpdir)
        # start molrepo jobs (one per Datasource), job will wait until finished
        Molrepo.molrepo_hpc(job_path, only_updatable=True)

        # check if the molrepos are really done
        if not Molrepo.test_all_available():
            self.__log.error("Something went WRONG while MOLREPO, should retry")
        else:
            self.mark_ready()
