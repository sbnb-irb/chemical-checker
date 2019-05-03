import tempfile

from chemicalchecker.util import logged
from chemicalchecker.database import Calcdata
from chemicalchecker.database import Molrepo
from chemicalchecker.util import BaseStep


@logged
class Calculatedata(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        final_ik_inchi = set()
        all_molrepos = Molrepo.get()
        molrepos_names = set()
        for molrepo in all_molrepos:
            molrepos_names.add(molrepo.molrepo_name)

        datas = self.config.STEPS[self.name].data

        for molrepo in molrepos_names:
            print(molrepo)
            molrepo_ik_inchi = Molrepo.get_fields_by_molrepo_name(
                molrepo, ["inchikey", "inchi"])
            final_ik_inchi.update(molrepo_ik_inchi)

        iks = set()

        for ik in final_ik_inchi:
            iks.add(ik[0])

        for data in datas:

            if self.is_ready(data):
                continue

            self.__log.info("Calculating data for " + data)

            job_path = tempfile.mkdtemp(
                prefix='jobs_molprop_' + data + "_", dir=self.tmpdir)

            calculator = Calcdata(data)

            # This method sends the job and waits for the job to finish
            calculator.calcdata_hpc(job_path, list(final_ik_inchi))
            missing = len(calculator.get_missing_from_set(iks))
            if missing > 0:
                self.__log.error("Not all molecular properties were calculated. There are " +
                                 str(missing) + " missing out of " + str(len(iks)))
                return
            else:
                self.mark_ready(data)

        self.mark_ready()
