import tempfile

from chemicalchecker.util import logged
from chemicalchecker.database import Molprop, Datasource
from chemicalchecker.database import Molrepo
from chemicalchecker.util import BaseStep


@logged
class Molproperties(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        final_ik_inchi = set()
        all_molrepos = Datasource.get_molrepos(only_updatable=True)
        molrepos_names = set()
        for ds in all_molrepos:
            molrepos_names.add(ds.molrepo_name)

        molprops = self.config.STEPS[self.name].molprops

        for molrepo in molrepos_names:
            print(molrepo)
            molrepo_ik_inchi = Molrepo.get_fields_by_molrepo_name(
                molrepo, ["inchikey", "inchi"])
            final_ik_inchi.update(molrepo_ik_inchi)

        iks = set()

        for ik in final_ik_inchi:
            iks.add(ik[0])

        for mol in molprops:

            if self.is_ready(mol):
                continue

            self.__log.info("Calculating mol properties for " + mol)

            job_path = tempfile.mkdtemp(
                prefix='jobs_molprop_' + mol + "_", dir=self.tmpdir)

            molprop = Molprop(mol)

            # This method sends the job and waits for the job to finish
            molprop.molprop_hpc(job_path, list(final_ik_inchi))
            missing = len(molprop.get_missing_from_set(iks))
            if missing > 0:
                self.__log.error("Not all molecular properties were calculated. There are " +
                                 str(missing) + " missing out of " + str(len(iks)))
                return
            else:
                self.mark_ready(mol)

        self.mark_ready()
