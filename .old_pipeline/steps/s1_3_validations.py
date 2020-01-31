from chemicalchecker.util import logged
from chemicalchecker.core import Validation, ChemicalChecker
from chemicalchecker.util import BaseTask
from chemicalchecker.util import Config


@logged
class Validations(BaseTask):

    def __init__(self, config, name, **params):

        BaseTask.__init__(self, config, name, **params)

    def run(self):
        """Run the validations step."""

        config = Config()

        cc = ChemicalChecker(config.PATH.CC_ROOT)

        vals = self.config.STEPS[self.name].sets

        for val_name in vals:

            if self.is_ready(val_name):
                continue

            self.__log.info("Generating validation set " + val_name)

            val = Validation(cc.get_validation_path(), val_name)

            try:
                val.run()

            except Exception as ex:
                self.__log.debug(ex)
                raise Exception("Validation %s not working" % val_name)

            self.mark_ready(val_name)

        self.mark_ready()
