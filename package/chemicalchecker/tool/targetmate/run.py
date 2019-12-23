from chemicalchecker.util import logged

from .evaluation import Validation
from .utils.tasks import Tasker
from .multi import MultiSetup

@logged
class MultiValidation(MultiSetup):
    """Helper class to create multiple validations."""
    def __init__(self,
                 data,
                 models_path,
                 **kwargs):
        """Initialize multi-validation.

        Args:
            data(list): List of data files.
            models_path(list): Parent directory where to store the results.
            datasets(list or list of lists): Datasets to be used.
            **kwargs: Arguments of the Validation class.
        """
        self.__log.info("Initialize multiple validation")
        MultiSetup.__init__(self, data=data, models_path=models_path, **kwargs)
        self.validation = Validation(**kwargs)

    def run(self, TargetMate, **kwargs):
        self.precalc_signatures(**kwargs)
        self.__log.info("Multiple trainings")
        tm_list   = []
        data_list = []
        for data, models_path, _ in self.tasks:
            tm = TargetMate(models_path=models_path, **kwargs)
            tm_list   += [tm]
            data_list += [tm.get_data_fit(data)]
        self.validation.validate(tm=tm_list, data=data_list)


class MultiFit(MultiSetup):
    pass


class MultiPredict(MultiSetup):
    pass