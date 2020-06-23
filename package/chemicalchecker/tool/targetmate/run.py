import os

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
                 overwrite,
                 **kwargs):
        """Initialize multi-validation.

        Args:
            data(list): List of data files.
            models_path(list): Parent directory where to store the results.
            datasets(list or list of lists): Datasets to be used.
            overwrite(bool): Overwrite results. Otherwise, checks if a validation.txt file has been written.
            **kwargs: Arguments of the Validation class.
        """
        self.__log.info("Initialize multiple validation")
        self.overwrite = overwrite
        MultiSetup.__init__(self, data=data, models_path=models_path, **kwargs)
        self.validation = Validation(**kwargs)

    def _is_done(self, models_path):
        if os.path.exists(os.path.join(models_path, "validation.txt")):
            self.__log.info("Validation exists in %s" % models_path)
            return True
        else:
            return False

    def run(self, TargetMate, use_cc, use_inchikey, **kwargs):
        if not use_cc:
            self.__log.info("Precalculating signatures of merged data")
            sign_paths = self.precalc_signatures(**kwargs)
        else:
            sign_paths = None
        self.__log.info("Multiple trainings")
        tm_list   = []
        data_list = []
        for data, models_path, _ in self.tasks:
            if not self.overwrite:
                if self._is_done(models_path):
                    continue
            tm  = TargetMate(models_path=models_path, use_cc=use_cc, use_inchikey=use_inchikey, master_sign_paths=sign_paths, **kwargs)
            dat = tm.get_data_fit(data, inchikey_idx=-1, use_inchikey=use_inchikey)
            if dat is not None:
                tm_list   += [tm.on_disk()]
                data_list += [dat.on_disk(tm.tmp_path)]
        if len(tm_list) > 0:
            self.validation.validate(tm=tm_list, data=data_list)

class MultiFit(MultiSetup):
    pass


class MultiPredict(MultiSetup):
    pass