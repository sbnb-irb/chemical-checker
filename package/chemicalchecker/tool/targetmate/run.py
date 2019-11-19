from .evaluation import Validation
from .utils.tasks import Tasker


class MultiValidation(object):
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
        self.validation = Validation(**kwargs)
        self.tasks = Tasker(data, models_path)

    def run(self, TargetMate, **kwargs):
        tm_list   = []
        data_list = []
        for data, models_path, _ in self.tasks:
            tm = TargetMate(models_path=models_path, **kwargs)
            tm_list   += [tm]
            data_list += [tm.get_data_fit(data)]
        self.validation.validate(tm=tm_list, data=data_list)


class MultiFit(object):
    pass


class MultiPredict(object):
    pass