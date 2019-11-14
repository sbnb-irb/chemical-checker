from .utils import HPCUtils
from .evaluation import Validation

class MultiValidate(HPCUtils):
    """Helper class to run multiple validations"""

    def __init__(self, TargetMate, hpc=True, **kwargs):
        """XXX"""
        HPCUtils.__init__(self, **kwargs)
        self.TargetMate = TargetMate
        self.hpc = hpc

    def prepare(self, data, models_path, datasets, **kwargs):
        tm = self.TargetMate(models_path = models_path, datasets = datasets, **kwargs)
        data = tm.get_data_fit(data = data)
        validation = Validation(**kwargs)
        return tm, data, validation
    
    def _run(self, tm, data, validation):
        validation.validate(tm, data)

    def run(self, tasks, wait=False, **kwargs):
        """
        tasks(list): List of task to do, expressed as [(data, models_path, datasets)]
        """
        jobs = []
        for data, models_path, datasets in tasks:
            tm, data, validation = self.prepare(data, models_path, datasets, **kwargs)
            if self.hpc:
                jobs += self.func_hpc("_run", tm, data, validation)
            else:
                self._run(tm, data, validation)
        if wait:
            self.waiter(jobs)
