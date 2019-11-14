import os

from .utils import HPCUtils
from .evaluation import Validation

class Tasker:
    """Generate tasks to run."""
    def __init__(self, data, models_path, datasets):
        """
        Initialize tasker.
        
        Args:
            data:
            models_path(list or str): 
            datasets(list of lists, list or str): (default=None).
        """
        # Data
        self.data = []
        if type(data) == str:
            self.data += [os.path.abspath(data)]
        else:
            for d in data:
                if type(d) == str:
                    self.data += [os.path.abspath(d)]
                else:
                    self.data += [d]
        # CC datasets
        self.datasets = []
        if datasets is None:
            for _ in self.data:
                self.datasets += [None]
        else:
            if type(datasets) == str:
                for _ in self.data:
                    self.datasets += [[datasets]]
            else:
                if type(datasets[0]) == str:
                    for _ in self.data:
                        self.datasets += [datasets]
                else:
                    self.datasets = datasets
        # Models
        self.models_path = []
        if type(models_path) == str:
            models_path = os.path.abspath(models_path)
            for i, d in enumerate(self.data):
                if type(d) == str:
                    fn = ".".join(d.split("/")[-1].split(".")[:-1])
                    if self.datasets[i] is not None:
                        fn = fn + "---" + "-".join(self.datasets[i])
                    self.models_path += [os.path.join(models_path, fn)]
                else:
                    fn = "%02d" % i
                    if self.datasets[i] is not None:
                        fn = fn + "---" + "-".join(self.datasets[i])
                    self.models_path += [os.path.join(models_path)]
        else:
            for m in models_path:
                self.models_path += [os.path.abspath(m)]
        # Assert
        assert len(self.data) == len(self.models_path) == len(self.datasets), "Wrong tasks specified."

    def __iter__(self):
        for i, data in enumerate(self.data):
            yield data, self.models_path[i], self.datasets[i]


class MultiValidate(HPCUtils):
    """Helper class to run multiple validations"""

    def __init__(self,
                 TargetMate,
                 data,
                 models_path,
                 datasets=None,
                 hpc=True, **kwargs):
        """XXX"""
        HPCUtils.__init__(self, **kwargs)
        self.TargetMate = TargetMate
        self.hpc = hpc
        self.tasks = Tasker(data, models_path, datasets)

    def prepare(self, data, models_path, datasets, **kwargs):
        tm = self.TargetMate(models_path = models_path, datasets = datasets, **kwargs)
        data = tm.get_data_fit(data = data)
        validation = Validation(**kwargs)
        return tm, data, validation
    
    def _run(self, tm, data, validation):
        validation.validate(tm, data)

    def run(self, wait=False, **kwargs):
        """
        tasks(list): List of task to do, expressed as [(data, models_path, datasets)]
        """
        jobs = []
        for data, models_path, datasets in self.tasks:
            tm, data, validation = self.prepare(data, models_path, datasets, **kwargs)
            if self.hpc:
                jobs += [self.func_hpc("_run", tm, data, validation)]
            else:
                self._run(tm, data, validation)
        if wait:
            self.waiter(jobs)
