import os


class Tasker:
    """Generate tasks to run."""
    def __init__(self, data, models_path, datasets=None):
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
                    # --------------------------------------
                    #    THIS IS SPECIFIC TO CHEMBL SETUP
                    #
                    fnt = d.split("/")[-2]
                    fn = ".".join(d.split("/")[-1].split(".")[:-1])
                    # --------------------------------------
                    # fn = ".".join(d.split("/")[-1].split(".")[:-1])
                    if self.datasets[i] is not None:
                        fn = fn + "---" + "-".join(self.datasets[i])
                    self.models_path += [os.path.join(models_path, fnt, fn)] ## Added by Paula: Create first target folder then threshold 27/08/20
                    # self.models_path += [os.path.join(models_path, fn)] ## Added by Paula: Create first target folder then threshold 27/08/20


                else:
                    fn = "%02d" % i
                    if self.datasets[i] is not None:
                        fn = fn + "---" + "-".join(self.datasets[i])
                    self.models_path += [os.path.join(models_path)]

        elif type(models_path) == list or type(models_path) == tuple: ## Added by Paula: Explicitly state output paths 27/08/20
            assert len(self.data) == len(models_path), "Wrong tasks specified."
            for m, d in zip(models_path, self.data):
                f = ".".join(d.split("/")[-1].split(".")[:-1])
                m = os.path.join(m, f)
                mp = os.path.abspath(m)
                self.models_path += [os.path.join(mp)]
        # Assert
        assert len(self.data) == len(self.models_path) == len(self.datasets), "Wrong tasks specified. %d , %d, %d" % (len(self.data), len(self.models_path) , len(self.datasets))


    def __iter__(self):
        for i, data in enumerate(self.data):
            yield data, self.models_path[i], self.datasets[i]
