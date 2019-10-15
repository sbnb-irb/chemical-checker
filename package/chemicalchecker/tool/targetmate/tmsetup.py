@logged
class TargetMateSetup(HPCUtils):
    """Set up the base TargetMate class"""

    def __init__(self,
                 models_path,
                 tmp_path = None,
                 cc_root = None,
                 overwrite = True,
                 n_jobs = None,
                 n_jobs_hpc = 8,
                 standardize = True,
                 applicability = True,
                 hpc = False,
                 **kwargs):
        """Basic setup of the TargetMate.

        Args:
            models_path(str): Directory where models will be stored.
            tmp_path(str): Directory where temporary data will be stored
                (relevant at predict time) (default=None)
            cc_root(str): CC root folder (default=None)
            overwrite(bool): Clean models_path directory (default=True)
            n_jobs(int): Number of CPUs to use, all by default (default=None)
            n_jobs(hpc): Number of CPUs to use in HPC (default=8)
            standardize(bool): Standardize small molecule structures (default=True)
            cv(int): Number of cv folds (default=5)
            applicability(bool): Perform applicability domain calculation (default=True)
            hpc(bool): Use HPC (default=False)
        """
        HPCUtils.__init__(self, **kwargs)
        # Jobs
        if not n_jobs:
            self.n_jobs = self.cpu_count()
        else:
            self.n_jobs = n_jobs
        # Models path
        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(self.models_path):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.models_path)
            os.mkdir(self.models_path)
        else:
            if overwrite:
                # Cleaning models directory
                self.__log.debug("Cleaning %s" % self.models_path)
                shutil.rmtree(self.models_path)
                os.mkdir(self.models_path)
        self.bases_models_path, self.signatures_models_path = self.directory_tree(self.models_path)
        # Temporary path
        if not tmp_path:
            self.tmp_path = os.path.join(
                Config().PATH.CC_TMP, str(uuid.uuid4()))
        else:
            self.tmp_path = os.path.abspath(tmp_path)
        if not os.path.exists(self.tmp_path): os.mkdir(self.tmp_path)
        self.bases_tmp_path, self.signatures_tmp_path = self.directory_tree(self.tmp_path)
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker(cc_root)
        # Standardize
        self.standardize = standardize
        # Do applicability
        self.applicability = applicability
        # Use HPC
        self.n_jobs_hpc = n_jobs_hpc
        self.hpc = hpc
        # Others
        self._is_fitted  = False
        self._is_trained = False
        self.is_tmp      = False

    # Chemistry functions
    @staticmethod
    def read_smiles(smi, standardize):
        return chemistry.read_smiles(smi, standardize)

    # Other functions
    @staticmethod
    def directory_tree(root):
        bases_path = os.path.join(root, "bases")
        if not os.path.exists(bases_path): os.mkdir(bases_path)
        signatures_path = os.path.join(root, "signatures")
        if not os.path.exists(signatures_path): os.mkdir(signatures_path)
        return bases_path, signatures_path

    @staticmethod
    def avg_and_std(values, weights=None):
        """Return the (weighted) average and standard deviation.

        Args:
            values(list or array): 1-d list or array of values
            weights(list or array): By default, no weightening is applied
        """
        if weights is None:
            weights = np.ones(len(values))
        average = np.average(values, weights=weights)
        variance = np.average((values - average)**2, weights=weights)
        return (average, math.sqrt(variance))

    # Read input data
    def read_data(self, data, standardize=None):
        if not standardize:
            standardize = self.standardize
        # Read data
        self.__log.info("Reading data")
        # Read data if it is a file
        if type(data) == str:
            self.__log.info("Reading file %s", data)
            with open(data, "r") as f:
                data = []
                for r in csv.reader(f, delimiter="\t"):
                    data += [[r[0]] + r[1:]]
        # Get only valid SMILES strings
        self.__log.info(
            "Parsing SMILES strings, keeping only valid ones for training.")
        data_ = []
        for i, d in enumerate(data):
            m = self.read_smiles(d[1], standardize)
            if not m:
                continue
            # data is always of [(initial index, activity, smiles,
            # inchikey)]
            data_ += [[i, float(d[0])] + [m[1], m[0]]]
        data = data_
        return InputData(data)

    # Loading functions
    @staticmethod
    def load(models_path):
        """Load previously stored TargetMate instance."""
        with open(os.path.join(models_path, "/TargetMate.pkl", "r")) as f:
            return pickle.load(f)

    def load_performances(self):
        """Load performance data"""
        fn = os.path.join(self.models_path, "perfs.json")
        if not os.path.exists(fn): return
        with open(fn, "r") as f:
            return json.load(f)

    def load_ad_data(self):
        """Load applicability domain data"""
        fn = os.path.join(self.models_path, "ad_data.pkl")
        if not os.path.exists(fn): return
        with open(fn, "rb") as f:
            return pickle.load(f)

    def load_data(self):
        self.__log.debug("Loading training data (only evidence)")
        fn = os.path.join(self.models_path, "trained_data.pkl")
        if not os.path.exists(fn): return
        with open(fn, "rb") as f:
            return pickle.load(f)

    def load_base_model(self, destination_dir, append_pipe=False):
        """Load a base model"""
        mod = joblib.load(destination_dir)
        if append_pipe:
            self.pipes += [pickle.load(open(destination_dir+".pipe", "rb"))]
        return mod

    # Saving functions
    def save(self):
        """Save TargetMate instance"""
        # we avoid saving signature instances
        self.sign_predict_fn = None
        with open(self.models_path + "/TargetMate.pkl", "wb") as f:
            pickle.dump(self, f)

    def save_performances(self, perfs):
        with open(self.models_path + "/perfs.json", "w") as f:
            json.dump(perfs, f)

    def save_data(self, data):
        self.__log.debug("Saving training data (only evidence)")
        with open(self.models_path + "/trained_data.pkl", "wb") as f:
            pickle.dump(data, f)
