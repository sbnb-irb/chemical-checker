@logged
class Old:
    """TargetMate class"""

    def __init__(self, models_path, tmp_path=None,
                 base_mod="logistic_regression", cv=5, k=5, min_sim=0.25,
                 min_class_size=10,
                 inactives_per_active=100,
                 datasets=None, metric="bedroc",
                 cc_root=None, universe_path=None, sign=None, sign_predict_fn=None,
                 n_jobs=None, naive_sampling=False, applicability=True):
        """Initialize the TargetMateEnsembleClassifier class

        Args:
            models_path(str): Directory where models will be stored.
            tmp_path(str): Directory where temporary data will be stored
                (relevant at predict time) (default=None)
            base_mod(clf): Classifier instance, containing fit and
                predict_proba methods (default="tpot")
                The following strings are also accepted: "logistic_regression",
                "random_forest", "naive_bayes" and "tpot"
                By default, sklearn LogisticRegressionCV is used.
            cv(int): Number of cv folds. The default cv generator used is
                Stratified K-Folds (default=5).
            k(int): Number of molecules to look across when doing the
                applicability domain (default=5).
            min_sim(float): Minimum Tanimoto chemical similarity to consider in
                the applicability domain determination (default=0.3).
            min_class_size(int): Minimum class size acceptable to train the
                classifier (default=10).
            inactives_per_active(int): Number of inactive to sample for each active.
                If None, only experimental actives and inactives are considered (default=100).
            datasets(list): CC datasets (A1.001-E5.999).
                By default, all datasets having a SMILES-to-sign predictor are
                used.
            metric(str): Metric to use in the meta-prediction (bedroc, auroc or aupr)
                (default="bedroc").
            cc_root(str): CC root folder (default=None).
            universe_path(str): Path to the universe. If not specified, the default one is used (default=None).
            sign_predict_fn(dict): pre-loaded predict_fn, keys are dataset
                codes, values are tuples of (sign, predict_fn).
            naive(bool): Sample naively (randomly), without using the OneClassSVM (default=False).
        """
        # Jobs
        if not n_jobs:
            self.n_jobs = hpc.cpu_count()
        else:
            self.n_jobs = n_jobs
        # Models path
        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.models_path)
            os.mkdir(self.models_path)
        # Temporary path
        if not tmp_path:
            import uuid
            self.tmp_path = os.path.join(
                Config().PATH.CC_TMP, str(uuid.uuid4()))
        else:
            self.tmp_path = os.path.abspath(tmp_path)
        # Set the base classifier
        if type(base_mod) == str:
            if base_mod == "logistic_regression":
                from sklearn.linear_model import LogisticRegressionCV
                self.base_mod = LogisticRegressionCV(
                    cv=3, class_weight="balanced", max_iter=1000,
                    n_jobs=self.n_jobs)
            if base_mod == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                self.base_mod = RandomForestClassifier(
                    n_estimators=100, class_weight="balanced", n_jobs=self.n_jobs)
            if base_mod == "naive_bayes":
                from sklearn.naive_bayes import GaussianNB
                self.base_mod = Pipeline(
                    [('feature_selection', VarianceThreshold()),
                     ('classify', GaussianNB())])
            if base_mod == "tpot":
                from tpot import TPOTClassifier
                from models import tpotconfigs
                self.base_mod = TPOTClassifier(
                    config_dict=tpotconfigs.minimal,
                    generations=10, population_size=30,
                    cv=3, scoring="balanced_accuracy",
                    verbosity=2, n_jobs=self.n_jobs,
                    max_time_mins=5, max_eval_time_mins=0.5,
                    random_state=42,
                    early_stop=3,
                    disable_update_check=True
                )
                self._is_tpot = True
            else:
                self._is_tpot = False
        else:
            self.base_mod = base_mod
        # Crossvalidation to determine the performances of the individual
        # predictors
        self.cv = cv
        # K-neighbors to search during the applicability domain calculation
        self.k = k
        # Minimal chemical similarity to consider
        self.min_sim = min_sim
        # Minimum size of the minority class
        self.min_class_size = min_class_size
        # Inactives per active
        self.inactives_per_active = inactives_per_active
        # Initialize the ChemicalChecker
        self.cc = ChemicalChecker(cc_root)
        # Load universe
        self.universe = Universe.load_universe(universe_path)
        # Store the paths to the sign (only the ones that have been already
        # trained)
        if not datasets:
            #self.datasets = list(self.cc.datasets)
            self.datasets = ["%s%s.001" % (x, y)
                             for x in "ABCDE" for y in "12345"]
        else:
            self.datasets = datasets
        # preloaded neural networks
        if sign_predict_fn is None:
            self.sign_predict_fn = dict()
            for ds in self.datasets:
                self.__log.debug("Loading sign predictor for %s" % ds)
                s3 = self.cc.get_signature("sign", "full", ds)
                self.sign_predict_fn[ds] = (s3, s3.get_predict_fn())
        else:
            self.sign_predict_fn = sign_predict_fn
        # Metric to use
        self.metric = metric
        # naive_samplingsampling
        self.naive_sampling = naive_sampling
        # Others
        self._is_fitted  = False
        self._is_trained = False

    @staticmethod
    def load(models_path):
        """Load previously stored TargetMate instance."""
        with open(os.path.join(models_path, "/TargetMate.pkl", "r")) as f:
            return pickle.load(f)

    def load_performances(self):
        """Load performance data"""
        with open(os.path.join(self.models_path, "perfs.json"), "r") as f:
            return json.load(f)

    def load_ad_data(self):
        """Load applicability domain data"""
        with open(os.path.join(self.models_path, "ad_data.pkl"), "r") as f:
            return pickle.load(f)

    def plot(self):
        """Plot model analytics"""
        return plots.ensemble_classifier_grid(self.load_performances, self.load_ad_data)

    def _read_sign(self, dataset, idxs=None, sign_folder=None, is_prd=False):
        # Identify HDF5 file
        if not sign_folder:
            if is_prd:
                h5file = os.path.join(self.tmp_path, dataset)
            else:
                h5file = os.path.join(self.models_path, dataset)
        else:
            h5file = os.path.join(sign_folder, dataset)
        # Read the file
        with h5py.File(h5file, "r") as hf:
            if idxs is None:
                V = hf["V"][:]
            else:
                V = hf["V"][:][idxs]
        return V

    def _save_ensemble(self, clf_ensemble):
        if self._is_tpot:
            ensdir = os.path.join(self.models_path, "ensemble")
            if os.path.exists(ensdir):
                shutil.rmtree(ensdir)
            os.mkdir(ensdir)
            for dataset, clf in clf_ensemble:
                joblib.dump(clf, ensdir + "/" + dataset + ".sav")
        else:
            with open(self.models_path + "/ensemble.pkl", "wb") as f:
                pickle.dump(clf_ensemble, f)

    def _load_ensemble(self):
        if self._is_tpot:
            clf_ensemble = []
            ensdir = os.path.join(self.models_path, "ensemble")
            for dataset in self.datasets:
                clf_ensemble += [(dataset,
                                  joblib.load(
                                      ensdir + "/" + dataset + ".sav"))]
        else:
            with open(self.models_path + "/ensemble.pkl", "r") as f:
                clf_ensemble = pickle.load(f)
        return clf_ensemble

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

    @staticmethod
    def performances(yt, yp):
        """Calculate standard prediction performance metrics.
        In addition, it calculates the corresponding weights.
        For the moment, AUPR and AUROC are used.
        Args:
            yt(list): Truth data (binary).
            yp(list): Prediction scores (probabilities).
        """
        perfs = {}
        yt = list(yt)
        yp = list(yp)
        perfs["auroc"] = metrics.roc_score(yt, yp)
        perfs["aupr"] = metrics.pr_score(yt, yp)
        perfs["bedroc"] = metrics.bedroc_score(yt, yp)
        perfs["y_true"] = yt
        perfs["y_pred"] = yp
        return perfs

    def metapredict(self, yp_dict, perfs, dataset_universe=None):
        """Do meta-prediction based on dataset-specific predictions.
        Weights are given according to the performance of the individual
        predictors.
        Standard deviation across predictions is kept to estimate
        applicability domain.
        """
        if dataset_universe is None:
            dataset_universe = set(self.datasets)
        M = []
        w = []
        for dataset in self.datasets:
            if dataset not in dataset_universe:
                continue
            w += [perfs[dataset]["perf_test"]
                  [self.metric][1]]  # Get the weight
            M += [yp_dict[dataset]]
        M = np.array(M)
        w = np.array(w)
        prds = []
        stds = []
        for j in range(0, M.shape[1]):
            avg, std = self.avg_and_std(M[:, j], w)
            prds += [avg]
            stds += [std]
        return np.clip(prds, 0.001, 0.999), np.clip(stds, 0.001, None)

    def fingerprint_arena(self, smiles, use_checkpoints=False, is_prd=False):
        if is_prd:
            fps_file = os.path.join(self.tmp_path, "arena.fps")
        else:
            fps_file = os.path.join(self.models_path, "arena.fps")
        if not use_checkpoints or not os.path.exists(fps_file):
            self.__log.debug("Writing Fingerprints")
            arena = morgan_arena(smiles, fps_file)
        else:
            arena = load_morgan_arena(fps_file)
        return arena

    @staticmethod
    def calculate_bias(yt, yp):
        """Calculate the bias of the QSAR model"""
        return np.array([np.abs(t - p) for t, p in zip(yt, yp)])

    @staticmethod
    def read_smiles(smi, standardize):
        return read_smiles(smi, standardize)

    def knearest_search(self, query_smi, target_fps, N):
        """Nearest neighbors search using chemical fingerprints"""
        k = np.min([self.k, len(target_fps.fps[0]) - 1])
        neighs = []
        for smi in query_smi:
            results = target_fps.similarity(
                smi, self.min_sim, n_workers=self.n_jobs)
            neighs += [results[:k]]
        if N is None:
            N = len(query_smi)
        sims = np.zeros((N, k), dtype=float )
        idxs = np.zeros((N, k), dtype=int )
        for q_idx, hits in enumerate(neighs):
            sims_ = []
            idxs_ = []
            for h in hits:
                sims_ += [h[1]]
                idxs_ += [int(h[0])]
            sims[q_idx, :len(sims_)] = sims_
            idxs[q_idx, :len(idxs_)] = idxs_
        return sims, idxs

    def calculate_weights(self, query_smi, target_fps, stds, bias, N=None):
        """Calculate weights using adaptation of Aniceto et al 2016."""
        self.__log.debug("Finding nearest neighbors")
        sims, idxs = self.knearest_search(query_smi, target_fps, N)
        self.__log.debug("Calculating weights from std and bias")
        weights = []
        for i, idxs_ in enumerate(idxs):
            ws = sims[i] / (stds[idxs_] * bias[idxs_])
            weights += [np.max(ws)]
        return np.array(weights)

    @staticmethod
    def _reassemble_activity_sets(act, inact, putinact):
        data = []
        for x in list(act):
            data += [(x[1], 1, x[0], x[-1])]
        for x in list(inact):
            data += [(x[1], -1, x[0], x[-1])]
        n = np.max([x[0] for x in data]) + 1
        for i, x in enumerate(list(putinact)):
            data += [(i + n, 0, x[0], x[-1])]
        return data

    def save(self):
        # we avoid saving signature 3 objects
        self.sign_predict_fn = None
        with open(self.models_path + "/TargetMate.pkl", "wb") as f:
            pickle.dump(self, f)

    def fit(self, data, standardize=False, use_checkpoints=False):
        """
        Fit SMILES-activity data.
        Invalid SMILES are removed from the prediction.

        Args:
            data(str or list of tuples):
            standardize(bool): If True, SMILES strings will be standardized
                (default=False)
            use_checkpoints(bool): Store signature files and others that can
                be re-utilized (default=False)
        """
        if not use_checkpoints:
            # Cleaning models directory
            self.__log.debug("Cleaning previous checkpoints")
            shutil.rmtree(self.models_path)
            os.mkdir(self.models_path)
        # Read data
        self.__log.info("Reading data")
        # Read data if it is a file
        if type(data) == str:
            self.__log.info("Reading file %s", data)
            with open(data, "r") as f:
                data = []
                for r in csv.reader(f, delimiter="\t"):
                    data += [[int(r[0])] + r[1:]]
        # Get only valid SMILES strings
        self.__log.info(
            "Parsing SMILES strings, keeping only valid ones for training.")
        data_ = []
        for i, d in enumerate(data):
            m = self.read_smiles(d[1], standardize)
            if not m:
                continue
            # data is always of [(initial index, activity, ..., smiles,
            # inchikey)]
            data_ += [[i, int(d[0])] + [m[1], m[0]]]
        data = data_
        # Save training data
        self.__log.debug("Saving training data (only evidence)")
        with open(self.models_path + "/trained_data.pkl", "wb") as f:
            pickle.dump(data, f)
        # Sample inactives, if necessariy
        if self.inactives_per_active:
            self.__log.info("Sampling putative inactives")
        actives = set([(d[-2], d[0], d[-1]) for d in data if d[1] == 1])
        inactives = set([(d[-2], d[0], d[-1]) for d in data if d[1] == -1])
        act, inact, putinact = self.universe.predict(actives, inactives,
                                                     inactives_per_active=self.inactives_per_active,
                                                     min_actives=self.min_class_size,
                                                     naive=self.naive_sampling)
        self.__log.info("Actives %d / Known inactives %d / Putative inactives %d" %
                        (len(act), len(inact), len(putinact)))
        print("Actives %d / Known inactives %d / Putative inactives %d" %
                        (len(act), len(inact), len(putinact)))
        self.__log.debug("Assembling and shuffling")
        data = self._reassemble_activity_sets(act, inact, putinact)
        random.shuffle(data)
        self.__log.debug("Prepare for machine learning")
        y = np.array([d[1] for d in data])
        # Consider putative inactives as inactives (i.e. set -1 to 0)
        self.__log.debug(
            "Considering putative inactives as inactives for training")
        y[y <= 0] = 0
        molecules = np.array([(d[-2], d[-1]) for d in data])
        smiles = np.array([m[0] for m in molecules])
        # Check that there are enough molecules for training.
        ny = np.sum(y)
        if ny < self.min_class_size or (len(y) - ny) < self.min_class_size:
            self.__log.warning(
                "Not enough valid molecules in the minority class..." +
                "Just keeping training data")
            self._is_fitted = True
            self.save()
            return
        self.__log.info("Actives %d / Merged inactives %d" % (ny, len(y) - ny))
        # Get signatures
        self.__log.info("Calculating sign for every molecule.")
        for dataset in self.datasets:
            destination_dir = os.path.join(self.models_path, dataset)
            if os.path.exists(destination_dir) and use_checkpoints:
                continue
            else:
                self.__log.debug("Calculating sign for %s" % dataset)
                s3, predict_fn = self.sign_predict_fn[dataset]
                s3.predict_from_smiles([d[2] for d in data],
                                       destination_dir, predict_fn=predict_fn, use_novelty_model=False)
        # Fitting the global predictor
        self.__log.info("Fitting individual classifiers trained on full data")
        clf_ensemble = []
        if self._is_tpot:
            pipes = []
        for dataset in self.datasets:
            self.__log.debug("Working on %s" % dataset)
            clf = clone(self.base_mod)
            X = self._read_sign(dataset)
            shuff = np.array(range(len(y)))
            random.shuffle(shuff)
            clf.fit(X[shuff], y[shuff])
            if self._is_tpot:
                pipes += [clf.fitted_pipeline_]
                clf_ensemble += [(dataset, pipes[-1].fit(X[shuff], y[shuff]))]
            else:
                clf_ensemble += [(dataset, clf)]
        # Initialize cross-validation generator
        skf = StratifiedKFold(n_splits=np.min(
            [self.cv, ny]), shuffle=True, random_state=42)
        # Do the individual predictors
        self.__log.info("Training individual predictors with cross-validation")
        yps_train = collections.defaultdict(list)
        yps_test = collections.defaultdict(list)
        yts_train = []
        yts_test = []
        smi_test = []
        for train_idx, test_idx in skf.split(smiles, y):
            self.__log.debug("CV fold")
            for i, dataset in enumerate(self.datasets):
                # Fit the classifier
                if self._is_tpot:
                    clf = pipes[i]
                else:
                    clf = clone(self.base_mod)
                X_train = self._read_sign(dataset, train_idx)
                y_train = y[train_idx]
                clf.fit(X_train, y_train)
                # Make predictions on train set itself
                yps_train[dataset] += [p[1]
                                       for p in clf.predict_proba(X_train)]
                # Make predictions on test set itself
                X_test = self._read_sign(dataset, test_idx)
                y_test = y[test_idx]
                yps_test[dataset] += [p[1] for p in clf.predict_proba(X_test)]
            yts_train += list(y_train)
            yts_test += list(y_test)
            smi_test += list(smiles[test_idx])
        # Evaluate individual performances
        self.__log.info(
            "Evaluating dataset-specific performances based on the CV and" +
            "getting weights correspondingly")
        perfs = {}
        for dataset in self.datasets:
            ptrain = self.performances(yts_train, yps_train[dataset])
            ptest = self.performances(yts_test, yps_test[dataset])
            perfs[dataset] = {"perf_train": ptrain, "perf_test": ptest}
        # Meta-predictor on train and test data
        self.__log.info("Meta-predictions on train and test data")
        self.__log.debug("Assembling for train set")
        mps_train, std_train = self.metapredict(yps_train, perfs)
        self.__log.debug("Assembling for test set")
        mps_test, std_test = self.metapredict(yps_test, perfs)
        # Assess meta-predictor performance
        self.__log.debug("Assessing meta-predictor performance")
        ptrain = self.performances(yts_train, mps_train)
        ptest = self.performances(yts_test, mps_test)
        perfs["MetaPred"] = {"perf_train": ptrain, "perf_test": ptest}
        # Save performances
        with open(self.models_path + "/perfs.json", "w") as f:
            json.dump(perfs, f)
        # Save ensemble
        self._save_ensemble(clf_ensemble)
        if self.applicability:
            # Nearest neighbors
            self.__log.info(
                "Calculating nearest-neighbors model to be used in the applicability domain.")
            self.__log.debug("Getting fingerprint arena")
            fps_test = self.fingerprint_arena(
                smi_test, use_checkpoints=use_checkpoints, is_prd=False)
            # Save AD data
            self.__log.info("Calculating applicability domain weights")
            self.__log.debug("Working on the bias")
            bias_test = self.calculate_bias(yts_test, mps_test)
            self.__log.debug("Working on the weights")
            weights_test = self.calculate_weights(
                smi_test, fps_test, std_test, bias_test)
            self.__log.debug("Stacking AD data")
            ad_data = np.vstack((std_test, bias_test, weights_test)).T
            self.__log.info("Saving applicability domain weights")
            with open(self.models_path + "/ad_data.pkl", "wb") as f:
                pickle.dump(ad_data, f)
        # Cleaning up, if necessary
        if not use_checkpoints:
            self.__log.debug("Removing signature files")
            for dataset in self.datasets:
                os.remove(os.path.join(self.models_path, dataset))
        # Finish
        self._is_fitted = True
        self._is_trained = True
        # Save the class
        self.__log.debug("Saving TargetMate instance")
        self.save()

    def predict(self, data, datasets=None, standardize=False, known=True,
                sign_folder=None):
        '''
        Predict SMILES-activity data.
        Invalid SMILES are given no prediction.
        We provide the ouptut probability, the applicability domain and the
        precision of the model (1 - std).

        Args:
            data(str or list): SMILES strings expressed as a list or a file.
                If a folder name is given, then TargetMate assumes that
                signatures are provided (use with caution).
            datasets(list): Subset of datasets to use in the metaprediction.
                All by default (default=None)
            standardize(bool): If True, SMILES strings will be standardized
                (default=False)
            known(bool): Look for exact matches based on InChIKey
                (default=True)
            sign_folder(str): Path to a folder containing sign.

        Returns:
            mps(list): Metapredictions, expressed as probabilities (range: 0-1)
            ad(list): Applicability domain of the predictions (range: 0-1)
            prc(list): Precision of the prediction, based on the standard
                deviation across the ensemble of predictors (range: 0-1)
        '''
        self.__log.info("Predicting with model: %s" % self.models_path)
        if not self._is_fitted:
            raise Exception("TargetMate instance needs to be fitted first")
        # Dataset subset
        if not datasets:
            my_datasets = set(self.datasets)
        else:
            my_datasets = set(datasets)
        if len(my_datasets.intersection(self.datasets)) < 1:
            raise Exception("At least one valid dataset is necessary")
        if type(data) == str:
            data = os.path.abspath(data)
            if os.path.isdir(data):
                # When data is a folder, we assume it contains signatures
                self.__log.debug("Signature folder found")
                sign_folder = os.path.abspath(data)
                # We need to get the SMILES strings from these signatures, and
                # make sure everything is in the same order.
                self.__log.debug("Making sure SMILES are correct")
                sorted_datasets = sorted(
                    my_datasets.intersection(self.datasets))
                filename = os.path.join(sign_folder, sorted_datasets[0])
                with h5py.File(filename, "r") as hf:
                    previous = hf["keys"][:]
                for dataset in sorted_datasets[1:]:
                    dataset_fn = os.path.join(sign_folder, dataset)
                    with h5py.File(dataset_fn, "r") as hf:
                        current = hf["keys"][:]
                    if not np.array_equal(previous, current):
                        raise Exception(
                            "All signatures provided do not have the same keys!")
                    previous = current
                data = list(previous)
                # If it was read from signatures, do not standardize
                standardize = False
            elif os.path.isfile(data):
                self.__log.info("Reading data from file")
                # Read data if it is a file
                if type(data) == str:
                    if not os.path.isfile(data):
                        raise Exception("File %d does not exist" % data)
                    with open(data, "r") as f:
                        data = []
                        for r in csv.reader(f, delimiter="\t"):
                            data += [r[0]]
            else:
                raise Exception("%s does not exist" % data)
            # Get only valid SMILES strings
            self.__log.info(
                "Parsing SMILES strings, keeping only valid ones for training.")
            data_ = []
            for i, d in enumerate(data):
                m = self.read_smiles(d, standardize)
                if not m:
                    continue
                data_ += [(i, m[0], m[1])]
            data = data_
        # if a list is passed we assume it's a set of already parsed SMILES
        N = len(data)
        self.__log.info("%s SMILES strings parsed." % N)
        if type(data) != list:
            raise Exception("Unexpected 'data' type: %s" % type(data))
        # Check if the model has been trained
        if not self._is_trained:
            self.__log.warning(
                "Model was not trained because not enough data was" +
                "available. Beware of NaN values!")
            # Just putting NaN values
            mps = np.full(len(data), np.nan)
            std = np.full(len(data), np.nan)
            ad = np.full(len(data), np.nan)
        else:
            self.__log.debug(
                "Model trained before with enough data. Making predictions")
            # If signatures were not provided, then we work with the temporary
            # directory
            if os.path.exists(self.tmp_path):
                shutil.rmtree(self.tmp_path)
            os.mkdir(self.tmp_path)
            if not sign_folder:
                # Get signatures
                self.__log.info("Calculating sign for every molecule.")
                for dataset in self.datasets:
                    if dataset not in my_datasets:
                        continue
                    destination_dir = os.path.join(self.tmp_path, dataset)
                    if os.path.exists(destination_dir):
                        os.remove(destination_dir)
                    self.__log.debug("Calculating sign for %s" % dataset)
                    s3, predict_fn = self.sign_predict_fn[dataset]
                    s3.predict_from_smiles([d[2] for d in data],
                                           destination_dir,
                                           predict_fn=predict_fn)
            # Read ensemble of models
            self.__log.debug("Reading ensemble of models")
            clf_ensemble = self._load_ensemble()
            # Make predictions
            self.__log.info("Doing individual predictions")
            yps = {}
            for dataset, clf in clf_ensemble:
                if dataset not in my_datasets:
                    continue
                X = self._read_sign(
                    dataset, sign_folder=sign_folder, is_prd=True)
                yps[dataset] = [p[1] for p in clf.predict_proba(X)]
            # Read performances
            self.__log.debug("Reading performances")
            with open(self.models_path + "/perfs.json", "r") as f:
                perfs = json.load(f)
            # Do the metaprediction
            self.__log.info("Metaprediction")
            mps, std = self.metapredict(
                yps, perfs, dataset_universe=my_datasets)
            if applicability:
                # Do applicability domain
                self.__log.info("Calculating applicability domain")
                # Nearest neighbors
                self.__log.debug("Loading fit-time fingerprint arena")
                fps_fit = load_morgan_arena(
                    os.path.join(self.models_path, "arena.fps"))
                # Applicability domain data
                self.__log.debug("Loading applicability domain data")
                with open(self.models_path + "/ad_data.pkl", "r") as f:
                    ad_data = pickle.load(f)
                # Calculate weights
                self.__log.debug("Calculating weights")
                # fps = self.fingerprint_arena([d[2] for d in data], is_prd=True)
                smiles = [d[-2] for d in data]
                weights = self.calculate_weights(
                    smiles, fps_fit, ad_data[:, 0], ad_data[:, 1], N)
                # Get percentiles
                self.__log.debug("Calculating percentiles of the weight (i.e. AD)")
                train_weights = ad_data[:, 2]
                ad = np.array([percentileofscore(train_weights, w)
                               for w in weights]) / 100.
        # Over-write with the actual value if known is True. This just uses the
        # InChIKey to match.
        if known:
            self.__log.info("Overwriting with known data")
            with open(self.models_path + "/trained_data.pkl", "r") as f:
                tr_data = pickle.load(f)
                tr_iks = [d[1] for d in tr_data]
                tr_iks_set = set(tr_iks)
                for i, d in enumerate(data):
                    if d[1] not in tr_iks_set:
                        continue
                    idx = tr_iks.index(d[1])
                    mps[i] = tr_data[idx][-1]
                    std[i] = 0
                    ad[i] = 1
        self.__log.debug("Done. Mapping results to original data")
        # Map results to original data and return
        mps_ = np.full(N, np.nan)
        ad_ = np.full(N, np.nan)
        prc_ = np.full(N, np.nan)
        for i, d in enumerate(data):
            idx = d[0]
            mps_[idx] = mps[i]
            ad_[idx] = ad[i]
            prc_[idx] = 1 - std[i]
        # Finish
        self.__log.debug("Remove temporary directory")
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)
        # Return
        return mps_, ad_, prc_, data

    @staticmethod
    def fit_all_hpc(activity_path, models_path, **kwargs):
        hpc.fit_all_hpc(activity_path, models_path, **kwargs)

    @staticmethod
    def predict_all_hpc(models_path, signature_path, results_path,
                        models_filter=None, **kwargs):
        hpc.predict_all_hpc(models_path, signature_path, results_path, models_filter=None, **kwargs)
