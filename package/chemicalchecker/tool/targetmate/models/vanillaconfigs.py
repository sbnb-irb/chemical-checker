class VanillaClassifierConfigs:

    def __init__(self, base_mod, **kwargs):
        
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

    def as_pipeline(self):
        return Pipeline([('variance_threshold_0', VarianceThreshold()),
                         ('classifier', self.base_mod)])


