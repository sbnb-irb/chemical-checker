from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

class VanillaClassifierConfigs:

    def __init__(self, base_mod, n_jobs, **kwargs):
        
        if base_mod == "logistic_regression":
            from sklearn.linear_model import LogisticRegressionCV
            self.base_mod = LogisticRegressionCV(
                cv=3, class_weight="balanced", max_iter=1000,
                n_jobs=n_jobs)
        if base_mod == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.base_mod = RandomForestClassifier(
                n_estimators=100, class_weight="balanced", n_jobs=n_jobs)
        if base_mod == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            self.base_mod = GaussianNB()
        if base_mod == "balanced_random_forest":
            from imblearn.ensemble import BalancedRandomForestClassifier
            self.base_mod = BalancedRandomForestClassifier(
                n_estimators=100, class_weight="balanced", n_jobs=n_jobs)

    def as_pipeline(self, X=None, y=None, **kwargs):
        return Pipeline([('variance_threshold_0', VarianceThreshold()),
                         ('classify', self.base_mod)])

