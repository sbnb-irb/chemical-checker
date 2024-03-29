from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

SEED = 42

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
                n_estimators=500,
                class_weight="balanced",
                max_features="sqrt",
                random_state=SEED,
                n_jobs=n_jobs)
        if base_mod == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            self.base_mod = GaussianNB()
        if base_mod == "balanced_random_forest":
            from imblearn.ensemble import BalancedRandomForestClassifier
            self.base_mod = BalancedRandomForestClassifier(
                n_estimators=500, class_weight="balanced", n_jobs=n_jobs)
        if base_mod == "xgboost":
            from xgboost.sklearn import XGBClassifier
            self.base_mod = XGBClassifier(n_estimators=500, n_jobs=n_jobs)
        if base_mod == "SVM":
            from sklearn.svm import SVC
            self.base_mod = SVC(probability=True, class_weight = 'balanced')
        if base_mod == "LinearSVM":
            from sklearn.svm import SVC
            self.base_mod = SVC(kernel = 'linear', probability=True, class_weight = 'balanced')
        if base_mod == "UnbalancedSVM":
            from sklearn.svm import SVC
            self.base_mod = SVC(probability=True)
        if base_mod == "UnbalancedLinearSVM":
            from sklearn.svm import SVC
            self.base_mod = SVC(kernel='linear', probability=True)


    def as_pipeline(self, X=None, y=None, **kwargs):
        return self.base_mod
        #return Pipeline([('variance_threshold_0', VarianceThreshold()),
        #                 ('classify', self.base_mod)])

