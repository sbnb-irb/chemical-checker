import numpy as np

tpot_configs = {

    # Naive Bayes
    "naive_bayes" = {
        'sklearn.naive_bayes.GaussianNB': {},
        'sklearn.feature_selection.VarianceThreshold': {}
    },

    # Random Forest
    "random_forest" = {
       'sklearn.ensemble.RandomForestClassifier': {
            'class_weight': ['balanced'],
            'n_estimators': [100, 500],
            'criterion': ['gini'],
            'min_samples_split': [2, 5, 10], 
            'min_samples_leaf': [1, 2, 4],
            'max_features': ["sqrt", "log2", "auto"],
            'bootstrap': [True, False]
            }
    },

    # Logistic Regression
    "logistic_regression" = {
        'sklearn.linear_model.LogisticRegression': {
            'class_weight': ['balanced'],
            'penalty': ['l1', 'l2'],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [False]
        }
    }

}

#     # Minimalistic TPOT Search
#     "minimal" = {
#         # Models
#         'sklearn.naive_bayes.GaussianNB': {
#             },
#         'sklearn.ensemble.RandomForestClassifier': {
#             'class_weight': ['balanced'],
#             'n_estimators': [10, 100],
#             'criterion': ['gini'],
#             'min_samples_split': [2], 
#             'min_samples_leaf': [1],
#             'max_features': ["sqrt", "log2"],
#             'bootstrap': [True]
#             },
#         'sklearn.linear_model.LogisticRegression': {
#             'class_weight': ['balanced'],
#             'penalty': ['l1', 'l2'],
#             'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
#             'dual': [False]
#             },
#         'sklearn.svm.LinearSVC':{
#             'penalty': ["l1", "l2"],
#             'loss': ["hinge", "squared_hinge"],
#             'dual': [True, False],
#             'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#             'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
#             'class_weight': ["balanced"]
#         },
#         # Selectors
#         'sklearn.feature_selection.SelectPercentile': {
#             'percentile': range(1, 100),
#             'score_func': {
#                 'sklearn.feature_selection.f_classif': None
#             }
#         }
#     },

#     # Light TPOT search, as defined by M. Orozco-Ruiz.
#     "light" = {
#         'sklearn.naive_bayes.GaussianNB': { 
#         }, 
#         'sklearn.tree.DecisionTreeClassifier': {
#             'class_weight': ["balanced"],
#             'criterion': ["gini", "entropy"],
#             'max_depth': range(1,11),
#             'min_samples_split': range(2,21),
#             'min_samples_leaf': range(1,21)
#         }, 
#         'sklearn.ensemble.RandomForestClassifier': {
#             'class_weight': ["balanced"],
#             'n_estimatiors': [100],
#             'criterion': ["gini", "entropy"],
#             'min_samples_split': range(2,21),
#             'min_samples_leaf': range(1,21),
#             'max_features': np.arange(0.05, 1.01, 0.05),
#             'bootstrap': [True, False]
#             }, 
#         'sklearn.linear_model.LogisticRegression': {
#             'class_weight': ["balanced"],
#             'penalty': ["l1", "l2"],
#             'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
#             'dual': [True, False]
#         }, 
#         'sklearn.svm.LinearSVC':{
#             'penalty': ["l1", "l2"],
#             'loss': ["hinge", "squared_hinge"],
#             'dual': [True, False],
#             'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#             'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
#             'class_weight': ["balanced"]
#         }, 
#         'sklearn.neighbors.KNeighborsClassifier': {
#             'weights': ['auto'],
#             'n_neighbors': range(1,101)
#         },
#         # Preprocessors
#         'sklearn.preprocessing.Binarizer': {
#             'threshold': np.arange(0.0, 1.01, 0.05)
#         },
#         'sklearn.cluster.FeatureAgglomeration': {
#             'linkage': ['ward', 'complete', 'average'],
#             'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
#         },
#         'sklearn.preprocessing.MaxAbsScaler': {
#         },
#         'sklearn.preprocessing.MinMaxScaler': { 
#         },
#         'sklearn.preprocessing.Normalizer': {
#             'norm': ['l1', 'l2', 'max']
#         },
#         'sklearn.decomposition.PCA': {
#             'svd_solver': ['randomized'],
#             'iterated_power': range(1, 11)
#         },
#         'sklearn.kernel_approximation.RBFSampler': {
#             'gamma': np.arange(0.0, 1.01, 0.05)
#         },
#         'sklearn.preprocessing.RobustScaler': {
#         },
#         'sklearn.preprocessing.StandardScaler': {
#         },
#         'tpot.builtins.ZeroCount': {
#         },
#         # Selectors
#         'sklearn.feature_selection.SelectPercentile': {
#             'percentile': range(1, 100),
#             'score_func': {
#                 'sklearn.feature_selection.f_classif': None
#             }
#         },
#         'sklearn.feature_selection.VarianceThreshold': {
#             'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
#         }
#     }

# }

class TPOTClassifierConfigs:

    def __init__(self, base_mod, **kwargs):
        from tpot import TPOTClassifier
        self.base_mod = TPOTClassifier(
            config_dict=tpot_configs[base_mod],
            generations=100,
            population_size=30,
            cv=self.cv,
            scoring="balanced_accuracy",
            verbosity=2,
            n_jobs=n_jobs,
            max_time_mins=30,
            max_eval_time_mins=10,
            random_state=42,
            early_stop=3,
            disable_update_check=True)
        
    def as_pipeline(self, X, y, **kwargs):
        """Select a pipeline, typically using hyper-parameter optimization methods e.g. TPOT.
        
        Args:
            X(array): Signatures matrix.
            y(array): Labels vector.
        """
        mod = clone(self.base_mod)
        mod.fit(X, y)
        return mod.fitted_pipeline_