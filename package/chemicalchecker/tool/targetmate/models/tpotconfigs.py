import numpy as np

naive = {
    'sklearn.naive_bayes.GaussianNB': {}
}

minimal = {
    'sklearn.naive_bayes.GaussianNB': {
        },
    'sklearn.ensemble.RandomForestClassifier': {
        'class_weight': ['balanced'],
        'n_estimators': [10, 100],
        'criterion': ['gini'],
        'min_samples_split': [2], 
        'min_samples_leaf': [1],
        'max_features': ["sqrt", "log2"],
        'bootstrap': [True]
        },
    'sklearn.linear_model.LogisticRegression': {
        'class_weight': ['balanced'],
        'penalty': ['l1', 'l2'],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [False]
        }
}

light = {
	'sklearn.naive_bayes.GaussianNB': { 
	}, 
	'sklearn.tree.DecisionTreeClassifier': {
		'class_weight': ["balanced"],
		'criterion': ["gini", "entropy"],
		'max_depth': range(1,11),
		'min_samples_split': range(2,21),
		'min_samples_leaf': range(1,21)
	}, 
	'sklearn.ensemble.RandomForestClassifier': {
		'class_weight': ["balanced"],
		'n_estimatiors': [100],
		'criterion': ["gini", "entropy"],
		'min_samples_split': range(2,21),
		'min_samples_leaf': range(1,21),
		'max_features': np.arange(0.05, 1.01, 0.05),
		'bootstrap': [True, False]
		}, 
	'sklearn.linear_model.LogisticRegression': {
		'class_weight': ["balanced"],
		'penalty': ["l1", "l2"],
		'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
		'dual': [True, False]
	}, 
	'sklearn.svm.LinearSVC':{
		'penalty': ["l1", "l2"],
		'loss': ["hinge", "squared_hinge"],
		'dual': [True, False],
		'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
		'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
		'class_weight': ["balanced"]
	}, 
	'sklearn.neighbors.KNeighborsClassifier': {
		'weights': ['auto'],
		'n_neighbors': range(1,101)
	},
        # Preprocesssors
	'sklearn.preprocessing.Binarizer': {
		'threshold': np.arange(0.0, 1.01, 0.05)
	},
	'sklearn.cluster.FeatureAgglomeration': {
		'linkage': ['ward', 'complete', 'average'],
		'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
	},
	'sklearn.preprocessing.MaxAbsScaler': {
	},
	'sklearn.preprocessing.MinMaxScaler': { 
	},
	'sklearn.preprocessing.Normalizer': {
		'norm': ['l1', 'l2', 'max']
	},
	'sklearn.decomposition.PCA': {
		'svd_solver': ['randomized'],
		'iterated_power': range(1, 11)
	},
	'sklearn.kernel_approximation.RBFSampler': {
		'gamma': np.arange(0.0, 1.01, 0.05)
	},
	'sklearn.preprocessing.RobustScaler': {
	},
	'sklearn.preprocessing.StandardScaler': {
	},
	'tpot.builtins.ZeroCount': {
	},
        # Selectors
	'sklearn.feature_selection.SelectPercentile': {
		'percentile': range(1, 100),
		'score_func': {
			'sklearn.feature_selection.f_classif': None
		}
	},
	'sklearn.feature_selection.VarianceThreshold': {
		'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
	}
}
