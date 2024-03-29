from ..utils.splitters import GetSplitter
from ..utils.metrics import Metric
import numpy as np

def validation_score(mod,
                     X, y,
                     smiles,
                     metric,
                     is_cv,
                     is_classifier,
                     is_stratified,
                     scaffold_split,
                     n_splits,
                     test_size,
                     random_state):

    Spl = GetSplitter(is_cv=is_cv,
                      is_classifier=is_classifier,
                      is_stratified=is_stratified,
                      scaffold_split=scaffold_split,
                      outofuniverse_split=False)

    kf = Spl(n_splits=n_splits, test_size=test_size, random_state=random_state)
    metric = Metric(metric)
    score = []
    for train_idx, test_idx in kf.split(X=smiles, y=y):
        mod.fit(X[train_idx], y[train_idx])
        if is_classifier:
            y_pred = mod.predict_proba(X[test_idx])[:,1]
        else:
            y_pred = mod.predict(X[test_idx])
        y_true = y[test_idx]
        score += [metric(y_true, y_pred)[0]]
    return np.mean(score)
