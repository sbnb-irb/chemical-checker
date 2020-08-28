"""Binary performances."""
import json
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import f1_score, roc_curve, auc, matthews_corrcoef
from sklearn.metrics import average_precision_score, davies_bouldin_score
from sklearn.metrics import confusion_matrix, precision_score, silhouette_score


class PerformanceBinary():
    """PerformanceBinary class.

    Compute performance metric for a binary classiier.
    """

    metrics = {
        "auc_roc": "AUC-ROC",
        "auc_pr": "AUC-PR",
        "thr": "Threshold",
        "sens": "Sensitivity",
        "spec": "Specificity",
        "mcc": "MCC",
        "f1": "F1",
        "prec": "Precision"
    }

    def __init__(self, y_true, y_pred):
        """Initialize a PerformanceBinary instance.

        Args:
            y_true(array): Array of truth labels.
            y_pred(array): Array of predicted labels.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        self.auc_roc = auc(fpr, tpr)
        self.auc_pr = average_precision_score(y_true, y_pred)
        max_bacc = 0
        self.thr = 0
        for i in range(len(thresholds)):
            bacc = np.mean([(1. - fpr[i]), tpr[i]])
            if bacc > max_bacc:
                self.thr = thresholds[i]
                max_bacc = bacc
        y_class = []
        for yp in y_pred:
            if yp >= self.thr:
                y_class += [1]
            else:
                y_class += [0]
        self.M = confusion_matrix(y_true, y_class)
        TN, FP, FN, TP = self.M.ravel()
        self.sens = float(TP) / (TP + FN)  # recall_score
        self.spec = float(TN) / (TN + FP)
        self.mcc = matthews_corrcoef(y_true, y_class)
        self.Bacc = (self.sens + self.spec) / 2.
        self.f1 = f1_score(y_true, y_class)
        self.prec = precision_score(y_true, y_class)

    def __str__(self):
        to_str = ""
        for attr in sorted(self.metrics):
            to_str += "{:15}{:15}\n".format(
                self.metrics[attr], getattr(self, attr))
        return to_str

    def toJSON(self, filename):
        """Save in stats in json format."""
        tmp = dict()
        for attr, name in self.metrics.items():
            tmp[name] = str(getattr(self, attr))
        with open(filename, 'w') as fh:
            json.dump(tmp, fh)


class PerformanceCluster():

    metrics = {
        "nr_clusters": "nr_clusters",
        "fraction_noisy": "fraction_noisy",
        "davies_bouldin_score": "davies_bouldin_score",
        "calinski_harabaz_score": "calinski_harabaz_score",
        "silhouette_score": "silhouette_score",
        "davies_bouldin_score_nonoisy": "davies_bouldin_score_nonoisy",
        "calinski_harabaz_score_nonoisy": "calinski_harabaz_score_nonoisy",
        "silhouette_score_nonoisy": "silhouette_score_nonoisy"
    }

    def __init__(self, X, labels, strengths):
        self.nr_clusters = len(set(labels))
        self.fraction_noisy = sum(labels == -1) / float(len(labels))
        self.silhouette_score = silhouette_score(X, labels)
        self.davies_bouldin_score = davies_bouldin_score(X, labels)
        self.calinski_harabaz_score = calinski_harabaz_score(X, labels)
        # same but without noisy data
        labels_nonoisy = labels[labels >= 0]
        X_nonoisy = X[labels >= 0]
        self.silhouette_score_nonoisy = silhouette_score(
            X_nonoisy, labels_nonoisy)
        self.davies_bouldin_score_nonoisy = davies_bouldin_score(
            X_nonoisy, labels_nonoisy)
        self.calinski_harabaz_score_nonoisy = calinski_harabaz_score(
            X_nonoisy, labels_nonoisy)

    def __str__(self):
        to_str = ""
        for attr in sorted(self.metrics):
            to_str += "{:15}{:15}\n".format(
                self.metrics[attr], getattr(self, attr))
        return to_str

    def toJSON(self, filename):
        tmp = dict()
        for attr, name in self.metrics.items():
            tmp[name] = str(getattr(self, attr))
        with open(filename, 'w') as fh:
            json.dump(tmp, fh)
