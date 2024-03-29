import numpy as np
from sklearn import metrics

ALPHA = 1e-6

# Base metrics
def roc_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = metrics.roc_auc_score(y_true, y_pred)
    weight = (max(score, 0.5) - 0.5) / (1. - 0.5)
    return score, weight + ALPHA

def pr_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    prec, rec, _ = metrics.precision_recall_curve(y_true, y_pred)
    score = metrics.auc(rec, prec)
    weight = (score - 0.) / (1. - 0.)
    return score, weight + ALPHA

def bedroc_score(y_true, y_pred, alpha=20.0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    big_n = len(y_true)
    n = sum(y_true == 1)
    order = np.argsort(-y_pred)
    m_rank = (y_true[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) -
                                      np.cosh(alpha / 2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    score = s * fac / rand_sum + cte
    weight = (score - 0.) / (1. - 0.)
    return score, weight + ALPHA

def bacc_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = 0
    fpr, tpr, roc_thr = metrics.roc_curve(y_true, y_pred)
    for i in range(len(roc_thr)):
        bacc = np.mean([(1. - fpr[i]), tpr[i]])
        if bacc > score:
            score = bacc
    weight = (max(score, 0.5) - 0.5) / (1. - 0.5)
    return score, weight + ALPHA

def cohen_kappa_score(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred)

def roc_curve(y_true, y_pred, **kwargs):
    return metrics.roc_curve(y_true, y_pred, **kwargs)

# def f1_score(y_true, y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     score = metrics.f1_score(y_true, y_pred)
#     weight = (score - 0.) / (1. - 0.)
#     return score, weight + ALPHA

# Metric assigner
def Metric(metric):
    if metric == "auroc" : return roc_score
    if metric == "aupr"  : return pr_score
    if metric == "bedroc": return bedroc_score
    if metric == "bacc": return bacc_score
    # if metric == "f1": return f1_score
