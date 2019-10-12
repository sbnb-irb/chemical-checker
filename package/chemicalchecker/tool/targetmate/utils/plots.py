"""
TargetMate model analytics plots.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import stats
import numpy as np

# Individual

def scores_distro(ax, perfs):
    p = perfs["MetaPred"]
    y_pred_tr = p["perf_train"]["y_pred"]
    y_pred_ts = p["perf_test"]["y_pred"]
    density = stats.kde.gaussian_kde(y_pred_tr)
    x = np.arange(0, 1, 0.01)
    ax.plot(x, density(x), color = "black")
    density = stats.kde.gaussian_kde(y_pred_ts)
    x = np.arange(0, 1, 0.01)
    ax.plot(x, density(x), color = "red")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    pos = np.sum(p["perf_train"]["y_true"]) + np.sum(p["perf_test"]["y_true"])
    neg = len(p["perf_train"]["y_true"]) + len(p["perf_test"]["y_true"]) - pos 
    ax.set_title("Pos. %d / Neg. %d" % (pos, neg))

def individual_performances(ax, perfs, metric):
    keys = sorted([k for k in perfs.keys() if k != "MetaPred"]) + ["MetaPred"]
    tr = []
    ts = []
    for k in keys:
        p = perfs[k]
        tr += [p["perf_train"][metric][0]]
        ts += [p["perf_test"][metric][0]]
    ax.scatter([i for i in range(0, len(keys))], tr, color = "white", edgecolor = ["black"]*(len(tr) - 1) + ["red"])
    ax.scatter([i for i in range(0, len(keys))], ts, color = ["black"]*(len(tr) - 1) + ["red"])
    ax.set_ylabel(metric.upper())
    ax.set_title("%.2f / Avg. %.2f / Best %.2f" % (ts[-1], np.mean(ts[:-1]), np.max(ts[:-1])))
    
def rocs(ax, perfs):
    p = perfs["MetaPred"]
    y_true_tr = p["perf_train"]["y_true"]
    y_true_ts = p["perf_test"]["y_true"]
    y_pred_tr = p["perf_train"]["y_pred"]
    y_pred_ts = p["perf_test"]["y_pred"]
    fpr, tpr, _ = roc_curve(y_true_tr, y_pred_tr)
    ax.plot([0] + list(fpr) + [1], [0] + list(tpr) + [1], color = "black")
    fpr, tpr, _ = roc_curve(y_true_ts, y_pred_ts)
    ax.plot([0] + list(fpr) + [1], [0] + list(tpr) + [1], color = "red")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Train %.2f / Test %.2f" % (p["perf_train"]["auroc"][0], p["perf_test"]["auroc"][0]))
        
def precrecs(ax, perfs):
    p = perfs["MetaPred"]
    y_true_tr = p["perf_train"]["y_true"]
    y_true_ts = p["perf_test"]["y_true"]
    y_pred_tr = p["perf_train"]["y_pred"]
    y_pred_ts = p["perf_test"]["y_pred"]
    pre, rec, _ = precision_recall_curve(y_true_tr, y_pred_tr)
    ax.plot(list(rec), list(pre), color = "black")
    pre, rec, _ = precision_recall_curve(y_true_ts, y_pred_ts)
    ax.plot(list(rec), list(pre), color = "red")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Train %.2f / Test %.2f" % (p["perf_train"]["aupr"][0], p["perf_test"]["aupr"][0]))
    
def applicability(ax, perfs, ad_data, col):
    if col == 0:
        name = "Standard deviation"
    if col == 1:
        name = "Precision"
    if col == 2:
        name = "Weight"
    x = np.array(ad_data[:, col])
    y = np.array(perfs["MetaPred"]["perf_test"]["y_pred"])
    xy = np.vstack([x,y])
    z = stats.kde.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]    
    ax.scatter(x, y, c=z, edgecolor = "", cmap = "Spectral")
    ax.set_ylabel("Score")
    ax.set_xlabel(name)

# Main functions
    
def ensemble_classifier_plots(perfs, ad_data):
    fig, axs = plt.subplots(3, 3, figsize = (10, 10))
    axs = axs.flatten()
    if perfs is not None:
        scores_distro(axs[0], perfs)
        rocs(axs[1], perfs)    
        precrecs(axs[2], perfs)
        individual_performances(axs[3], perfs, "bedroc")
        individual_performances(axs[4], perfs, "auroc")
        individual_performances(axs[5], perfs, "aupr")
    if ad_data is not None:
        applicability(axs[6], perfs, ad_data, 0)
        applicability(axs[7], perfs, ad_data, 1)
        applicability(axs[8], perfs, ad_data, 2)
    plt.tight_layout()
    return fig, axs