"""Utility for plotting the conventional diagnostics CC plots"""
import os
import pandas as pd
from chemicalchecker.util import logged
import matplotlib.pyplot as plt
import seaborn as sns
from chemicalchecker.util.plot.style.util import coord_color, set_style
import random
import numpy as np
import pickle

set_style()

@logged
class DiagnosisPlot(object):

    def __init__(self, cc, sign):
        """Initialize diagnostics plotter. The plotter works mainly on precomputed data (using the Diagnose class).. If you need to do computations, please see the Plot class, which is the one used in the CC pipeline.
        
            Args:
                cc(ChemicalChecker): A CC instance.
                sign(CC signature): A CC signature to be be diagnosed.
        """
        self.cc = cc
        self.sign = sign
        folds = self.sign.data_path.split("/")
        self.cctype = folds[-2]
        self.dataset = folds[-3]
        self.molset = folds[-6]

    @staticmethod
    def _get_ax(ax):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(5, 5))
        return ax

    def _get_color(self, color):
        if color is None:
            return coord_color(self.sign.dataset)
        else:
            return color

    def load_diagnosis_pickle(self, fn):
        with open(os.path.join(self.sign.stats_path, fn), "rb") as f:
            results = pickle.load(f)
        return results

    def available(self):
        d = {
            "across_coverage": "Coverage of other CC signatures",
            "across_roc": "ROC against other CC signatures",
            "atc_roc": "ROC for the ATC CC space (E1)",
            "cosine_distances": "Cosine distance distribution",
            "cross_coverage": "Coverage of another signature",
            "cross_roc": "ROC curve against another signature",
            "dimensions": "Latent dimensions",
            "euclidean_distances": "Euclidean distance distribution",
            "features_iqr": "IQR of the features values",
            "image": "Signature seen as a heatmap",
            "keys_iqr": "IQR of the keys values",
            "moa_roc": "ROC for the MoA CC space (B1)",
            "projection": "tSNE 2D projection",
            "values": "Values distibution of the signature"
        }
        R = []
        for k in sorted(d.keys()):
            R += [(k, d[k])]
        df = pd.DataFrame(R, columns=["method", "description"])
        return df

    def cross_coverage(self, sign=None, ax=None, title=None, color=None):
        ax = self._get_ax(ax)
        color = self._get_color(color)
        fn = os.path.join(self.sign.stats_path, "cross_coverage_%s.pkl" % self.cc.sign_name(sign))
        results = self.load_diagnosis_pickle(fn)
        ax.bar([0, 1], [results["my_overlap"], results["vs_overlap"]], hatch="////", edgecolor=color, lw=2, color="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Overlap")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["T / R", "R / T"])
        if title is None:
            title = "T = %s | R = %s" % (self.cc.sign_name(self.sign), self.cc.sign_name(sign))
        ax.set_title(title)

    def _roc(self, ax, results, color):
        step = 0.001
        fpr = np.arange(0, 1+step, step)
        tpr = np.interp(fpr, results["fpr"], results["tpr"])
        auc_ = results["auc"]
        if color is None:
            color = coord_color(dataset_code)
        ax.plot(fpr, tpr, color=color)
        ax.fill_between(fpr, tpr, color=color, alpha=0.25)
        ax.plot([0,1],[0,1], color="gray", linestyle="--")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        return ax
                      
    def cross_roc(self, sign=None, ax=None, title=None, color=None):
        ax = self._get_ax(ax)
        color = self._get_color(color)
        fn = os.path.join(self.sign.stats_path, "cross_roc_%s.pkl" % self.cc.sign_name(sign))
        results = self.load_diagnosis_pickle(fn)
        ax = self._roc(ax, results, color)
        if title is None:
            title = "%s | %s (%.3f)" % (self.cc.sign_name(self.sign), self.cc.sign_name(sign), auc_)
        ax.set_title(title)
        return ax

    def atc_roc(self, ax=None, title=None):
        ax = self._get_ax(ax)
        color = coord_color("E1.001")
        results = self.load_diagnosis_pickle("atc_roc.pkl")
        ax = self._roc(ax, results, color)
        if title is None:
            title = "ATC ROC (%.3f)" % results["auc"]
        ax.set_title(title)
        return ax

    def moa_roc(self, ax=None, title=None):
        ax = self._get_ax(ax)
        color = coord_color("B1.001")
        results = self.load_diagnosis_pickle("moa_roc.pkl")
        ax = self._roc(ax, results, color)
        if title is None:
            title = "MoA ROC (%.3f)" % results["auc"]
        ax.set_title(title)
        return ax

    def image(self, ax=None, title=None):
        ax = self._get_ax(ax)
        results = self.load_diagnosis_pickle("image.pkl")
        ax.imshow(results["X"], cmap="viridis", aspect="auto")
        if title is None:
            title = "Image"
        ax.set_ylabel("Keys")
        ax.set_xlabel("Dimensions")
        ax.set_title(title)
        ax.grid()
        return ax

    def projection(self, ax=None, density=True, color=None, title=None):
        ax = self._get_ax(ax)
        results = self.load_diagnosis_pickle("projection.pkl")
        P = results["P"]
        x = P[:,0]
        y = P[:,1]
        if density:
            from scipy.stats import gaussian_kde
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax.scatter(x, y, c=z, s=10, edgecolor="")
        else:
            color = self._get_color(color)
            ax.scatter(x, y, s=10, color=color, alpha=0.5)
        P_focus = results["P_focus"]
        if P_focus is not None:
            x = P_focus[:,0]
            y = P_focus[:,1]
            ax.scatter(x, y, edgecolor="black", color="white")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        if title is None:
            title = "2D projection"
        ax.set_title(title)
        return ax

    def _distance_distribution(self, results, ax=None, color=None):
        """Distance distribution plot"""
        ax = self._get_ax(ax)
        color = self._get_color(color)
        dists = results["dists"]
        sns.kdeplot(dists, ax=ax, shade=True, color=color)
        ax.set_ylabel("Density")
        return ax

    def euclidean_distances(self, ax=None, color=None, title=None):
        results = self.load_diagnosis_pickle("euclidean_distances.pkl")
        ax = self._distance_distribution(results, ax=ax, color=color)
        ax.set_xlabel("Euclidean distance")
        if title is None:
            title = "Euclidean distances"
        ax.set_title(title)

    def cosine_distances(self, ax=None, color=None, title=None):
        results = self.load_diagnosis_pickle("cosine_distances.pkl")
        ax = self._distance_distribution(results, ax=ax, color=color)
        if title is None:
            title = "Cosine distances"
        ax.set_title(title)
        ax.set_xlabel("Cosine distance")

    def _iqr(self, results, ax):
        ax = self._get_ax(ax)
        p50 = results["p50"]
        idxs = np.argsort(-p50)
        p25 = results["p25"][idxs]
        p50 = p50[idxs]
        p75 = results["p75"][idxs]
        x = [i for i in range(len(p50))]
        ax.scatter(x, p50, c=p50, cmap="Spectral", s=10, zorder=100)
        ax.fill_between(x, p75, p25, color="lightgray", alpha=0.5, zorder=1)
        ax.set_ylabel("Value")
        ax.axhline(0, color="black", lw=1)
        return ax

    def values(self, ax=None, color=None, title=None):
        ax = self._get_ax(ax)
        color = self._get_color(color)
        results = self.load_diagnosis_pickle("values.pkl")
        x = results["x"]
        y = results["y"]
        ax.plot(x, y, color=color)
        ax.fill_between(x, y, color=color, alpha=0.2)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if title is None:
            title = "Values distribution"
        ax.set_title(title)
        ax.set_ylim(0, np.max(y)*1.05)

    def features_iqr(self, ax=None, title=None):
        results = self.load_diagnosis_pickle("features_iqr.pkl")
        ax = self._iqr(results, ax=ax)
        ax.set_xlabel("Features (sorted)")
        if title is None:
            title = "Values by feature"
        ax.set_title(title)

    def keys_iqr(self, ax=None, title=None):
        results = self.load_diagnosis_pickle("keys_iqr.pkl")
        ax = self._iqr(results, ax=ax)
        ax.set_xlabel("Keys (sorted)")
        if title is None:
            title = "Values by key"
        ax.set_title(title)

    def _across(self, values, datasets, ax, title, exemplary, cctype, molset):
        ax = self._get_ax(ax)
        datasets = np.array(datasets)
        values = np.array(values)
        idxs = np.array(list(pd.DataFrame({"ds": datasets, "vl": -values}).sort_values(["vl", "ds"]).index)).astype(np.int)
        #idxs = np.argsort(-values)
        datasets = datasets[idxs]
        values = values[idxs]
        colors = [coord_color(ds) for ds in datasets]
        x = [i+1 for i in range(0, len(values))]
        ax.scatter(x, values, color=colors)
        for i, x_ in enumerate(x):
            ax.plot([x_,x_], [-1, values[i]], color=colors[i])
        if title is None:
            title = "%s | %s_%s" % (self.cc.sign_name(self.sign), cctype, molset)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([ds[1] for ds in datasets])
        ax.set_xlabel("Datasets")
        return ax

    def across_coverage(self, ax=None, title=None, exemplary=True, cctype="sign1", molset="full"):
        results = self.load_diagnosis_pickle("across_coverage.pkl")
        datasets = []
        covs = []
        for k,v in results.items():
            datasets += [k]
            covs += [v["vs_overlap"]]
        ax = self._across(covs, datasets, ax=ax, title=title, exemplary=exemplary, cctype=cctype, molset=molset)
        ax.set_ylabel("Coverage")
        ax.set_ylim(-0.05, 1.05)
        if title is None:
            title = "Coverage"
        ax.set_title(title)

    def across_roc(self, ax=None, title=None, exemplary=True, cctype="sign1", molset="full"):
        results = self.load_diagnosis_pickle("across_roc.pkl")
        datasets = []
        rocs = []
        for k,v in results.items():
            datasets += [k]
            rocs += [v["auc"]]
        ax = self._across(rocs, datasets, ax=ax, title=title, exemplary=exemplary, cctype=cctype, molset=molset)
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.45, 1.05)
        if title is None:
            title = "ROC across CC"
        ax.set_title(title)

    def dimensions(self, ax=None, title=None, exemplary=True, cctype="sign1", molset="full"):
        ax = self._get_ax(ax)
        results = self.load_diagnosis_pickle("dimensions.pkl")
        datasets = []
        colors = []
        x = []
        y = []
        for k,v in results.items():
            if k == "MY": continue
            datasets += [k]
            colors += [coord_color(k)]
            y += [v["keys"]]
            x += [len(v["expl"])]
        x = np.log10(x)
        y = np.log10(y)
        ax.scatter(x, y, color =colors)
        max_x = np.max(x)
        max_y = np.max(y)
        v = results["MY"]
        y = [v["keys"]]
        x = [len(v["expl"])]
        x = np.log10(x)
        y = np.log10(y)
        ax.scatter(x, y, color="white", edgecolor="black", s=80)
        ax.set_xlabel("Latent variables (log10)")
        ax.set_ylabel("Molecules (log10)")
        if title is None:
            title = "Latent dimensions"
        ax.set_title(title)
        return ax
       
    def canvas(self, title=None):
        fig = plt.figure(constrained_layout=True, figsize=(12,8))
        gs = fig.add_gridspec(4, 6)
        ax = fig.add_subplot(gs[0,0])
        self.euclidean_distances(ax)
        ax = fig.add_subplot(gs[0,1])
        self.cosine_distances(ax)
        ax = fig.add_subplot(gs[1,0])
        self.features_iqr(ax)
        ax = fig.add_subplot(gs[1,1])
        self.keys_iqr(ax)
        ax = fig.add_subplot(gs[0:2,2:4])
        self.projection(ax)
        ax = fig.add_subplot(gs[0,4:6])
        self.image(ax)
        ax = fig.add_subplot(gs[1,4])
        self.moa_roc(ax)
        ax = fig.add_subplot(gs[1,5])
        self.atc_roc(ax)
        ax = fig.add_subplot(gs[-2:,:2])
        self.dimensions(ax)
        ax = fig.add_subplot(gs[-2:,2:4])
        self.across_coverage(ax)
        ax = fig.add_subplot(gs[-2:,4:6])
        self.across_roc(ax)
        if title is None:
            title = "%s %s" % (self.dataset, self.cctype)
        fig.suptitle(title, fontweight="bold")
        plt.close()
        return fig
