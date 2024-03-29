"""Diagnostics CC plots."""
import os
import pickle
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.font_manager import FontProperties
from .util import coord_color, set_style, homogenous_ticks

from chemicalchecker.util import logged
#from chemicalchecker.util.decorator import safe_return

set_style()

pad_factor = 0


@logged
class DiagnosisPlot(object):
    """DiagnosisPlot class."""

    def __init__(self, diag):
        """Initialize a DiagnosisPlot instance.

        The plotter works on data precomputed using
        :mod:`~chemicalchecker.core.diagnostics`.

            Args:
                diag (Diagnosis): A Diagnosis object.
        """
        self.diag = diag

    @staticmethod
    def _get_ax(ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        return ax

    def _get_color(self, color):
        if color is None:
            return coord_color(self.diag.sign.dataset)
        else:
            return color

    @staticmethod
    def _categorical_colors(n):
        norm = mpl.colors.Normalize(vmin=1., vmax=n)
        cmap = cm.get_cmap("tab20b")
        colors = cmap(norm([i + 1 for i in range(0, n)]))
        return colors

    def load_diagnosis_pickle(self, fn):
        with open(os.path.join(self.diag.path, fn), "rb") as f:
            results = pickle.load(f)
        return results

    def available(self):
        d = {
            "across_coverage": "Coverage of other CC signatures",
            "across_roc": "ROC against other CC signatures",
            "atc_roc": "ROC for the ATC CC space (E1)",
            "cosine_distances": "Cosine distance distribution",
            "cross_coverage": "Coverage of another signature",
            "cross_pr": "PR curve against another signature",
            "cross_roc": "ROC curve against another signature",
            "cluster_sizes": "Size of identified clusters",
            "clusters_projection": "Projection of the clusters",
            "dimensions": "Latent dimensions",
            "euclidean_distances": "Euclidean distance distribution",
            "features_bins": "Binned features values",
            "features_iqr": "IQR of the features values",
            "global_ranks_agreement":
            "Agreement between similarity ranks and a CC consensus signature",
            "global_ranks_agreement_projection":
            "Projections of global ranks agreements",
            "image": "Signature seen as a heatmap",
            "intensities": "Intensities of the signatures",
            "intensities_projection": "Projection with intensities",
            "confidence": "Confidence of the signatures",
            "confidence_projection": "Projection with confidences",
            "keys_bins": "Binned keys values",
            "key_coverage": "Dataset coverage of the keys across the CC",
            "key_coverage_projection":
            "Projections with coverage of keys across the CC",
            "keys_iqr": "IQR of the keys values",
            "moa_roc": "ROC for the MoA CC space (B1)",
            "roc": "ROC for any specified space",
            "orthogonality": "Orthogonality of features",
            "outliers": "Detected outliers",
            "pr": "Precision recall for any specified space",
            "projection": "tSNE 2D projection",
            "ranks_agreement":
            "Agreement between similarity ranks across the CC",
            "ranks_agreement_projection": "Projections of ranks agreements",
            "redundancy": "Redundant keys",
            "values": "Values distibution of the signature"
        }
        R = []
        for k in sorted(d.keys()):
            R += [(k, d[k])]
        df = pd.DataFrame(R, columns=["method", "description"])
        return df

    # @safe_return(None)
    def cross_coverage(self, results=None, sign_qualified_name=None, ax=None,
                       title=None, color=None):
        ax = self._get_ax(ax)
        color = self._get_color(color)
        fn = os.path.join(self.diag.path,
                          "cross_coverage_%s.pkl" % sign_qualified_name)
        if results is None:
            results = self.load_diagnosis_pickle(fn)
        ax.bar([0, 1], [results["my_overlap"], results["vs_overlap"]],
               hatch="////", edgecolor=color, lw=2, color="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Overlap")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["T / R", "R / T"])
        if title is None:
            title = "T = %s | R = %s" % (self.diag.sign.qualified_name,
                                         sign_qualified_name)
        ax.set_title(title)

    def _roc(self, ax, results, color, dataset_code=None, alpha=0.25,
             label=None, xylabels=True):
        step = 0.001
        fpr = np.arange(0, 1 + step, step)
        tpr = np.interp(fpr, results["fpr"], results["tpr"])
        if color is None:
            color = coord_color(dataset_code)
        ax.plot(fpr, tpr, color=color, label=label)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.fill_between(fpr, tpr, color=color, alpha=alpha)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        if xylabels:
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
        return ax

    def _pr(self, ax, results, color, dataset_code=None, alpha=0.25,
            label=None, xylabels=True):
        recall = results["recall"]
        precision = results["precision"]
        if color is None:
            color = coord_color(dataset_code)
        ax.plot(recall, precision, color=color, label=label)
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.fill_between(recall, precision, color=color, alpha=alpha)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        if xylabels:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        return ax

    def _pr(self, ax, results, color, dataset_code=None, alpha=0.25,
            label=None):
        recall = results["recall"]
        precision = results["precision"]
        if color is None:
            color = coord_color(dataset_code)
        ax.plot(recall, precision, color=color, label=label)
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.fill_between(recall, precision, color=color, alpha=alpha)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        return ax

    # @safe_return(None)
    def cross_roc(self, results=None, sign=None, ax=None, title=None,
                  color=None, xylabels=True):
        ax = self._get_ax(ax)
        color = self._get_color(color)
        fn = os.path.join(self.diag.path,
                          "cross_roc_%s.pkl" % sign.qualified_name)
        if results is None:
            results = self.load_diagnosis_pickle(fn)
        ax = self._roc(ax, results, color, xylabels=xylabels)
        if title != False:
            if title is None:
                title = "%s | %s (%.3f)" % (self.diag.sign.qualified_name,
                                            sign.qualified_name, results["auc"])
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def cross_pr(self, results=None, sign=None, ax=None, title=None,
                 color=None, xylabels=True):
        ax = self._get_ax(ax)
        color = self._get_color(color)
        fn = os.path.join(self.diag.path,
                          "cross_roc_%s.pkl" % sign.qualified_name)
        if results is None:
            results = self.load_diagnosis_pickle(fn)
        ax = self._pr(ax, results, color, xylabels=xylabels)
        if title != False:
            if title is None:
                title = "%s | %s (%.3f)" % (self.diag.sign.qualified_name,
                                            sign.qualified_name,
                                            results["average_precision_score"])
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def atc_roc(self, results=None, ax=None, title=None, color=None,
                xylabels=True):
        ax = self._get_ax(ax)
        if color is None:
            color = coord_color("E1.001")
        if results is None:
            results = self.load_diagnosis_pickle("atc_roc.pkl")
        ax = self._roc(ax, results, color, xylabels=xylabels)
        if title != False:
            if title is None:
                title = "ATC (%.3f)" % results["auc"]
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def moa_roc(self, results=None, ax=None, title=None, color=None,
                xylabels=True):
        ax = self._get_ax(ax)
        if color is None:
            color = coord_color("B1.001")
        if results is None:
            results = self.load_diagnosis_pickle("moa_roc.pkl")
        ax = self._roc(ax, results, color, xylabels=xylabels)
        if title != False:
            if title is None:
                title = "MoA (%.3f)" % results["auc"]
            ax.set_title(title)
        return ax

    # @safe_return(None)
    def roc(self, ds, results=None, ax=None, title=None):
        ax = self._get_ax(ax)
        color = coord_color(ds)
        if results is None:
            results = self.load_diagnosis_pickle("roc.pkl")
        ax = self._roc(ax, results, color)
        if title is None:
            title = "AUROC (%.3f)" % results["auc"]
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def pr(self, ds, results=None, ax=None, title=None):
        ax = self._get_ax(ax)
        color = coord_color(ds)
        if results is None:
            results = self.load_diagnosis_pickle("pr.pkl")
        ax = self._pr(ax, results, color)
        if title is None:
            title = "Average precision score (%.3f)" % results["average_precision_score"]
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def neigh_roc(self, ds, results=None, ax=None, title=None):
        ax = self._get_ax(ax)
        color = coord_color(ds)
        if results is None:
            results = self.load_diagnosis_pickle("neigh_roc.pkl")
        for nn, res in results.items():
            if res is None:
                continue
            ax = self._roc(
                ax, res, color,
                label='NN {:<5} (AUC {:.3f})'.format(nn, res['auc']))
        ax.legend()
        return ax

    # @safe_return(None)
    def image(self, results=None, ax=None, title=None, cmap="coolwarm",
              cap_percentile=None):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("image.pkl")
        X = results["X"]
        eps = cap_percentile
        if eps is None:
            eps = np.clip(np.std(X), 0, 1)
        ax.imshow(X, cmap=cmap, aspect="auto", vmin=np.percentile(X, eps),
                  vmax=np.percentile(X, 100 - eps), interpolation='nearest')
        if title is None:
            title = "Image"
        ax.set_ylabel("Keys")
        ax.set_xlabel("Features")
        ax.set_title(title)
        ax.grid(True)
        return ax

    def _proj_lims(self, P):
        xlim = [np.min(P[:, 0]), np.max(P[:, 0])]
        ylim = [np.min(P[:, 1]), np.max(P[:, 1])]
        xscale = (xlim[1] - xlim[0]) * 0.05
        yscale = (ylim[1] - ylim[0]) * 0.05
        xlim[0] -= xscale
        xlim[1] += xscale
        ylim[0] -= yscale
        ylim[1] += yscale
        return xlim, ylim

    # @safe_return(None)
    def projection(self, results=None, ax=None, density=True, color=None,
                   title=None, focus_keys=None):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("projection.pkl")
        P = results["P"]
        x = P[:, 0]
        y = P[:, 1]
        if density:
            from scipy.stats import gaussian_kde
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax.scatter(x, y, c=z, s=10, edgecolor=None)
        else:
            color = self._get_color(color)
            ax.scatter(x, y, s=10, color=color, alpha=0.5)

        if focus_keys is not None:
            focus_keys = list(set(focus_keys) & set(results["keys"]))
            self.__log.debug("%d focus keys found" % len(focus_keys))
            focus_idxs = np.isin(results["keys"], focus_keys)
            P_focus = P[focus_idxs]
        else:
            P_focus = None

        if P_focus is not None:
            x = P_focus[:, 0]
            y = P_focus[:, 1]
            ax.scatter(x, y, edgecolor="black", color="white")
        xlim, ylim = self._proj_lims(P)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        homogenous_ticks(ax, 4)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if title is None:
            title = "2D projection"
        ax.set_title(title)
        return ax

    def _distance_distribution(self, results, ax=None, color=None):
        """Distance distribution plot"""
        ax = self._get_ax(ax)
        color = self._get_color(color)
        dists = results["dists"]
        sns.kdeplot(dists, ax=ax, fill=True, color=color, alpha=0.25,
                    linewidth=1.5)
        ax.set_ylabel("Density")
        ax.set_yticklabels([])
        return ax

    # @safe_return(None)
    def euclidean_distances(self, results=None, ax=None, color=None, title=None):
        if results is None:
            results = self.load_diagnosis_pickle("euclidean_distances.pkl")
        ax = self._distance_distribution(results, ax=ax, color=color)
        ax.set_xlabel("Euclidean")
        if title is None:
            title = "Euclidean dist."
        ax.set_title(title)

    # @safe_return(None)
    def cosine_distances(self, results=None, ax=None, color=None, title=None):
        if results is None:
            results = self.load_diagnosis_pickle("cosine_distances.pkl")
        ax = self._distance_distribution(results, ax=ax, color=color)
        if title is None:
            title = "Cosine dist."
        ax.set_title(title)
        ax.set_xlabel("Cosine")

    # @safe_return(None)
    def values(self, results=None, ax=None, s=1, cmap="coolwarm", title=None):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("values.pkl")
        x = results["x"]
        y = results["y"]
        ax.scatter(x, y, c=x, cmap=cmap, s=s)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if title is None:
            title = "Values distr."
        ax.set_title(title)
        ax.set_ylim(0, np.max(y) * 1.05)
        ax.set_yticklabels([])

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

    # @safe_return(None)
    def features_iqr(self, results=None, ax=None, title=None):
        if results is None:
            results = self.load_diagnosis_pickle("features_iqr.pkl")
        ax = self._iqr(results, ax=ax)
        ax.set_xlabel("Features (sorted)")
        if title is None:
            title = "Values by feat."
        ax.set_title(title)

    # @safe_return(None)
    def keys_iqr(self, results=None, ax=None, title=None):
        if results is None:
            results = self.load_diagnosis_pickle("keys_iqr.pkl")
        ax = self._iqr(results, ax=ax)
        ax.set_xlabel("Keys (sorted)")
        if title is None:
            title = "Values by key"
        ax.set_title(title)

    def _bins(self, results, ax, scaling, cmap):
        ax = self._get_ax(ax)
        H = results["H"]
        p50 = results["p50"]
        bins = results["bins"]
        idxs = np.argsort(-p50)
        p50 = p50[idxs]
        x_ = [i + 1 for i in range(0, H.shape[1])]
        y_ = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))]
        x = np.array(x_ * H.shape[0])
        y = np.array([[yy] * H.shape[1] for yy in y_]).ravel()
        v = H.ravel()
        ax.scatter(x, y, c=v, s=np.sqrt(v) / scaling,
                   cmap=cmap, alpha=1, zorder=1)
        ax.set_xlabel("Value")
        return ax

    # @safe_return(None)
    def features_bins(self, results=None, ax=None, title=None, scaling=30,
                      cmap="coolwarm"):
        if results is None:
            results = self.load_diagnosis_pickle("features_bins.pkl")
        ax = self._bins(results, ax=ax, scaling=scaling, cmap=cmap)
        ax.set_xlabel("Features (sorted)")
        if title is None:
            title = "Values by feature"
        ax.set_title(title)

    # @safe_return(None)
    def keys_bins(self, results=None, ax=None, title=None, scaling=30,
                  cmap="coolwarm"):
        if results is None:
            results = self.load_diagnosis_pickle("keys_bins.pkl")
        ax = self._bins(results, ax=ax, scaling=scaling, cmap=cmap)
        ax.set_xlabel("Keys (sorted)")
        if title is None:
            title = "Values by key"
        ax.set_title(title)

    def _across(self, values, datasets, ax, title, exemplary, cctype, molset,
                vertical=False, numeral_marker=True):
        from matplotlib.path import Path
        from matplotlib.textpath import TextPath
        ax = self._get_ax(ax)
        datasets = np.array(datasets)
        values = np.array(values)
        idxs = np.array(list(pd.DataFrame(
            {"ds": datasets, "vl": -values}).sort_values(["vl", "ds"]).index)
        ).astype(int)
        datasets = datasets[idxs]
        values = values[idxs]
        colors = np.array([coord_color(ds) for ds in datasets])
        x = np.array([i + 1 for i in range(0, len(values))])
        if numeral_marker:
            top_pad = 0.03
            if vertical:
                for num in ['1', '2', '3', '4', '5']:
                    fp = FontProperties(family="monospace",
                                        weight='bold',
                                        style='normal', size=16)
                    path_num = TextPath((0, 0), num, size=50, prop=fp,)
                    path_num_centered = Path(path_num.vertices
                                             - path_num.vertices.mean(axis=0))
                    mask = np.array([ds[1] == num for ds in datasets])
                    ax.scatter(values[mask]+top_pad, x[mask], color=colors[mask],
                               marker=path_num_centered,
                               linewidths=0.02, edgecolors='k', zorder=2)
            else:
                for num in ['1', '2', '3', '4', '5']:
                    fp = FontProperties(family="monospace",
                                        weight='bold',
                                        style='normal', size=16)
                    path_num = TextPath((0, 0), num, size=50, prop=fp)
                    path_num_centered = Path(path_num.vertices
                                             - path_num.vertices.mean(axis=0))
                    mask = np.array([ds[1] == num for ds in datasets])
                    ax.scatter(x[mask], values[mask]+top_pad, color=colors[mask],
                               marker=path_num_centered,
                               linewidths=0.04, edgecolors='k', zorder=2)
        else:
            if vertical:
                ax.scatter(values, x, color=colors, zorder=1)
            else:
                ax.scatter(x, values, color=colors, zorder=1)
        for i, x_ in enumerate(x):
            if vertical:
                ax.plot([-1, values[i]], [x_, x_], color=colors[i], zorder=1)
            else:
                ax.plot([x_, x_], [-1, values[i]], color=colors[i], zorder=1)
        if title is None:
            title = "%s | %s_%s" % (
                self.diag.sign.qualified_name, cctype, molset)
        ax.set_title(title)
        if vertical:
            ax.set_yticks(x)
            ax.set_yticklabels([ds[1] for ds in datasets])
            ax.set_ylabel("Datasets")
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([ds[1] for ds in datasets])
            ax.set_xlabel("Datasets")
        return ax

    # @safe_return(None)
    def across_coverage(self, results=None, ax=None, title=None, exemplary=True,
                        cctype="sign1", molset="full", vs=True, numeral_marker=False):
        if results is None:
            results = self.load_diagnosis_pickle("across_coverage.pkl")
        datasets = []
        covs = []
        if vs:
            pref = "vs"
        else:
            pref = "my"
        for k, v in results.items():
            datasets += [k]
            covs += [v["%s_overlap" % pref]]
        ax = self._across(covs, datasets, ax=ax, title=title,
                          exemplary=exemplary, cctype=cctype, molset=molset,
                          numeral_marker=numeral_marker)
        ax.set_ylabel("Coverage")
        if vs:
            ax.set_ylim(-np.max(covs) * 0.05,
                        np.min([1.05, np.max(covs) * 1.1]))
        else:
            ax.set_ylim(-0.05, 1.05)
        if title is None:
            if vs:
                title = "CC wrt Sign"
            else:
                title = "Sign wrt CC"
        ax.set_title(title)

    # @safe_return(None)
    def across_roc(self, results=None, ax=None, title=None, exemplary=True,
                   cctype="sign1", molset="full", vertical=False,
                   numeral_marker=False):
        if results is None:
            results = self.load_diagnosis_pickle("across_roc.pkl")
        datasets = []
        rocs = []
        for k, v in results.items():
            if v is None:
                continue
            datasets += [k]
            rocs += [v["auc"]]
        ax = self._across(rocs, datasets, ax=ax, title=title,
                          exemplary=exemplary, cctype=cctype, molset=molset,
                          vertical=vertical, numeral_marker=numeral_marker)
        if vertical:
            ax.set_xlabel("ROC-AUC")
            ax.set_xlim(0.45, 1.05)
        else:
            ax.set_ylabel("ROC-AUC")
            ax.set_ylim(0.45, 1.05)
        if title is None:
            title = "ROC across CC"
        ax.set_title(title)

    # @safe_return(None)
    def dimensions(self, results=None, ax=None, title=None, exemplary=True,
                   cctype="sign1", molset="full", highligth=True):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("dimensions.pkl")
        datasets = []
        colors = []
        x = []
        y = []
        for k, v in results.items():
            if k == "MY":
                continue
            datasets += [k]
            colors += [coord_color(k)]
            y += [v["keys"]]
            x += [v["features"]]
        x = np.log10(x)
        y = np.log10(y)
        ax.scatter(x, y, color=colors)
        # max_x = np.max(x)
        # max_y = np.max(y)
        v = results["MY"]
        y = [v["keys"]]
        x = [v["features"]]
        x = np.log10(x)
        y = np.log10(y)
        if highligth:
            ax.scatter(x, y, color="white", edgecolor="black", s=80)
        ax.set_xlabel("Features (log10)")
        ax.set_ylabel("Keys (log10)")
        if title is None:
            title = "Keys: %d / Feat: %d" % (
                v["keys"], v["features"])
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def redundancy(self, results=None, ax=None, title=None):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("redundancy.pkl")
        counts = results["counts"]
        x = [i + 1 for i in range(0, len(counts))]
        y = [np.log10(c[1]) for c in counts]
        ax.scatter(x, y, c=y, cmap="Spectral", s=10, zorder=100)
        if title is None:
            title = "Redund. (%.1f%%)" % (
                100 - results["n_ref"] / results["n_full"] * 100)
        yticks = sorted(set(np.array(ax.get_yticks(), int)))
        if len(yticks) == 1:
            yticks = [0, 1]
        ax.set_yticks(yticks)
        ax.set_xlabel("Non-red. keys")
        ax.set_ylabel("Red. keys (log10)")
        ax.set_title(title)
        ax.set_ylim(-0.1, max(1, np.max(y)) + 0.1)
        return ax

    # @safe_return(None)
    def cluster_sizes(self, results=None, ax=None, max_clusters=20, s=5,
                      show_outliers=False, title=None):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("clusters.pkl")
        y = np.array([r[1] for r in results["lab_counts"]
                      if r[0] != -1]) / results["P"].shape[0]
        y = np.cumsum(y)
        x = [i + 1 for i in range(0, len(y))]
        xticks = [1, max_clusters, max_clusters * 2]
        xticklabels = [1, max_clusters, len(x)]
        # plot first part
        y_ = y[:max_clusters]
        x_ = x[:max_clusters]
        colors = self._categorical_colors(len(x_))
        ax.scatter(x_, y_, color=colors, zorder=100, s=s)
        if len(x) > max_clusters:
            ax.axvline(max_clusters, color="gray", lw=1, linestyle="--")
            # plot second part
            y_ = y[max_clusters:]
            if len(y_) > 0:
                xmax = max_clusters * 2
                x_ = list(np.linspace(max_clusters +
                                      1 / len(y_), xmax, len(y_)))
                ax.plot(x_, y_, color="gray", zorder=10)
            if show_outliers:
                ax.plot([xmax, xmax], [np.max(y), 1], lw=1, color="red")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Clusters")
        ax.set_ylabel("Prop. of keys")
        if title is None:
            title = "Cluster sizes (%d)" % results["n_clusters"]
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def clusters_projection(self, results=None, ax=None, max_clusters=20, s=1,
                            show_beyond=False, show_outliers=False,
                            title=None):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("clusters.pkl")
        P = results["P"]
        labels = results["labels"]
        labs = [r[0] for r in results["lab_counts"] if r[0] != -1]
        labs_ = labs[:max_clusters]
        colors = self._categorical_colors(len(labs_))
        for lab, col in zip(labs_, colors):
            mask = labels == lab
            ax.scatter(P[mask, 0], P[mask, 1], color=col, s=s, zorder=3)
        if show_beyond:
            for lab in labs:
                if lab in labs_:
                    continue
                mask = labels == lab
                ax.scatter(P[mask, 0], P[mask, 1], color="gray", s=s, zorder=2)
            if show_outliers:
                mask = labels == -1
                ax.scatter(P[mask, 0], P[mask, 1], color="red", s=s, zorder=1)
        xlim, ylim = self._proj_lims(P)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        homogenous_ticks(ax, 2)
        ax.set_ylabel("t-SNE 2")
        ax.set_xlabel("t-SNE 1")
        if title is None:
            title = "Top clusters"
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def intensities(self, results=None, ax=None, s=1, title=None,
                    cmap="Spectral"):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("intensities.pkl")
        x = results["x"]
        y = results["y"]
        vmin = np.min(x)
        vmax = np.max(x)
        pad = (vmax - vmin) * pad_factor
        ax.scatter(x, y, c=x, cmap=cmap, s=s, vmin=vmin + pad, vmax=vmax - pad)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
        if title is None:
            title = "Intensities"
        ax.set_title(title)
        ax.set_ylim(0, np.max(y) * 1.05)
        ax.set_yticklabels([])
        return ax

    # @safe_return(None)
    def confidences(self, results=None, ax=None, s=1, title=None,
                    cmap="Spectral"):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("confidences.pkl")
        x = results["x"]
        y = results["y"]
        vmin = np.min(x)
        vmax = np.max(x)
        pad = (vmax - vmin) * pad_factor
        ax.scatter(x, y, c=x, cmap=cmap, s=s, vmin=vmin + pad, vmax=vmax - pad)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        if title is None:
            title = "Confidence"
        ax.set_title(title)
        ax.set_ylim(0, np.max(y) * 1.05)
        return ax

    def _binned_projection(self, ax, results, cmap, s, vmin=None, vmax=None):
        ax = self._get_ax(ax)
        H = results["H"]
        S = results["S"]
        bins_x = results["bins_x"]
        bins_y = results["bins_y"]
        # cmap
        scores = results["scores"]
        if vmin is None:
            vmin = np.min(scores)
        if vmax is None:
            vmax = np.max(scores)
        pad = (vmax - vmin) * pad_factor
        norm = mpl.colors.Normalize(
            vmin=vmin + pad_factor, vmax=vmax - pad_factor)
        cmap = cm.get_cmap(cmap)
        x = []
        y = []
        z = []
        v = []
        for j in range(0, len(bins_x)):
            for i in range(0, len(bins_y)):
                if H[i, j] == 0:
                    continue
                x += [bins_x[j]]
                y += [bins_y[i]]
                z += [S[i, j]]
                v += [H[i, j]]
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        v = np.array(v)
        idxs = np.argsort(-v)
        x = x[idxs]
        y = y[idxs]
        z = z[idxs]
        v = v[idxs]
        colors = cmap(norm(z))
        v = v / np.max(v)
        #ax.scatter(x,y,color=colors, s=np.sqrt(v)*s)
        ax.scatter(x, y, c=z, cmap=cmap, s=np.sqrt(v) * s)
        xlim, ylim = self._proj_lims(results["lims"])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        homogenous_ticks(ax, 2)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        return ax

    # @safe_return(None)
    def intensities_projection(self, results=None, ax=None, s=10, title=None,
                               cmap="Spectral"):
        if results is None:
            results = self.load_diagnosis_pickle("intensities_projection.pkl")
        ax = self._binned_projection(ax, results, cmap, s)
        if title is None:
            title = "Intensities"
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def confidences_projection(self, results=None, ax=None, s=10, title=None,
                               cmap="Spectral"):
        if results is None:
            results = self.load_diagnosis_pickle("confidences_projection.pkl")
        ax = self._binned_projection(ax, results, cmap, s)
        if title is None:
            title = "Confidence"
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def key_coverage(self, results=None, ax=None, exemplary=True, s=5,
                     title=None, cmap="Spectral"):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("key_coverage.pkl")
        counts = collections.defaultdict(int)
        maxv = 0
        for k, v in results["counts"].items():
            counts[v] += 1
            maxv = np.max([maxv, v])
        if exemplary:
            maxv = 25
        x = np.arange(0, maxv + 1)
        y = np.array([counts[x_] for x_ in x])
        y = y / np.sum(y)
        vmin = 0
        vmax = np.max(x)
        pad = (vmax - vmin) * pad_factor
        norm = mpl.colors.Normalize(vmin=vmin + pad, vmax=vmax - pad)
        cmap = cm.get_cmap(cmap)
        colors = cmap(norm(x))
        ax.scatter(x, y, color=colors, s=5)
        for i, x_ in enumerate(x):
            y_ = y[i]
            ax.plot([x_, x_], [0, y_], color=colors[i])
        ax.set_ylabel("Prop. keys")
        ax.set_xlabel("Datasets")
        if title is None:
            title = "Key coverage"
        ax.set_title(title)
        ax.set_ylim(0, np.max(y) * 1.05)
        return ax

    # @safe_return(None)
    def key_coverage_projection(self, results=None,  ax=None, exemplary=True,
                                s=10, title=None, cmap="coolwarm"):
        if results is None:
            results = self.load_diagnosis_pickle("key_coverage_projection.pkl")
        if exemplary:
            vmin = 0
            vmax = 25
        else:
            vmin = None
            vmax = None
        ax = self._binned_projection(
            ax, results, cmap, s, vmin=vmin, vmax=vmax)
        if title is None:
            title = "Key coverage"
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def ranks_agreement(self, results=None, ax=None, stat="mean", s=1,
                        title=None, exemplary=True, cctype="sign1",
                        molset="full", cmap="Spectral"):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("ranks_agreement.pkl")
        scores = results[stat]
        scores = scores[~np.isnan(scores)]
        kernel = gaussian_kde(scores)
        x = np.linspace(np.min(scores), np.max(scores), 1000)
        y = kernel(x)
        vmin = np.min(x)
        vmax = np.max(x)
        pad = (vmax - vmin) * pad_factor
        ax.scatter(x, y, c=x, cmap=cmap, s=s, vmin=vmin + pad, vmax=vmax - pad)
        ax.set_xlabel("RBO")
        ax.set_ylabel("Density")
        if title is None:
            title = "CC ranks agree."
        ax.set_title(title)
        ax.set_ylim(0, np.max(y) * 1.05)
        ax.set_yticklabels([])
        return ax

    # @safe_return(None)
    def ranks_agreement_projection(self, results=None, ax=None, s=10,
                                   title=None, cmap="Spectral"):
        if results is None:
            results = self.load_diagnosis_pickle(
                "ranks_agreement_projection.pkl")
        ax = self._binned_projection(ax, results, cmap, s)
        if title is None:
            title = "CC ranks agree."
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def global_ranks_agreement(self, results=None, ax=None, stat="mean", s=1,
                               title=None, cmap="Spectral"):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("global_ranks_agreement.pkl")
        scores = results[stat]
        scores = scores[~np.isnan(scores)]
        kernel = gaussian_kde(scores)
        x = np.linspace(np.min(scores), np.max(scores), 1000)
        y = kernel(x)
        vmin = np.min(x)
        vmax = np.max(x)
        pad = (vmax - vmin) * pad_factor
        ax.scatter(x, y, c=x, cmap=cmap, s=s, vmin=vmin + pad, vmax=vmax - pad)
        ax.set_xlabel("RBO")
        ax.set_ylabel("Density")
        if title is None:
            title = "CC ranks agree."
        ax.set_title(title)
        ax.set_ylim(0, np.max(y) * 1.05)
        ax.set_yticklabels([])
        return ax

    # @safe_return(None)
    def global_ranks_agreement_projection(self, results=None, ax=None, s=10,
                                          title=None, cmap="Spectral"):
        if results is None:
            results = self.load_diagnosis_pickle(
                "global_ranks_agreement_projection.pkl")
        ax = self._binned_projection(ax, results, cmap, s)
        if title is None:
            title = "CC ranks agree."
        ax.set_title(title)
        return ax

    # @safe_return(None)
    def orthogonality(self, results=None, ax=None, title=None, s=1,
                      cmap="coolwarm"):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("orthogonality.pkl")
        scores = results["dots"]
        kernel = gaussian_kde(scores)
        x = np.linspace(np.min(scores), np.max(scores), 1000)
        y = kernel(x)
        ax.scatter(x, y, c=x, cmap=cmap, s=s, vmin=-1, vmax=1)
        ax.set_xlabel("Dot product")
        ax.set_ylabel("Density")
        if title is None:
            title = "Orthogonality"
        ax.set_title(title)
        ax.set_ylim(0, np.max(y) * 1.05)
        ax.set_xlim(-1.05, 1.05)
        ax.set_yticklabels([])
        return ax

    # @safe_return(None)
    def outliers(self, results=None, ax=None, title=None, s=1,
                 cmap="coolwarm_r"):
        ax = self._get_ax(ax)
        if results is None:
            results = self.load_diagnosis_pickle("outliers.pkl")
        scs = -results["scores"]
        pds = results["pred"]
        idxs = np.argsort(scs)
        scs = scs[idxs]
        pds = pds[idxs]
        x = [i + 1 for i in range(0, len(scs))]

        norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5)
        cmap = cm.get_cmap(cmap)
        colors = cmap(norm(np.clip(scs, -0.5, 0.5)))
        ax.scatter(x=x, y=scs, color=colors, s=s)
        if title is None:
            title = "Outliers"
        ax.set_title(title)
        ax.set_xlabel("Keys")
        ax.set_ylabel("Anomaly score")
        ax.axhline(0, color="gray", linestyle="--")
        xlim = ax.get_xlim()
        ax.set_xlim(xlim)
        return ax

    def legend(self, ax=None, s=10):
        ax = self._get_ax(ax)
        colors = [coord_color(x) for x in "ABCDE"]
        ax.scatter([0] * 5, [1, 2, 3, 4, 5], color=colors)
        R = [("A", "Chemistry"),
             ("B", "Targets"),
             ("C", "Networks"),
             ("D", "Cells"),
             ("E", "Clinics")]
        for i, r in enumerate(R):
            ax.text(0.1, i + 1, s="%s: %s" % r, va="center")
        ax.set_axis_off()
        ax.set_ylim(6, 0)
        ax.set_xlim(-0.1, 1)
        ax.set_title("CC levels")
        return ax

    def canvas_small(self, title, skip_plots):
        fig = plt.figure(figsize=(14, 4))
        gs = fig.add_gridspec(2, 6, wspace=0.6, hspace=0.6)
        if "image" not in skip_plots:
            self.image(ax=fig.add_subplot(gs[0, :2]))
        if "features_bins" not in skip_plots:
            self.features_bins(ax=fig.add_subplot(gs[1, 0]))
        if "keys_bins" not in skip_plots:
            self.keys_bins(ax=fig.add_subplot(gs[1, 1]))
        if "values" not in skip_plots:
            self.values(ax=fig.add_subplot(gs[0, 2]))
        if "redundancy" not in skip_plots:
            self.redundancy(ax=fig.add_subplot(gs[1, 2]))
        if "projection" not in skip_plots:
            self.projection(ax=fig.add_subplot(gs[0:2, 3:5]))
        if "euclidean_distances" not in skip_plots:
            self.euclidean_distances(ax=fig.add_subplot(gs[0, 5]))
        if "cosine_distances" not in skip_plots:
            self.cosine_distances(ax=fig.add_subplot(gs[1, 5]))

        if title is None:
            title = "%s %s" % (self.diag.sign.dataset, self.diag.sign.cctype)
        fig.suptitle(title, fontweight="bold", y=1, size='xx-large')
        plt.close()
        return fig

    def canvas_medium(self, title, skip_plots):
        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(6, 6, wspace=0.6, hspace=0.6)
        ax = fig.add_subplot(gs[0, 0])
        self.legend(ax=ax)
        if "redundancy" not in skip_plots:
            ax = fig.add_subplot(gs[1, 5])
            self.redundancy(ax=ax)
        if "outliers" not in skip_plots:
            ax = fig.add_subplot(gs[0, 5])
            self.outliers(ax=ax)
        if "values" not in skip_plots:
            ax = fig.add_subplot(gs[1, 0])
            self.values(ax=ax)
        if "confidences" or "intensities" not in skip_plots:
            ax = fig.add_subplot(gs[0, 1])
            if self.diag.sign.cctype == 'sign3':
                if "confidences" not in skip_plots:
                    self.confidences(ax=ax)
                    if "confidences_projection" not in skip_plots:
                        ax = fig.add_subplot(gs[0, 2])
                        self.confidences_projection(ax=ax)
            else:
                if "intensities" not in skip_plots:
                    self.intensities(ax=ax)
                    if "intensities_projection" not in skip_plots:
                        ax = fig.add_subplot(gs[0, 2])
                        self.intensities_projection(ax=ax)
        if "key_coverage" not in skip_plots:
            ax = fig.add_subplot(gs[1, 1])
            self.key_coverage(ax=ax)
            if "key_coverage_projection" not in skip_plots:
                ax = fig.add_subplot(gs[1, 2])
                self.key_coverage_projection(ax=ax)
        if "global_ranks_agreement" not in skip_plots:
            ax = fig.add_subplot(gs[0, 4])
            self.global_ranks_agreement(ax=ax)
            if "global_ranks_agreement_projection" not in skip_plots:
                ax = fig.add_subplot(gs[0, 3])
                self.global_ranks_agreement_projection(ax=ax)
        if "cluster_sizes" not in skip_plots:
            ax = fig.add_subplot(gs[1, 4])
            self.cluster_sizes(ax=ax)
            if "clusters_projection" not in skip_plots:
                ax = fig.add_subplot(gs[1, 3])
                self.clusters_projection(ax=ax)
        if "euclidean_distances" not in skip_plots:
            ax = fig.add_subplot(gs[2, 4])
            self.euclidean_distances(ax=ax)
        if "cosine_distances" not in skip_plots:
            ax = fig.add_subplot(gs[2, 5])
            self.cosine_distances(ax=ax)
        if "features_bins" not in skip_plots:
            ax = fig.add_subplot(gs[3, 0])
            self.features_bins(ax=ax)
        if "keys_bins" not in skip_plots:
            ax = fig.add_subplot(gs[3, 1])
            self.keys_bins(ax=ax)
        if "projection" not in skip_plots:
            ax = fig.add_subplot(gs[2:4, 2:4])
            self.projection(ax=ax)
        if "image" not in skip_plots:
            ax = fig.add_subplot(gs[2, :2])
            self.image(ax=ax)
        if "moa_roc" not in skip_plots:
            ax = fig.add_subplot(gs[3, 4])
            self.moa_roc(ax=ax)
        if "atc_roc" not in skip_plots:
            ax = fig.add_subplot(gs[3, 5])
            self.atc_roc(ax=ax)
        if "dimensions" not in skip_plots:
            ax = fig.add_subplot(gs[-2:, :2])
            self.dimensions(ax=ax)
        if "across_coverage" not in skip_plots:
            ax = fig.add_subplot(gs[-2, 2:4])
            self.across_coverage(ax=ax, vs=True)
            ax = fig.add_subplot(gs[-1, 2:4])
            self.across_coverage(ax=ax, vs=False)
        if "across_roc" not in skip_plots:
            ax = fig.add_subplot(gs[-2:, 4:6])
            self.across_roc(ax=ax)
        if title is None:
            title = "%s %s" % (self.diag.sign.dataset, self.diag.sign.cctype)
        fig.suptitle(title, fontweight="bold", y=0.92, size='xx-large')
        plt.close()
        return fig

    def canvas_large(self, title, skip_plots):
        pass

    def custom_comparative_vertical(self, title, skip_plots):
        fig = plt.figure(figsize=(9, 9))
        gs = fig.add_gridspec(4, 4, wspace=0.6, hspace=0.6)
        #plt.figtext(0.02, 0.5, 'TEST', fontsize=12, fontweight="bold")
        ax = fig.add_subplot(gs[0:2, 0:2])
        self.projection(ax=ax)
        #text = ax.text(-55,45,"A)",fontsize=12, fontweight="bold")
        ax = fig.add_subplot(gs[2, 1])
        self.legend(ax=ax)
        ax = fig.add_subplot(gs[0, 2])
        if self.diag.sign.cctype == 'sign3':
            self.confidences(ax=ax)
        else:
            self.intensities(ax=ax)
        #text1 = ax.text(-1,1,"B)",fontsize=12, fontweight="bold")
        ax = fig.add_subplot(gs[0, 3])
        self.clusters_projection(ax=ax)
        #text1 = ax.text(-50,-50,"C)",fontsize=12, fontweight="bold")
        ax = fig.add_subplot(gs[1, 2])
        self.moa_roc(ax=ax)
        ax = fig.add_subplot(gs[1, 3])
        self.atc_roc(ax=ax)
        ax = fig.add_subplot(gs[2:4, 2:4])
        self.across_roc(ax=ax)
        if title is None:
            title = "%s %s" % (self.diag.sign.dataset, self.diag.sign.cctype)
        fig.suptitle(title, fontweight="bold", y=0.95, size='xx-large')
        plt.close()
        return fig

    def canvas(self, size="medium", title=None, skip_plots=[]):
        self.__log.debug("Plotting Canvas %s" % size)
        if size == "small":
            return self.canvas_small(title=title, skip_plots=skip_plots)
        elif size == "medium":
            return self.canvas_medium(title=title, skip_plots=skip_plots)
        elif size == "large":
            return self.canvas_large(title=title, skip_plots=skip_plots)
        elif size == "compare_v":
            return self.custom_comparative_vertical(title=title, skip_plots=skip_plots)
        else:
            return None
