"""Utility for plotting the typical CC plots"""
import os
import pandas as pd
from chemicalchecker.util import logged
import matplotlib.pyplot as plt
import seaborn as sns
from chemicalchecker.util.plot.style.util import coord_color, set_style
import random
import numpy as np

set_style()


@logged
class DefaultPlot(object):

    def __init__(self, cc):
        self.cc = cc
    
    @staticmethod
    def _get_ax(ax):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        return ax

    def validation_auc(self, cctype, dataset_code, molset="full", valtype="moa", ax=None, title=None, color=None):
        """
        ROC curve
        """
        from sklearn.metrics import auc
        ax = self._get_ax(ax)
        path = os.path.join(self.cc.get_signature_path(cctype, molset, dataset_code), "stats")
        path = os.path.join(path, "%s_%s_auc_validation.tsv" % (valtype, cctype))
        df  = pd.read_csv(path, header=None, names=["fpr", "tpr"], delimiter="\t")
        step = 0.001
        fpr = np.arange(0, 1+step, step)
        tpr = np.interp(fpr, df["fpr"], df["tpr"])
        auc_ = auc(fpr, tpr)
        if color is None:
            color = coord_color(dataset_code)
        ax.plot(fpr, tpr, color=color)
        ax.plot([0,1],[0,1], color="gray", linestyle="--")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        if title is None:
            title = "%s %s %s (%.3f)" % (dataset_code, valtype.upper(), cctype, auc_)
        ax.set_title(title)
        plt.savefig("test.png")

    def image(self, cctype, dataset_code, molset="full", keys=None, max=1000, ax=None):
        """

        """
        sign = self.cc_get_signature(cctype, molset, dataset_code)
        ax.imshow(V, cmap="Spectral", aspect="auto")

    def projection(self, cctype, dataset_code, molset="full", keys=None, max=10000, density=True, ax=None, title=None):
        ax = self._get_ax(ax)
        sign = self.cc.get_signature(cctype, molset, dataset_code)
        if keys is None:
            keys = sign.keys
        else:
            keys = list(set(keys).intersection(sign.keys))
            self.__log.debug("%d keys found" % len(keys))
        keys = sorted(random.sample(keys, np.min([max, len(keys)])))
        if "sign" in cctype:
            from sklearn.manifold import TSNE
            self.__log.info("A multidimensional signature was specified. TSNE will be performed (default parameters).")
            X = sign.get_vectors(keys)[1]
            self.__log.debug("Fitting t-SNE")
            tsne = TSNE()
            P = tsne.fit_transform(X)
        elif "proj" in cctype:
            self.__log.info("A 2D signature (projection) was already spacified.")
            P = sign.get_vectors(keys)
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
            ax.scatter(x, y, s=10)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        if title is None:
            title = "%s %s (%d)" % (dataset_code, cctype, len(keys))
        ax.set_title(title)
        plt.savefig("test.png")